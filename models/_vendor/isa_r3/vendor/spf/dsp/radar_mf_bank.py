"""Doppler matched-filter bank — the regime where SPFTensor *should* pay off.

A moving target has delay ``d`` and Doppler ``f``: ``echo[n] = a·s[n−d]·e^(j2πfn)``.
The bank tests a grid of Doppler hypotheses; the (delay, Doppler) cell that peaks is
the detection (the ambiguity surface). This is the multi-hypothesis regime where
SPF's O(1) phase ops and SPFTensor's operator/reducer surface are supposed to shine.

Three engines, measured against each other (see ``scripts/run_radar_mf_bank.py``):

- ``naive``     — Doppler-compensate the echo with a phase ramp and **FFT per
  hypothesis** (no amortization): cost ∝ ``n_hyp`` FFTs.
- ``flat``      — **one** ``FFT(echo)``; each Doppler is a spectral roll (the
  modulation↔shift identity), then ``spf_mul_words`` by the precomputed reference,
  then phase-slope range. Cost ∝ 1 FFT + ``n_hyp`` O(1) multiplies.
- ``spftensor`` — same as ``flat`` but the reference multiply is a
  :class:`DiagonalOperator` (``SOV_DIAG``) and the per-cell peak uses the AUTO
  domain-routed :func:`reduce_words` (TROPICAL for a dominant peak).

Honest finding baked into the design: ``DiagonalOperator.apply_words`` is exactly
``spf_mul_words`` (it dispatches ``spf_mul_planes`` against memoised weight planes),
so ``spftensor`` is idiom over the same arithmetic as ``flat`` and runs at parity once
the operator's materialisation is cached — the real win is the *algorithm* (amortize
the FFT, Doppler-as-roll, O(1) multiplies), all plain SPF hotpath. SPFTensor adds
composability and a (lossy) analytic-storage option (:func:`chirp_v2_block`), not CPU
speed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from spf.dsp.kernels import phase_ramp_apply
from spf.dsp.radar_mf import phase_slope_delay, spf_matched_filter_reference
from spf.dsp.radar_mf_sim import RadarScene
from spf.fft import spf32_fft
from spf.tensor import encode_words
from spf.tensor.core import (
    ActiveEncoding,
    AffineTensorBlock,
    DiagonalOperator,
    OperatorHeader,
    Profile,
    ReducerPolicy,
    decode_complex,
    reduce_words,
    spf_mul_words,
    unpack_L,
    unpack_theta,
)

_PHASE_MOD = 65536
BankEngine = str  # "naive" | "flat" | "spftensor"


def synthesize_moving_target(
    scene: RadarScene, *, delay_bin: int, doppler_bin: int, snr_db: float | None, seed: int
) -> RadarScene:
    """Return a copy of ``scene`` whose echo is a chirp at ``delay_bin`` modulated by
    Doppler ``doppler_bin`` (``echo[n] = s[n−d]·e^(j2π·m·n/N)``) + AWGN."""
    import dataclasses

    cfg = scene.config
    n, ns = cfg.n, cfg.chirp_samples
    chirp = scene.reference_chirp[:ns]
    echo = np.zeros(n, dtype=np.complex128)
    echo[delay_bin : delay_bin + ns] += chirp
    echo = echo * np.exp(1j * 2 * np.pi * doppler_bin / n * np.arange(n))
    if snr_db is not None:
        mean_p = float(np.mean(np.abs(chirp) ** 2))
        sigma = float(np.sqrt(mean_p / (10.0 ** (snr_db / 10.0))))
        rng = np.random.default_rng(seed)
        echo = echo + (sigma / np.sqrt(2.0)) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return dataclasses.replace(scene, echo_samples=echo)


def doppler_bin_shifts(n_hyp: int, step: int = 1) -> list[int]:
    """Centered integer Doppler-bin hypotheses, e.g. n_hyp=5 -> [-2,-1,0,1,2]·step."""
    half = n_hyp // 2
    return [(i - half) * step for i in range(n_hyp)]


def build_reference_diagonal(conj_h0_words: npt.NDArray[np.uint32]) -> DiagonalOperator:
    """Wrap a conjugate-spectrum reference as a SOV-DIAG :class:`DiagonalOperator`.

    Dense, all-residual (the LFM reference spectrum is not affine, so the header
    ramps stay zero and the words live in the residual planes). ``apply_words`` is
    then bit-identical to :func:`spf_mul_words` (zero sentinels included), and the
    operator memoises its materialised planes, so a per-hypothesis bank loop runs at
    ``spf_mul_words`` speed. The operator is idiom, not speedup.
    """
    words = np.asarray(conj_h0_words, dtype=np.uint32).ravel()
    header = OperatorHeader(
        profile_id=Profile.SOV_DIAG, reducer_id=ReducerPolicy.NONE, dims=(int(words.size),)
    )
    block = AffineTensorBlock(
        header=header,
        active_encoding=ActiveEncoding.DENSE_ALL,
        dL=np.asarray(unpack_L(words), dtype=np.int16),
        dT=np.asarray(unpack_theta(words), dtype=np.int16),
    )
    return DiagonalOperator(block)


@dataclass(frozen=True, slots=True)
class BankCell:
    """One Doppler hypothesis: its bin shift, estimated delay, slope confidence."""

    doppler_bin: int
    delay_est: int
    confidence: int


@dataclass(frozen=True, slots=True)
class BankResult:
    engine: BankEngine
    cells: tuple[BankCell, ...]
    n_ffts: int  # forward FFTs actually performed (the cost knob)

    @property
    def detection(self) -> BankCell:
        """Highest-confidence cell = the (delay, Doppler) detection."""
        return max(self.cells, key=lambda c: c.confidence)

    def to_dict(self) -> dict[str, object]:
        det = self.detection
        return {
            "engine": self.engine,
            "n_ffts": self.n_ffts,
            "detected_doppler_bin": det.doppler_bin,
            "detected_delay": det.delay_est,
            "detected_confidence": det.confidence,
            "n_hypotheses": len(self.cells),
        }


def run_doppler_bank(
    scene: RadarScene,
    shifts: list[int],
    *,
    engine: BankEngine = "flat",
    h_conj_fft: npt.NDArray[np.uint32] | None = None,
) -> BankResult:
    """Run the Doppler matched-filter bank with the chosen engine (see module docstring)."""
    n = scene.config.n
    echo_words = encode_words(scene.echo_samples)
    h0 = h_conj_fft if h_conj_fft is not None else spf_matched_filter_reference(scene.reference_chirp)

    if engine == "naive":
        cells: list[BankCell] = []
        n_ffts = 0
        for k in shifts:
            # Doppler-compensate echo by a phase ramp, then FFT per hypothesis.
            dtheta = (-k * _PHASE_MOD // n) % _PHASE_MOD
            echo_k = phase_ramp_apply(echo_words, 0, dtheta)
            x_k = spf32_fft.fft(echo_k)
            n_ffts += 1
            p_k = spf_mul_words(h0, x_k)
            d, conf = phase_slope_delay(p_k, n)
            cells.append(BankCell(k, d, conf))
        return BankResult("naive", tuple(cells), n_ffts)

    # amortized engines: one FFT, Doppler == spectral roll
    x_fft = spf32_fft.fft(echo_words)
    diag = build_reference_diagonal(h0) if engine == "spftensor" else None
    cells = []
    for k in shifts:
        x_k = np.roll(x_fft, -k)  # compensate Doppler hypothesis k (matches naive's e^(-jk))
        if engine == "spftensor":
            assert diag is not None
            p_k = diag.apply_words(x_k)
        else:  # "flat"
            p_k = spf_mul_words(h0, x_k)
        d, conf = phase_slope_delay(p_k, n)
        cells.append(BankCell(k, d, conf))
    return BankResult(engine, tuple(cells), 1)


# --- affine operator fusion (SPFTensor's real payoff) ------------------------
#
# compose_header_only fuses two *affine-ramp* diagonals (DENSE_ALL, no bands, zero
# residuals) into one by adding their header coefficients — O(1) in N, no per-element
# work. Several radar front-end corrections are exactly affine:
#   - bulk range-gate delay        -> e^(-j2πkd/N): integer phase ramp (ramp_alpha)
#   - constant phase calibration   -> global_T_bias
#   - fixed gain / range-law       -> global_L_bias (log2 ticks, Q7.8)
# A whole correction chain therefore fuses into ONE operator applied once. The dense
# matched-filter *reference* is NOT fusible (it lives in residuals) — which is exactly
# why the Doppler bank above stays flat. The payoff is the affine steering layer.


def _affine_diag(
    n: int, *, gain_l: int = 0, delay_bins: int = 0, phase_ticks: int = 0
) -> DiagonalOperator:
    """Pure-affine SOV-DIAG operator (zero residuals -> composable via
    ``compose_header_only``): constant gain ``gain_l`` (log2 Q7.8 ticks), integer-bin
    ``delay`` as a frequency-domain phase ramp ``e^(-j2π k d/N)``, and constant phase
    ``phase_ticks`` (16-bit). Integer-bin delay is exact for power-of-two ``n``;
    sub-bin delay would need a residual (not fusible) — surfaced, not hidden."""
    slope = (-int(delay_bins) * _PHASE_MOD // int(n)) % _PHASE_MOD
    header = OperatorHeader(
        profile_id=Profile.SOV_DIAG,
        reducer_id=ReducerPolicy.NONE,
        dims=(int(n),),
        global_L_bias=int(gain_l),
        global_T_bias=int(phase_ticks) % _PHASE_MOD,
        ramp_alpha=(int(slope),),
    )
    block = AffineTensorBlock(
        header=header,
        active_encoding=ActiveEncoding.DENSE_ALL,
        dL=np.zeros(int(n), dtype=np.int16),
        dT=np.zeros(int(n), dtype=np.int16),
    )
    return DiagonalOperator(block)


def gain_operator(n: int, gain_l: int) -> DiagonalOperator:
    """Constant-gain affine diagonal (``global_L_bias`` only)."""
    return _affine_diag(n, gain_l=gain_l)


def delay_operator(n: int, delay_bins: int) -> DiagonalOperator:
    """Integer-bin delay affine diagonal (``ramp_alpha`` phase ramp only)."""
    return _affine_diag(n, delay_bins=delay_bins)


def phase_operator(n: int, phase_ticks: int) -> DiagonalOperator:
    """Constant-phase affine diagonal (``global_T_bias`` only)."""
    return _affine_diag(n, phase_ticks=phase_ticks)


def fuse_affine(ops: list[DiagonalOperator]) -> DiagonalOperator:
    """Fuse a chain of affine-ramp diagonals into one via ``compose_header_only``
    (header-coefficient adds only — O(len(ops)), independent of N)."""
    fused = ops[0]
    for op in ops[1:]:
        fused = fused.compose_header_only(op)
    return fused


@dataclass(frozen=True, slots=True)
class SteeringFusionReport:
    n_ops: int
    bitexact: bool  # fuse-then-apply-once == apply each in sequence
    fused_applies: int  # 1
    sequential_applies: int  # n_ops


def run_steering_fusion(
    x_words: npt.NDArray[np.uint32], ops: list[DiagonalOperator]
) -> SteeringFusionReport:
    """Apply an affine correction chain two ways and check they agree:

    - sequential: apply each operator in turn (``n_ops`` applies, O(n_ops·N));
    - fused: ``compose_header_only`` the chain into one operator, apply once
      (O(n_ops) header adds + 1 apply, O(n_ops + N)).
    """
    words = np.asarray(x_words, dtype=np.uint32)
    seq = words
    for op in ops:
        seq = op.apply_words(seq)
    fused = fuse_affine(ops).apply_words(words)
    return SteeringFusionReport(
        n_ops=len(ops),
        bitexact=bool(np.array_equal(fused, seq)),
        fused_applies=1,
        sequential_applies=len(ops),
    )


@dataclass(frozen=True, slots=True)
class CorrelatorBankResult:
    delays: tuple[int, ...]
    scores_l: tuple[int, ...]  # log2|coherent sum| per delay (Q7.8), argmax = peak
    detected_delay: int
    steer_ops_built_by_header: int  # operators built by O(1) header composition


def delay_correlator_bank(
    x_fft_words: npt.NDArray[np.uint32],
    h_conj_words: npt.NDArray[np.uint32],
    delays: list[int],
) -> CorrelatorBankResult:
    """Coarse delay search by *steered coherent reduction* — an alternative matched
    filter where the steering bank is pure operator algebra.

    ``base = X·conj(H)`` (dense). The correlation at lag ``d`` is
    ``P[d] = Σ_k base[k]·e^(+j2π k d/N)`` (the IDFT magnitude at lag ``d``): steer
    ``base`` by ``e^(+j2π k d/N)`` then coherently sum. The steering operators are built
    by **incremental ``compose_header_only``** — ``steer(d+step) = steer(d) ∘ steer(step)``,
    one header add per hypothesis — instead of materialising each length-N steering
    vector (O(N) each). ``argmax_d log2|Σ|`` is the delay.
    """
    n = int(np.asarray(x_fft_words).size)
    base = spf_mul_words(h_conj_words, x_fft_words)
    if not delays:
        return CorrelatorBankResult((), (), 0, 0)
    # build the steering bank incrementally: steer(d) applies e^(+j2π k d/N) = delay(-d).
    sorted_delays = sorted(delays)
    steer = {sorted_delays[0]: delay_operator(n, -sorted_delays[0])}
    built = 1
    for prev, cur in pairwise(sorted_delays):
        # one header add: steer(cur) = steer(prev) ∘ delay(-(cur-prev))
        steer[cur] = steer[prev].compose_header_only(delay_operator(n, -(cur - prev)))
        built += 1
    scores: list[tuple[int, int]] = []
    for d in sorted_delays:
        coherent = reduce_words(steer[d].apply_words(base), ReducerPolicy.CARTESIAN_KAHAN)
        scores.append((d, int(unpack_L(np.asarray([coherent], dtype=np.uint32))[0])))
    best = max(scores, key=lambda ds: ds[1])
    return CorrelatorBankResult(
        delays=tuple(d for d, _ in scores),
        scores_l=tuple(s for _, s in scores),
        detected_delay=best[0],
        steer_ops_built_by_header=built,
    )


# --- v2 analytic chirp (storage) ---------------------------------------------


@dataclass(frozen=True, slots=True)
class V2ChirpReport:
    block_bytes: int
    words_bytes: int
    reconstruct_evm: float
    # Quantized phase coefficients actually stored in the v2 header (post odd-nudge
    # on q_b). Exposed so callers can verify they share this analytic-chirp source
    # of truth instead of re-deriving the a/b/d math; see pipelines.lfm_chirp_ramp_coeffs.
    q_a: int = 0
    q_b: int = 0
    q_d: int = 0
    # q_b before the odd-nudge (the value the host-executor ramp path keeps, since it
    # never inverts ramp_R mod 2). Equals q_b when no nudge was applied. Exposed so
    # callers can compare against the pre-nudge value directly rather than reverse-
    # engineering whether the nudge fired from q_b's parity.
    q_b_pre_nudge: int = 0


def chirp_v2_block(
    n: int,
    *,
    sample_rate_hz: float,
    bandwidth_hz: float,
    doppler_bin: int = 0,
) -> V2ChirpReport:
    """Represent a rect LFM chirp (+ optional Doppler) as a v2 quadratic-ramp block.

    Phase ``φ[k] = a·k² + b·k + d`` is exact for a rect LFM; the v2 header stores it
    as ``quadratic_ramp`` (a), ``ramp_R`` (b + Doppler), ``ramp_b`` (d) — a handful of
    coefficients vs ``n`` words. Reports header vs dense byte cost and reconstruction EVM.

    Two caveats the EVM folds in: (1) ``quadratic_ramp`` is *integer* ticks/k², so a
    sub-tick coefficient error grows as k² — the dominant term, measured not assumed;
    (2) ``ramp_R`` must be invertible mod 2, so an even linear coefficient is nudged by
    one tick (≤1 tick/sample of extra linear phase). ``block_bytes`` counts only the
    analytic header coefficients: a rect LFM has *all-zero* residual planes (compressible
    to ~0), though this dense reference still materialises ``n`` zero ``dL``/``dT`` entries.
    """
    from spf.tensor.v2 import TensorBlock, TensorBlockHeaderV2

    ns = n
    fs, b_hz = sample_rate_hz, bandwidth_hz
    c = (ns - 1) / 2.0
    rate = b_hz / (ns / fs)
    k2 = math.pi * rate / fs**2  # rad per k²
    tick = _PHASE_MOD / (2.0 * math.pi)
    q_a = round(k2 * tick)
    q_b = round((-2.0 * c * k2) * tick + doppler_bin * _PHASE_MOD / ns) % _PHASE_MOD
    q_b_pre_nudge = q_b
    if q_b % 2 == 0:
        q_b += 1  # ramp_R must be invertible mod 2 (odd); ≤1 tick/sample linear error
    q_d = round(c * c * k2 * tick) % _PHASE_MOD

    header = TensorBlockHeaderV2(
        dims=(ns,),
        order=1,
        global_L_bias=0,  # |z| = 2^0 = 1 (rect)
        ramp_R=((q_b,),),
        ramp_b=(q_d,),
        quadratic_ramp=(q_a,),
    )
    # zero residual streams (required by the API); a real envelope omits all-zero
    # residuals, so the analytic cost is the header coefficients alone.
    block = TensorBlock(
        header=header, dL=np.zeros(ns, dtype=np.int16), dT=np.zeros(ns, dtype=np.int16)
    )
    # reconstruct() validates the block; decode straight from its planes rather than
    # round-tripping through pack_words -> decode_words (identical result, one less pass).
    rl, rt = block.reconstruct()
    got = decode_complex(rl, rt)

    # analytic reference (rect LFM + Doppler)
    t = (np.arange(ns) - c) / fs
    ref = np.exp(1j * (math.pi * rate * t * t)).astype(np.complex128)
    ref = ref * np.exp(1j * 2 * math.pi * doppler_bin / ns * np.arange(ns))
    evm = float(np.sqrt(np.sum(np.abs(got - ref) ** 2) / np.sum(np.abs(ref) ** 2)))

    # storage: header metadata ints vs dense uint32 words
    meta = header.to_metadata()
    block_bytes = sum(
        (len(v) if isinstance(v, list) else 1) * 4
        for v in meta.values()
        if isinstance(v, (int, list))
    )
    return V2ChirpReport(
        block_bytes=block_bytes,
        words_bytes=ns * 4,
        reconstruct_evm=evm,
        q_a=int(q_a),
        q_b=int(q_b),
        q_d=int(q_d),
        q_b_pre_nudge=int(q_b_pre_nudge),
    )
