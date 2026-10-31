"""Radar range-Doppler and OFDM equalization as spf.tensor ExecutionPlans.

These are the two target pipelines for the spf_tensor_pl overlays, expressed in
the ExecutionPlan vocabulary so the PL lowering (:mod:`spf.tensor.pl_lowering`)
can dispatch them:

* **OFDM equalization** — CFO ``ramp`` → ``fft`` → ``diagonal_scale`` (equalize
  by a precomputed channel inverse). Uses the spf_tensor_ofdm_pl CU set
  {fft64, ramp, mul, div, conj}.
* **Radar pulse compression (matched filter)** — deramp ``ramp`` → range
  ``fft`` → spectral conjugate-``diagonal_scale`` by the matched-filter
  spectrum → ``ifft``. This is the validated radar matched filter (PR #287:
  range-profile EVM 0.4-0.8%, peak bin exact) — a spectral conjugate-MULTIPLY +
  IFFT, which is why the radar overlay carries ``mul`` (not ``reduce``). Power
  (``abs2``) + CFAR are downstream. Uses the spf_tensor_radar_pl CU set
  {fft64, ramp, mul, conj, abs2}. (``spf_reduce`` stays for tensor reductions
  and the future phase-slope IFFT-elimination path — PR #287's only stacking
  FB7-reduction, 2.4x at high SNR.)

Each builder returns ``(plan, inputs, fp64_reference)`` where ``fp64_reference``
executes the *same* ops in complex128 (the decoded-contract oracle), so a test
scores SPF-vs-FP64 for one algorithm — the honest accuracy comparison.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .core import decode_words, encode_words, spf_conj_words
from .v2 import ExecutionPlan, ExecutionStep, _ramp_theta_increment

_TWO_PI = 2.0 * np.pi
_PHASE_MOD = 1 << 16


def _ramp_phase_rad(
    shape: tuple[int, ...],
    ramp_R,
    ramp_b,
    *,
    quadratic_ramp=None,
    nonlinear: bool = False,
) -> np.ndarray:
    """FP64 phase (radians) the ExecutionPlan `ramp` op applies, per element.

    ``nonlinear=True`` folds the ``quadratic_ramp`` term, mirroring the host
    executor's ``_execute_ramp`` (which passes ``nonlinear`` through to
    :func:`_ramp_theta_increment`). Linear-only by default so existing callers
    are unchanged.
    """
    inc = _ramp_theta_increment(
        shape, ramp_R, ramp_b, quadratic_ramp, None, nonlinear=nonlinear
    )
    return inc.astype(np.float64) * (_TWO_PI / _PHASE_MOD)


def ofdm_equalize_plan(
    n_sym: int = 4, n_sub: int = 64, *, seed: int = 0, cfo_word: int = 5
) -> tuple[ExecutionPlan, dict[str, np.ndarray], np.ndarray]:
    """OFDM RX equalization: CFO ramp -> FFT -> diagonal_scale(h_inv).

    Returns ``(plan, inputs, fp64_equalized)``. ``fp64_equalized`` is the
    complex128 result of the same three ops on the decoded inputs.
    """
    rng = np.random.default_rng(seed)
    shape = (n_sym, n_sub)
    x = encode_words(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    # Per-subcarrier channel; equalize by its inverse (precomputed on the host).
    h = rng.uniform(0.6, 1.4, (1, n_sub)) * np.exp(1j * rng.uniform(0, _TWO_PI, (1, n_sub)))
    h_inv = encode_words(np.broadcast_to(1.0 / h, shape).copy())

    ramp_R = [[cfo_word, 0], [0, 0]]  # CFO: phase linear in the symbol index
    ramp_b = [0, 0]
    plan = ExecutionPlan(
        steps=(
            ExecutionStep("ramp", ("x",), ("c",), params={"ramp_R": ramp_R, "ramp_b": ramp_b}),
            ExecutionStep("fft", ("c",), ("X",)),
            ExecutionStep("diagonal_scale", ("X", "h_inv"), ("eq",)),
        )
    )
    inputs = {"x": x, "h_inv": h_inv}

    # FP64 oracle: same ops in complex128.
    xz = decode_words(x) * np.exp(1j * _ramp_phase_rad(shape, ramp_R, ramp_b))
    fp64 = np.fft.fft(xz, axis=-1) * decode_words(h_inv)
    return plan, inputs, fp64


def radar_pulse_compression_plan(
    n_range: int = 64, *, seed: int = 1, deramp_word: int = 3
) -> tuple[ExecutionPlan, dict[str, np.ndarray], np.ndarray]:
    """Radar pulse compression: deramp ramp -> range FFT -> spectral conjugate-
    multiply by the matched-filter spectrum -> IFFT (the validated #287 path).

    The matched-filter coefficients ``conj(fft(replica))`` are precomputed on the
    host and applied as a ``diagonal_scale`` (spectral ``mul``); the IFFT is the
    conj/fft/conj/scale macro. Returns ``(plan, inputs, fp64_range_profile)``.
    """
    rng = np.random.default_rng(seed)
    shape = (n_range,)
    echo = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    replica = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    x = encode_words(echo)
    # Matched-filter spectrum = conj(FFT(replica)); SPF-encoded so both arms use
    # identical coefficients (score against the decoded version, not the fp64 one).
    conj_h = encode_words(np.conj(np.fft.fft(replica)))

    ramp_R = [[deramp_word]]
    ramp_b = [0]
    plan = ExecutionPlan(
        steps=(
            ExecutionStep("ramp", ("x",), ("c",), params={"ramp_R": ramp_R, "ramp_b": ramp_b}),
            ExecutionStep("fft", ("c",), ("X",)),
            ExecutionStep("diagonal_scale", ("X", "conj_h"), ("mf",)),
            ExecutionStep("ifft", ("mf",), ("y",)),
        )
    )
    inputs = {"x": x, "conj_h": conj_h}

    # FP64 oracle: same ops in complex128, using the SPF-encoded matched filter.
    cz = decode_words(x) * np.exp(1j * _ramp_phase_rad(shape, ramp_R, ramp_b))
    mf = np.fft.fft(cz) * decode_words(conj_h)
    fp64 = np.fft.ifft(mf)
    return plan, inputs, fp64


def lfm_chirp_ramp_coeffs(
    n: int, *, sample_rate_hz: float, bandwidth_hz: float, doppler_bin: int = 0
) -> dict[str, object]:
    """Quadratic-ramp ``params`` for a rect LFM (+ optional Doppler) replica.

    Returns the ``ramp`` op params ``{ramp_R, ramp_b, quadratic_ramp, nonlinear}``
    whose per-sample phase ``φ[k] = a·k² + b·k + d`` reproduces the rect LFM
    ``e^{jπ·rate·t²}`` (centered, ``t = (k−c)/fs``). The ``a/b/d`` quantization
    (``round(coeff · 2^W/2π)``) is the same derivation as
    :func:`spf.dsp.radar_mf_bank.chirp_v2_block`.

    **Deliberate divergence from chirp_v2_block:** that block applies an odd-nudge
    to ``q_b`` (``if q_b % 2 == 0: q_b += 1``) because the *v2 block format* requires
    ``ramp_R`` to be invertible mod 2 (≤1 tick/sample of extra linear phase). That is
    a block-format constraint, **not** a general ramp constraint: this pipeline's
    ramp is evaluated directly by the host executor's ``nonlinear=True`` quadratic
    path, which never inverts ``ramp_R`` mod 2. So the nudge is intentionally omitted
    here — adding it would inject a needless ≤1 tick/sample linear error. The a/b/d
    *formulas* match chirp_v2_block; the post-quantization nudge does not (and must
    not) carry over. See the T7.49 ``chirp_coeffs_a_b_d_derivation`` test.

    Exact only for LFM/FMCW: an arbitrary waveform has no quadratic-phase
    structure, so it cannot be expressed this way (use a dense replica instead).
    """
    fs, b_hz = float(sample_rate_hz), float(bandwidth_hz)
    c = (n - 1) / 2.0
    rate = b_hz / (n / fs)
    k2 = math.pi * rate / fs**2  # rad per k²
    tick = _PHASE_MOD / _TWO_PI
    q_a = round(k2 * tick)
    q_b = round((-2.0 * c * k2) * tick + doppler_bin * _PHASE_MOD / n)
    q_d = round(c * c * k2 * tick)
    # order-1 ramp over the single range axis: ramp_R=[[b]], ramp_b=[d], quad=[a].
    return {
        "ramp_R": [[int(q_b) % _PHASE_MOD]],
        "ramp_b": [int(q_d) % _PHASE_MOD],
        "quadratic_ramp": [int(q_a) % _PHASE_MOD],
        "nonlinear": True,
    }


def _unit_words(shape: tuple[int, ...]) -> np.ndarray:
    """SPF words for an all-ones (|z|=1, Θ=0) array — the ramp carrier."""
    return encode_words(np.ones(shape, dtype=np.complex128))


def radar_range_doppler_plan(
    n_range: int = 64,
    n_pulse: int = 16,
    *,
    seed: int = 1,
    waveform: str = "lfm",
    sample_rate_hz: float = 1.0e6,
    bandwidth_hz: float = 4.0e5,
) -> tuple[ExecutionPlan, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Range-Doppler radar map via an SPF-native quadratic-ramp chirp.

    Pipeline (per pulse, then across pulses):

    1. **Deramp** — a ``quadratic_ramp`` ``ramp`` op (``nonlinear=True``), NOT a
       linear ramp. An LFM's phase is exactly ``a·k²+b·k+d``, so the deramp is
       O(1) metadata; coefficients come from :func:`lfm_chirp_ramp_coeffs`
       (same ``a/b/d`` derivation as :func:`chirp_v2_block`, minus that block's
       v2-format odd-nudge on ``q_b`` — see that helper's docstring).
    2. **Range FFT** over the fast-time (range) axis.
    3. **Matched filter** as a *structured* ``diagonal_scale`` by ``conj(fft(replica))``
       where the replica is **generated from the quadratic-ramp metadata** (free
       ``conj`` via :func:`spf_conj_words`), replacing a dense
       ``encode_words(conj(fft(replica)))`` array. Exact for **LFM/FMCW only** —
       an arbitrary waveform has no quadratic-phase structure and must stay dense
       (see ``waveform != "lfm"`` below and the T7.49 test).
    4. **Range IFFT** → range profile per pulse.
    5. **Corner-turn** (``permute``) + **slow-time FFT** over the pulse axis =
       Doppler processing (2nd FFT).

    Returns ``(plan, inputs, fp64_range_doppler, mf_replica_words)``. CFAR /
    detection is host-side at egress (:func:`cfar_detections`), NOT on the
    multiplier-free hot path. ``mf_replica_words`` is the analytic replica the
    structured MF was built from (so a test can compare it to a dense replica).
    """
    if waveform != "lfm":
        raise ValueError(
            "radar_range_doppler_plan: structured quadratic-ramp MF is exact for "
            f"LFM/FMCW only; waveform={waveform!r} has no quadratic-phase structure "
            "— use a dense conj(fft(replica)) matched filter instead."
        )
    rng = np.random.default_rng(seed)
    shape = (n_pulse, n_range)

    # Analytic LFM replica from quadratic-ramp metadata (one row); the chirp is
    # the same across pulses, so the replica is order-1 over the range axis.
    chirp_params = lfm_chirp_ramp_coeffs(
        n_range, sample_rate_hz=sample_rate_hz, bandwidth_hz=bandwidth_hz
    )
    replica_step = ExecutionStep(
        "ramp", ("ones1d",), ("replica",), params=chirp_params
    )
    # Build the replica via the host executor (the v2 ramp kernel) so the MF
    # coefficients are *derived from* the metadata, not an independent dense array.
    replica_words = ExecutionPlan(steps=(replica_step,)).execute(
        {"ones1d": _unit_words((n_range,))}
    )["replica"]
    # Matched-filter spectrum = conj(FFT(replica)); conj is free (a sign flip).
    from spf.fft import spf32_fft

    conj_h = spf_conj_words(spf32_fft.fft(replica_words))

    # Synthetic cube: an LFM echo per pulse with a per-pulse Doppler phase.
    echo = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    x = encode_words(echo)

    # Per-pulse deramp is the SAME quadratic ramp (broadcast over the pulse axis):
    # extend the order-1 chirp params to an order-2 (pulse, range) ramp that is
    # zero over the pulse axis.
    deramp_R = [[0, 0], [0, int(chirp_params["ramp_R"][0][0])]]
    deramp_b = [0, int(chirp_params["ramp_b"][0])]
    deramp_quad = [0, int(chirp_params["quadratic_ramp"][0])]

    plan = ExecutionPlan(
        steps=(
            ExecutionStep(
                "ramp",
                ("x",),
                ("c",),
                params={
                    "ramp_R": deramp_R,
                    "ramp_b": deramp_b,
                    "quadratic_ramp": deramp_quad,
                    "nonlinear": True,
                },
            ),
            ExecutionStep("fft", ("c",), ("X",)),  # range FFT (last axis)
            ExecutionStep("diagonal_scale", ("X", "conj_h"), ("mf",)),
            ExecutionStep("ifft", ("mf",), ("rp",)),  # range profile per pulse
            ExecutionStep("permute", ("rp",), ("rpT",), params={"axes": (1, 0)}),
            ExecutionStep("fft", ("rpT",), ("rd_T",)),  # slow-time (Doppler) FFT
            ExecutionStep("permute", ("rd_T",), ("rd",), params={"axes": (1, 0)}),
        )
    )
    inputs = {"x": x, "conj_h": conj_h}

    # FP64 oracle: same ops in complex128, scored against the decoded contract.
    cz = decode_words(x) * np.exp(
        1j
        * _ramp_phase_rad(
            shape, deramp_R, deramp_b, quadratic_ramp=deramp_quad, nonlinear=True
        )
    )
    X = np.fft.fft(cz, axis=-1)
    mf = X * decode_words(conj_h)
    rp = np.fft.ifft(mf, axis=-1)
    rd = np.fft.fft(rp, axis=0)  # Doppler FFT over pulse axis
    return plan, inputs, rd, replica_words


def cfar_detections(
    rd_words_or_decoded: np.ndarray,
    *,
    guard: int = 1,
    train: int = 4,
    threshold_db: float = 12.0,
) -> np.ndarray:
    """Host-side CA-CFAR over a range-Doppler map (egress boundary, NOT hot path).

    ``abs2`` → cell-average noise estimate (training cells, excluding guard +
    cell-under-test) → threshold. Returns a boolean detection mask. This is the
    add-domain detection head; it runs once at egress, off the multiplier-free
    SPF datapath, and is deliberately a plain FP64 reduction.
    """
    rd = rd_words_or_decoded
    if rd.dtype == np.uint32:
        rd = decode_words(rd)
    power = np.abs(rd) ** 2  # abs2
    flat = power.ravel()
    n = flat.size
    thr_lin = 10.0 ** (threshold_db / 10.0)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - guard - train)
        hi = min(n, i + guard + train + 1)
        idx = [
            j for j in range(lo, hi) if abs(j - i) > guard
        ]
        if not idx:
            continue
        noise = float(np.mean(flat[idx]))
        mask[i] = noise > 0.0 and flat[i] > thr_lin * noise
    return mask.reshape(power.shape)


def _delay_replica_spectrum_block(n_range: int, delay: int):
    """The matched-filter spectrum of a pure-delay replica, as a v3 block.

    A unit-impulse replica at sample ``delay`` has matched-filter spectrum
    ``conj(fft(δ_delay))[m] = exp(+i·2π·m·delay/N)`` — a constant-modulus, single
    closed-form linear phase, i.e. an exact ``SEPARABLE_PHASE`` DFT row of index
    ``delay`` whenever ``N`` divides ``2^16`` (so the per-index increment lands on
    an integer row). Building the coefficient *from* this block keeps the
    declared structure and the words in lock-step, which is what lets
    :func:`~spf.tensor.rewrite.provenance_from_v3` certify the roll structurally.
    """
    from .v3 import BaseModelKind, BaseModelRef, SPFBlockV3

    c = (int(delay) % n_range) * (1 << 16) // n_range
    return SPFBlockV3(
        dims=(n_range,),
        base=BaseModelRef(
            int(BaseModelKind.SEPARABLE_PHASE),
            params={"phase_terms": [
                {"axis_i": 0, "axis_j": 0, "closed_form": True, "omega": [c] * n_range}
            ]},
        ),
    )


def radar_delay_compression_plan(
    n_range: int = 64, *, seed: int = 2, delay: int = 5, provenance: str = "fit",
    rounding_contract: str = "packed_steps",
) -> tuple[ExecutionPlan, dict[str, np.ndarray], np.ndarray, Any]:
    """Pulse compression against a **pure-delay** replica → a roll, with provenance.

    A unit-impulse replica at sample ``delay`` has a unit-modulus linear-phase
    matched-filter spectrum ``conj(fft(δ_delay))[m] = exp(+i·2π·m·delay/N)``, so
    the whole ``fft → diagonal_scale → ifft`` pulse-compression is a circular shift
    of the echo. Unlike :func:`radar_pulse_compression_plan` (a general spectrum),
    this builder's coefficient *is* a roll, built from the ``SEPARABLE_PHASE``
    replica-spectrum block, so :class:`RollThroughFourier` fires on builder output
    and collapses the triple to one ``gather``.

    ``provenance`` selects how that provenance is derived — both routes are honest
    and gated, neither hand-fed:

    * ``"fit"`` (default) — :func:`~spf.tensor.rewrite.fit_phase_ramp` measures the
      shift from the actual coefficient *words*, returning ``phase_ramp`` only when
      the modulus is unit and the linear-phase fit is within tolerance.
    * ``"v3"`` — :func:`~spf.tensor.rewrite.provenance_from_v3` certifies the shift
      *structurally* from the ``SEPARABLE_PHASE`` block (an exact ``dft_row``),
      which RollThroughFourier now consumes too (the modulation theorem makes a
      DFT-row diagonal a shift by ``-index``). This is the Speculation-#1 path:
      the rewrite fires off a v3-declared coefficient, not a fitted one.

    Returns ``(plan, inputs, fp64_range_profile, ctx)``.
    """
    from .rewrite import RewriteContext, fit_phase_ramp, provenance_from_v3
    from .v3 import block_to_complex

    rng = np.random.default_rng(seed)
    echo = rng.standard_normal(n_range) + 1j * rng.standard_normal(n_range)
    block = _delay_replica_spectrum_block(n_range, delay)
    x = encode_words(echo)
    # Coefficient built from the SEPARABLE_PHASE block (bit-identical to
    # conj(fft(δ_delay))) so the declared structure and the words match exactly.
    conj_h = encode_words(block_to_complex(block))
    plan = ExecutionPlan(
        steps=(
            ExecutionStep("fft", ("x",), ("X",)),
            ExecutionStep("diagonal_scale", ("X", "conj_h"), ("mf",)),
            ExecutionStep("ifft", ("mf",), ("y",)),
        )
    )
    inputs = {"x": x, "conj_h": conj_h}
    fp64 = np.fft.ifft(np.fft.fft(echo) * decode_words(conj_h))
    if provenance == "v3":
        pv: Any = provenance_from_v3(block)
        prov = {"conj_h": pv} if pv.exact else {}
    elif provenance == "fit":
        pv = fit_phase_ramp(conj_h)
        prov = {"conj_h": pv} if pv is not None else {}
    else:
        raise ValueError(f"provenance must be 'fit' or 'v3', got {provenance!r}")
    ctx = RewriteContext(provenance=prov, rounding_contract=rounding_contract)
    return plan, inputs, fp64, ctx


def dft_row_orthogonality_plan(
    f: int, g: int, *, n: int = 64, rounding_contract: str = "packed_steps"
) -> tuple[ExecutionPlan, dict[str, np.ndarray], np.ndarray, Any]:
    """Contract two exact ``SEPARABLE_PHASE`` DFT rows, with v3-derived provenance.

    Builds ``conj(row_f)`` and ``row_g`` as constant-modulus ``SEPARABLE_PHASE``
    blocks, contracts them with a ``decoded_contract_reference``, and returns a
    :class:`RewriteContext` whose provenance is derived *structurally* from each
    v3 block via :func:`~spf.tensor.rewrite.provenance_from_v3`. The inner product
    is the closed-form ``N·δ_{fg}``, so :class:`PhaseRampOrthogonality` fires on
    builder output and folds the whole contraction away. Returns
    ``(plan, inputs, fp64_inner_product, ctx)``.
    """
    from dataclasses import replace as _replace

    from .core import ReducerPolicy, ReducerSpec
    from .rewrite import RewriteContext, provenance_from_v3
    from .v3 import BaseModelKind, BaseModelRef, SPFBlockV3, block_to_complex

    def _row_block(idx: int) -> SPFBlockV3:
        c = int(idx) * (1 << 16) // n
        return SPFBlockV3(
            dims=(n,),
            base=BaseModelRef(
                int(BaseModelKind.SEPARABLE_PHASE),
                params={"phase_terms": [
                    {"axis_i": 0, "axis_j": 0, "closed_form": True, "omega": [c] * n}
                ]},
            ),
        )

    bf, bg = _row_block(f), _row_block(g)
    row_f, row_g = block_to_complex(bf), block_to_complex(bg)
    a = encode_words(np.conj(row_f))
    b = encode_words(row_g)
    spec = ReducerSpec(policy=ReducerPolicy.CARTESIAN_KAHAN)
    plan = ExecutionPlan(
        steps=(ExecutionStep("decoded_contract_reference", ("a", "b"), ("ip",), reducer=spec),)
    )
    inputs = {"a": a, "b": b}
    fp64 = np.tensordot(decode_words(a), decode_words(b), axes=1)
    # 'a' carries conj(row_f): mark conjugated; its row index stays f so the rule's
    # index compare (f == g) matches the true N·δ_{fg}.
    pa = _replace(provenance_from_v3(bf), conjugated=True)
    pb = provenance_from_v3(bg)
    ctx = RewriteContext(provenance={"a": pa, "b": pb}, rounding_contract=rounding_contract)
    return plan, inputs, fp64, ctx


def circulant_inverse_plan(
    n: int = 64, *, seed: int = 3
) -> tuple[ExecutionPlan, dict[str, np.ndarray], np.ndarray, Any]:
    """Apply a circulant **inverse** ``C(h)⁻¹ x`` — the spectral-map solve path.

    The spectral mapping theorem ``f(C(h)) = C(f(ĥ))`` with ``f(z)=1/z`` makes the
    inverse of a circulant another circulant whose kernel is the *pointwise
    reciprocal spectrum*. :func:`~spf.tensor.rewrite.spectral_map` materialises
    ``1/ĥ`` at plan time, so the inverse-apply is the same ``fft → diagonal_scale →
    ifft`` shape as a forward circulant — there is no dense solve. Because the SPF
    reciprocal is an exact word negation in log-polar, a forward circulant followed
    by this inverse collapses to the identity *bit-exactly* under
    :func:`~spf.tensor.rewrite.rewrite_plan` (cancel + diagonal-compose +
    no-op-prune); the returned :class:`RewriteContext` carries both kernels as
    constants so that chain rewrites. Returns ``(plan, inputs, fp64_inverse, ctx)``.
    """
    from .rewrite import RewriteContext, spectral_map

    rng = np.random.default_rng(seed)
    # Spectrum kept clear of zero so the reciprocal is well-conditioned.
    h = (1.0 + 0.3 * rng.standard_normal(n)) + 0.2j * rng.standard_normal(n)
    h_words = encode_words(h)
    h_inv_words = spectral_map(h_words, lambda z: 1.0 / z)
    x = encode_words(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    plan = ExecutionPlan(
        steps=(
            ExecutionStep("fft", ("x",), ("X",), params={"n": n}),
            ExecutionStep("diagonal_scale", ("X", "h_inv"), ("Y",)),
            ExecutionStep("ifft", ("Y",), ("y",), params={"n": n}),
        )
    )
    inputs = {"x": x, "h_inv": h_inv_words}
    # FP64 oracle: C(h)⁻¹ via the same reciprocal spectrum (decoded), not 1/h direct,
    # so both arms share the SPF-rounded kernel.
    fp64 = np.fft.ifft(np.fft.fft(decode_words(x)) * decode_words(h_inv_words))
    ctx = RewriteContext(constants={"h": h_words, "h_inv": h_inv_words})
    return plan, inputs, fp64, ctx


def _dense_toeplitz(col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Dense ``N×N`` Toeplitz from its first column / first row (``row[0]==col[0]``)."""
    n = col.size
    T = np.empty((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            T[i, j] = col[i - j] if i >= j else row[j - i]
    return T


def toeplitz_apply_plan(
    n: int = 64, *, seed: int = 5
) -> tuple[ExecutionPlan, dict[str, np.ndarray], np.ndarray, Any]:
    """Apply a **Toeplitz** operator ``T x`` as one ``2N`` circulant (relationship #10).

    A Toeplitz apply is linear (not circular) convolution, so it embeds in a single
    ``2N`` circulant: with first column ``c = [col, 0, row[1:][::-1]]`` the product
    ``T x`` is the first ``N`` samples of ``C(c)·[x; 0]`` (the zero-padded input). The
    ``2N`` spectrum ``ĉ`` is a plan-time constant, so the apply is the familiar
    ``fft → diagonal_scale(ĉ) → ifft`` over the padded input, then a ``gather`` of the
    first ``N`` rows — the off-periodic bridge that carries the circulant algebra
    (#1/#2) onto a general convolution / trained layer.

    This is a **builder, not a peephole rewrite**: the zero-pad of the *runtime*
    input to ``2N`` and the slice back to ``N`` are not ExecutionPlan ops, and
    ``gather`` (``np.take``) cannot introduce zeros that are not already in the
    source — the same structural gap that makes :func:`circulant_inverse_plan` and
    Cooley–Tukey lowering builders rather than rules. The *solve* does not collapse
    to a spectral reciprocal — truncating the ``2N`` circulant breaks its inverse —
    so a Toeplitz deconvolution is an iterative solve whose fast matvec is exactly
    this plan, optionally with the circulant spectral inverse as a preconditioner.

    Returns ``(plan, inputs, fp64_apply, ctx)``. ``fp64_apply`` is the dense ``T x``
    (an independent oracle); ``ctx`` carries the ``2N`` spectrum constant.
    """
    from .rewrite import RewriteContext

    rng = np.random.default_rng(seed)
    col = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    row = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    row[0] = col[0]  # the shared diagonal — col and row must agree at lag 0
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    T = _dense_toeplitz(col, row)

    m = 2 * n
    c = np.concatenate([col, [0.0], row[1:][::-1]])  # 2N circulant first column
    c_spec = encode_words(np.fft.fft(c))
    xpad = encode_words(np.concatenate([x, np.zeros(n, dtype=np.complex128)]))

    plan = ExecutionPlan(
        steps=(
            ExecutionStep("fft", ("xpad",), ("X",), params={"n": m}),
            ExecutionStep("diagonal_scale", ("X", "c_spec"), ("Y",)),
            ExecutionStep("ifft", ("Y",), ("yfull",), params={"n": m}),
            ExecutionStep(
                "gather", ("yfull",), ("y",),
                params={"indices": np.arange(n, dtype=np.int64), "axis": -1},
            ),
        )
    )
    inputs = {"xpad": xpad, "c_spec": c_spec}
    fp64 = T @ x
    ctx = RewriteContext(constants={"c_spec": c_spec})
    return plan, inputs, fp64, ctx


def evm(actual_words_or_decoded: np.ndarray, reference: np.ndarray) -> float:
    """RMS error-vector magnitude of an SPF result vs an FP64 reference."""
    got = actual_words_or_decoded
    if got.dtype == np.uint32:
        got = decode_words(got)
    denom = float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    if denom == 0.0:
        return float(np.sqrt(np.mean(np.abs(got - reference) ** 2)))
    return float(np.sqrt(np.mean(np.abs(got - reference) ** 2)) / denom)


def plan_inputs(builder_inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Defensive copy of a builder's inputs (lowered + reference share names)."""
    return {k: np.asarray(v, dtype=np.uint32).copy() for k, v in builder_inputs.items()}
