"""Input-conditioned diagonal-state sequence mixing on live SPF32 words.

The parameter generator is a sparse phase circuit. By default the state is
retained by the packed-record executor (phase clocks with winding, independent
rational attenuation, balanced packed reads). ``accumulator="packed"``
keeps the original contract: one SPF32 word per state, updated by the scalar
adder every token. A packed state stops absorbing a write ``v`` once
``|S|/|v| > 1/(2^(1/512) - 1) ~ 738`` (constant unit writes stall at 739.9 after
648 tokens), and a packed transition word cannot hold a decay smaller than one
log code per token (half-life 256), so slower decays round to none.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from spf.scalar import spf32

from .core import (
    SPF32_L_MIN_FINITE,
    SPF32_L_ZERO,
    SPF32_PHASE_MASK,
    SPF32_WORD_ZERO,
    SPFError,
    pack_words,
    spf_mul_words,
    unpack_L,
    unpack_theta,
)
from .phase_circuits import _integer, canonical_character, execute_phase_shear, phase_character_plan

if TYPE_CHECKING:
    from .clocked_state import ClockedState, ClockedStateParameters
    from .v2 import ExecutionPlan

Accumulator = Literal["clocked", "packed"]


@dataclass(frozen=True)
class PhaseShearSpec:
    """One nonlinear feature coupling, in phase ticks."""

    target: int
    control: int
    reference: int | None = None
    coefficients: tuple[int, ...] = (0, 0, 1)
    table: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", tuple(_integer(v) for v in self.coefficients))
        if self.table is not None:
            object.__setattr__(self, "table", tuple(_integer(v) for v in self.table))


@dataclass(frozen=True)
class PhaseFeatureNetwork:
    """Token-local shears followed by shared sparse characters.

    This declares a feature program; it does not train or select features.
    The token-local construction makes parameter generation streamable.
    """

    n_channels: int
    characters: Sequence[Sequence[Sequence[int]]]
    biases: Sequence[int] | None = None
    shears: tuple[PhaseShearSpec, ...] = ()
    _plan: ExecutionPlan = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        n_channels = _integer(self.n_channels)
        rows = tuple(canonical_character(row, n_channels) for row in self.characters)
        biases = None if self.biases is None else tuple(_integer(v) & SPF32_PHASE_MASK for v in self.biases)
        plan = phase_character_plan(rows, n_channels=n_channels, biases=biases)
        # Validate shear programs even for an empty sequence.
        unit = np.zeros((_integer(self.n_channels),), dtype=np.uint32)
        for shear in self.shears:
            unit = execute_phase_shear(unit, **vars(shear))
        object.__setattr__(self, "n_channels", n_channels)
        object.__setattr__(self, "characters", rows)
        object.__setattr__(self, "biases", biases)
        object.__setattr__(self, "shears", tuple(self.shears))
        object.__setattr__(self, "_plan", plan)

    def execute(self, addresses: npt.ArrayLike) -> np.ndarray:
        words = _words(addresses, "addresses", minimum_ndim=2)
        if words.shape[-1] != self.n_channels:
            raise SPFError("feature network address channel count differs from its program")
        for shear in self.shears:
            words = execute_phase_shear(words, **vars(shear))
        return self._plan.execute({"x": words})["out"]

    @property
    def feature_statistics(self) -> dict[str, int | bool]:
        """Counts for the shared character bank; shear/memory work is separate."""
        return dict(self._plan.factorization_metadata["phase_circuit"])


@dataclass(frozen=True)
class PhaseMixerDirection:
    """Feature selectors for each state's transition phase, write, and read."""

    transition_features: tuple[int, ...]
    write_features: tuple[int, ...]
    read_features: tuple[int, ...]
    write_log_codes: tuple[int, ...] | None = None
    read_log_codes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        for name, values in vars(self).items():
            if values is not None:
                object.__setattr__(self, name, tuple(_integer(v) for v in values))


@dataclass(frozen=True)
class DiagonalStateParameters:
    """Live packed diagonal transition and write/read planes, shape (..., T, N)."""

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray


@dataclass(frozen=True)
class SequenceMixerParameters:
    """Conditioned parameters; the scalar diagonal is applied exactly once."""

    forward: DiagonalStateParameters
    backward: DiagonalStateParameters | None = None
    diagonal: np.uint32 = SPF32_WORD_ZERO


@dataclass(frozen=True)
class PhaseSequenceMixer:
    """Native feature generator for a causal or bidirectional sequence mixer.

    A[t,n] has fixed shared log decay and phase omega[n]*feature[t,n] mod 65536.
    These phase controllers are not Hydra's positive real-valued dt. Forward
    and reverse scans share omega/decay but may select different features.

    ``decay_codes`` are SPF32 log codes (1/256 of a doubling per token), so the
    slowest nonzero decay they hold has a half-life of 256 tokens. ``decay_q16``
    gives the decay in 1/65536 of a doubling per token instead (half-life
    65536/|q| tokens); the clocked accumulator applies it exactly, and the packed
    transition plane rounds it to the nearest log code.
    """

    features: PhaseFeatureNetwork
    forward: PhaseMixerDirection
    transition_frequencies: tuple[int, ...]
    decay_codes: tuple[int, ...] | None = None
    backward: PhaseMixerDirection | None = None
    diagonal: int = int(SPF32_WORD_ZERO)
    decay_q16: tuple[int, ...] | None = None
    decay_half_lives: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        n = len(self.transition_frequencies)
        if n == 0:
            raise SPFError("a sequence mixer requires at least one state")
        omega = tuple(_integer(v) & SPF32_PHASE_MASK for v in self.transition_frequencies)
        decay = (0,) * n if self.decay_codes is None else tuple(_integer(v) for v in self.decay_codes)
        if self.decay_half_lives is not None:
            lives = tuple(_integer(v) for v in self.decay_half_lives)
            if (
                len(lives) != n
                or any(v <= 0 for v in lives)
                or self.decay_q16 is not None
                or self.decay_codes is not None
            ):
                raise SPFError("half lives require one positive integer per state and no quantized decay override")
            object.__setattr__(self, "decay_half_lives", lives)
            from .phase_clock import round_ratio

            decay = tuple(-int(v) for v in round_ratio(256, np.asarray(lives, np.int64)))
        if self.decay_q16 is not None:
            fine = tuple(_integer(v) for v in self.decay_q16)
            if len(fine) != n or any(not SPF32_L_MIN_FINITE * 256 <= v <= 0 for v in fine):
                raise SPFError("one finite nonpositive Q16 log decay is required per state")
            from .phase_clock import round_ratio

            rounded = tuple(int(v) for v in round_ratio(np.asarray(fine, np.int64), 256))
            if self.decay_codes is not None and decay != rounded:
                raise SPFError("decay_codes must be omitted or equal decay_q16 rounded to log codes")
            decay = rounded
            object.__setattr__(self, "decay_q16", fine)
        if len(decay) != n or any(not SPF32_L_MIN_FINITE <= v <= 0 for v in decay):
            raise SPFError("one finite nonpositive log decay code is required per state")
        m = len(self.features.characters)
        for direction in (self.forward, self.backward):
            if direction is None:
                continue
            for selectors in (direction.transition_features, direction.write_features, direction.read_features):
                if len(selectors) != n or any(not 0 <= _integer(v) < m for v in selectors):
                    raise SPFError("each direction needs one valid feature selector per state and field")
            for logs in (direction.write_log_codes, direction.read_log_codes):
                if logs is not None and (len(logs) != n or any(not -32768 <= v <= 32766 for v in logs)):
                    raise SPFError("write/read gains require one finite or zero log code per state")
        diagonal = _integer(self.diagonal)
        if not 0 <= diagonal <= 0xFFFFFFFF:
            raise SPFError("diagonal must be a packed SPF32 word")
        _words(np.asarray(diagonal, dtype=np.uint32), "diagonal")
        object.__setattr__(self, "transition_frequencies", omega)
        object.__setattr__(self, "decay_codes", decay)

    def _directions(self, addresses: npt.ArrayLike):
        """Per direction: transition phase plane (16-bit ticks), write and read gain planes."""
        features = self.features.execute(addresses)

        def gains(selectors, logs):
            selected = features[..., list(selectors)]
            if logs is None:
                return selected.copy()
            return pack_words(np.broadcast_to(np.asarray(logs, dtype=np.int16), selected.shape), unpack_theta(selected))

        def direction(spec: PhaseMixerDirection | None):
            if spec is None:
                return None
            control = unpack_theta(features[..., list(spec.transition_features)]).astype(np.int64)
            phase = (control * np.asarray(self.transition_frequencies, dtype=np.int64)) & SPF32_PHASE_MASK
            return (
                phase,
                gains(spec.write_features, spec.write_log_codes),
                gains(spec.read_features, spec.read_log_codes),
            )

        return direction(self.forward), direction(self.backward)

    def generate(self, addresses: npt.ArrayLike) -> SequenceMixerParameters:
        """Generate packed A/B/C without coherent additions or complex decode."""

        def direction(planes) -> DiagonalStateParameters | None:
            if planes is None:
                return None
            phase, b, c = planes
            a = pack_words(np.broadcast_to(np.asarray(self.decay_codes, dtype=np.int16), phase.shape), phase)
            return DiagonalStateParameters(a, b, c)

        forward, backward = self._directions(addresses)
        return SequenceMixerParameters(direction(forward), direction(backward), np.uint32(self.diagonal))

    def generate_clocked(self, addresses: npt.ArrayLike) -> ClockedStateParameters:
        """Generate clock planes: exact Q16 decay and 2^32-ring phase, no rounding."""
        from .clocked_state import ClockedDirectionParameters, ClockedStateParameters

        q16 = self.decay_q16 if self.decay_q16 is not None else tuple(256 * v for v in self.decay_codes)

        def direction(planes) -> ClockedDirectionParameters | None:
            if planes is None:
                return None
            phase, b, c = planes
            log = np.broadcast_to(np.asarray(q16, dtype=np.int64), phase.shape).copy()
            if self.decay_half_lives is not None:
                return ClockedDirectionParameters(
                    log,
                    phase.astype(np.uint32) << np.uint32(16),
                    b,
                    c,
                    np.ones(phase.shape, np.int64),
                    -1,
                    np.asarray(self.decay_half_lives, np.int64),
                )
            return ClockedDirectionParameters(log, phase.astype(np.uint32) << np.uint32(16), b, c)

        forward, backward = self._directions(addresses)
        return ClockedStateParameters(direction(forward), direction(backward), np.uint32(self.diagonal))

    def execute(
        self,
        addresses: npt.ArrayLike,
        values: npt.ArrayLike,
        *,
        accumulator: Accumulator = "clocked",
        **kwargs,
    ) -> SequenceMixerResult:
        if accumulator == "clocked":
            return _execute_clocked(values, self.generate_clocked(addresses), **kwargs)
        return execute_sequence_mixer(values, self.generate(addresses), accumulator=accumulator, **kwargs)


@dataclass
class SequenceMixerCounters:
    """Actual word-operation counts, not latency or peak-memory measurements."""

    word_multiplications: int = 0
    coherent_additions: int = 0
    fallback_conversions: int = 0
    state_bytes: int = 0


@dataclass(frozen=True)
class SequenceMixerResult:
    """Output words and streaming state: packed words, or a ``ClockedState``."""

    words: np.ndarray
    forward_state: np.ndarray | ClockedState
    backward_state: np.ndarray | ClockedState | None
    counters: SequenceMixerCounters


def _words(value: npt.ArrayLike, name: str, minimum_ndim: int = 0) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != np.uint32 or arr.ndim < minimum_ndim:
        raise SPFError(f"{name} must be packed SPF32 uint32 words with at least {minimum_ndim} axes")
    L = unpack_L(arr)
    if np.any(L == 32767) or np.any((L == SPF32_L_ZERO) & (arr != SPF32_WORD_ZERO)):
        raise SPFError(f"{name} must contain finite words or canonical zero")
    return arr


def _validate_parameters(parameters: SequenceMixerParameters) -> tuple[tuple[int, ...], int]:
    shape = _words(parameters.forward.a, "forward A", 2).shape
    if shape[-1] == 0:
        raise SPFError("a sequence mixer requires at least one state")
    for direction in (parameters.forward, parameters.backward):
        if direction is None:
            continue
        for name, plane in vars(direction).items():
            if _words(plane, name, 2).shape != shape:
                raise SPFError("all sequence parameter planes must share shape (..., T, N)")
        L = unpack_L(direction.a)
        if np.any(L > 0):
            raise SPFError("sequence transitions must not amplify magnitude")
    diagonal = _words(parameters.diagonal, "diagonal")
    if diagonal.ndim != 0:
        raise SPFError("sequence diagonal must be scalar")
    return shape[:-1], shape[-1]


def _resets(reset_before: npt.ArrayLike | None, shape: tuple[int, ...]) -> np.ndarray:
    if reset_before is None:
        return np.zeros(shape, dtype=np.bool_)
    reset = np.asarray(reset_before)
    if reset.dtype != np.bool_ or reset.shape != shape:
        raise SPFError("reset_before must be a boolean plane with the parameter batch/time shape")
    return reset


def clocked_sequence_parameters(parameters: SequenceMixerParameters) -> ClockedStateParameters:
    """Exact clock form of packed transitions: Q16 log = 256 * L, 2^32-ring phase = Theta << 16."""
    from .clocked_state import ClockedDirectionParameters, ClockedStateParameters

    _validate_parameters(parameters)

    def direction(d: DiagonalStateParameters | None) -> ClockedDirectionParameters | None:
        if d is None:
            return None
        if np.any(d.a == SPF32_WORD_ZERO):
            raise SPFError("clocked accumulation needs nonzero transitions; use reset_before or accumulator='packed'")
        log = unpack_L(d.a).astype(np.int64) * 256
        return ClockedDirectionParameters(log, unpack_theta(d.a).astype(np.uint32) << np.uint32(16), d.b, d.c)

    return ClockedStateParameters(
        direction(parameters.forward), direction(parameters.backward), np.uint32(parameters.diagonal)
    )


def _execute_clocked(
    values: npt.ArrayLike,
    parameters: ClockedStateParameters,
    *,
    initial_state: ClockedState | None = None,
    reset_before: npt.ArrayLike | None = None,
    chunk: int = 64,
) -> SequenceMixerResult:
    from .clocked_state import ClockedState, execute_clocked_state

    if initial_state is not None and not isinstance(initial_state, ClockedState):
        raise SPFError("clocked accumulation streams a ClockedState; packed word state needs accumulator='packed'")
    result = execute_clocked_state(
        values, parameters, chunk=chunk, initial_state=initial_state, reset_before=reset_before
    )
    states = [state for state in (result.forward_state, result.backward_state) if state is not None]
    counters = SequenceMixerCounters(
        word_multiplications=result.counters.word_multiplications,
        coherent_additions=result.counters.coherent_additions,
        state_bytes=sum(state.storage_bits() // 8 for state in states),
    )
    return SequenceMixerResult(result.words, result.forward_state, result.backward_state, counters)


def execute_sequence_mixer(
    values: npt.ArrayLike,
    parameters: SequenceMixerParameters,
    *,
    initial_state: npt.ArrayLike | ClockedState | None = None,
    reset_before: npt.ArrayLike | None = None,
    accumulator: Accumulator = "clocked",
    chunk: int = 64,
) -> SequenceMixerResult:
    """Execute strict off-diagonal scans plus D*v on live words.

    For each token: reset if requested, state=A*state, output=sum(C*state),
    state=state+B*v. Read before injection excludes self from each direction.
    The returned forward state is post-injection and supports causal chunks.
    Bidirectional calls require full context and reject initial_state.

    ``accumulator="clocked"`` (default) runs the same recurrence through
    :func:`spf.tensor.execute_clocked_state`: transitions become exact integer
    clocks, state lives in frame-relative fixed point, and each output is rounded
    to SPF32 once (``chunk`` sets the rebase interval). ``accumulator="packed"``
    keeps one SPF32 word per state and rounds every update; see the module note
    for where that state stops absorbing writes.
    """
    if accumulator == "clocked":
        return _execute_clocked(
            values,
            clocked_sequence_parameters(parameters),
            initial_state=initial_state,
            reset_before=reset_before,
            chunk=chunk,
        )
    if accumulator != "packed":
        raise SPFError("accumulator must be 'clocked' or 'packed'")
    shape, n = _validate_parameters(parameters)
    words = _words(values, "values", 2)
    if words.shape[:-1] != shape:
        raise SPFError("values must share parameter batch/time axes, with a payload channel axis")
    reset = _resets(reset_before, shape)
    state_shape = (*shape[:-1], n, words.shape[-1])
    if initial_state is not None and parameters.backward is not None:
        raise SPFError("streaming initial_state is supported only for a causal mixer")
    initial = np.full(state_shape, SPF32_WORD_ZERO, dtype=np.uint32)
    if initial_state is not None:
        initial = _words(initial_state, "initial_state").copy()
        if initial.shape != state_shape:
            raise SPFError("initial_state must have shape (..., N, payload_channels)")
    counters = SequenceMixerCounters(state_bytes=initial.nbytes * (1 + (parameters.backward is not None)))

    def multiply(a, b):
        result = spf_mul_words(a, b)
        counters.word_multiplications += result.size
        return result

    def add(a, b):
        # This is dispatch/zero bypass, not a second arithmetic implementation.
        a, b = np.broadcast_arrays(a, b)
        both = (a != SPF32_WORD_ZERO) & (b != SPF32_WORD_ZERO)
        result = np.where(a == SPF32_WORD_ZERO, b, a).copy()
        if np.any(both):
            result[both] = spf32.add(a[both], b[both])
            counters.coherent_additions += int(np.count_nonzero(both))
        return result

    def scan(direction: DiagonalStateParameters, reverse: bool, state: np.ndarray):
        output = np.empty(words.shape, dtype=np.uint32)
        length = shape[-1]
        indices = range(length - 1, -1, -1) if reverse else range(length)
        for t in indices:
            if not reverse:
                boundary = reset[..., t]
            else:
                boundary = reset[..., t + 1] if t + 1 < length else np.zeros(shape[:-1], dtype=np.bool_)
            state = np.where(boundary[..., None, None], SPF32_WORD_ZERO, state)
            state = multiply(direction.a[..., t, :, None], state)
            terms = multiply(direction.c[..., t, :, None], state)
            # Fixed left fold is the packed rounding contract.
            readout = terms[..., 0, :].copy()
            for k in range(1, n):
                readout = add(readout, terms[..., k, :])
            output[..., t, :] = readout
            state = add(state, multiply(direction.b[..., t, :, None], words[..., t, None, :]))
        return output, state

    output, forward_state = scan(parameters.forward, False, initial)
    backward_state = None
    if parameters.backward is not None:
        backward, backward_state = scan(
            parameters.backward, True, np.full(state_shape, SPF32_WORD_ZERO, dtype=np.uint32)
        )
        output = add(output, backward)
    if parameters.diagonal != SPF32_WORD_ZERO:
        output = add(output, multiply(parameters.diagonal, words))
    return SequenceMixerResult(output, forward_state, backward_state, counters)


__all__ = [
    "DiagonalStateParameters",
    "PhaseFeatureNetwork",
    "PhaseMixerDirection",
    "PhaseSequenceMixer",
    "PhaseShearSpec",
    "SequenceMixerCounters",
    "SequenceMixerParameters",
    "SequenceMixerResult",
    "clocked_sequence_parameters",
    "execute_sequence_mixer",
]
