"""Sb Test 4: lazy multi-phasor tail accumulator — Python reference.

Mirror of ``native/src/spf32/core/spf32_accum.c`` used for C ↔ Python parity tests and as a
readable reference for the pairwise tree-reduction collapse heuristic.
See ``docs/research/sb-test-4-lazy-multiphasor.md`` for the design rationale
(and the FTO analysis of Lemurian Labs WO2022094710A1).

This module does not replace ``spf.scalar._ops.add``. It provides a distinct
accumulator API that callers opt into for long summations (inner products,
FIR taps, Softmax denominators) where the amortization win is structural.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from spf.scalar._constants import SPF32, Format
from spf.scalar._ops import add as spf_add
from spf.scalar._ops import mul as spf_mul

__all__ = [
    "ACCUM_K_MAX_DEFAULT",
    "ACCUM_K_MAX_HARD",
    "ACCUM_TROPICAL_DROP_LOG2",
    "ACCUM_TROPICAL_DROP_Q78",
    "LazyAccum",
    "accum_tropical_drop_code",
    "inner_product",
    "sum_array",
]

# Compile-time knobs — match the C header (native/include/spf32.h and
# native/src/spf32/core/spf32_accum.h). The `_Q78` name is legacy (SPF-32 is Q7.8); the
# physical threshold is a constant 32 log₂-magnitudes below max_L — the
# format's far-tail cutoff above which a phasor contributes less than one
# Q-fraction ULP to the sum. For SPF-32 that lands at 32 × 256 = 8192
# L-code units. SPF-16 (Q3.4) maps to 32 × 16 = 512; SPF-64 (Q15.16) to
# 32 × 65536 = 2_097_152. The ``_Q78`` constant below is pinned to SPF-32
# for backwards compatibility; ``LazyAccum`` computes the right per-format
# value via ``accum_tropical_drop_code(fmt)`` internally so SPF-16 /
# SPF-32 / SPF-64 accumulators all scale correctly.
ACCUM_K_MAX_DEFAULT: int = 8
ACCUM_K_MAX_HARD: int = 16
# Number of log₂-magnitudes below max_L at which a phasor's contribution
# falls below the format's far-tail cutoff. Same across SPF-16/32/64 —
# the threshold is a physical log-magnitude distance, not a Q-unit count;
# only the conversion to L-code units is format-specific.
ACCUM_TROPICAL_DROP_LOG2: int = 32
ACCUM_TROPICAL_DROP_Q78: int = ACCUM_TROPICAL_DROP_LOG2 * SPF32.l_scale


def accum_tropical_drop_code(fmt: Format = SPF32) -> int:
    """Per-format tropical-drop threshold in L-code (integer) units.

    ``LazyAccum`` compares ``max_L − threshold`` against each queued word's
    L code; the code-space scale is ``fmt.l_scale`` (Q-format divisor).
    Evaluated per push at call time, not import time, so mixed-format
    callers (SPF-16 / SPF-32 / SPF-64) all scale correctly against the
    same physical ``ACCUM_TROPICAL_DROP_LOG2`` cutoff.
    """
    return ACCUM_TROPICAL_DROP_LOG2 * fmt.l_scale


def _l_code(word: int, fmt: Format = SPF32) -> int:
    """Extract the signed L code from a packed SPF word.

    Mirrors the C ``accum_l_code`` helper; pulled out as a module-level
    function so the tests can call it directly on raw words without
    plumbing the struct.
    """
    raw = int(word) >> fmt.theta_bits
    mask = (1 << fmt.l_bits) - 1
    val = raw & mask
    # Sign-extend from l_bits.
    if val & (1 << (fmt.l_bits - 1)):
        val -= 1 << fmt.l_bits
    return int(val)


def _is_zero(word: int, fmt: Format = SPF32) -> bool:
    return int(word) == int(fmt.word_zero)


def _is_nan(word: int, fmt: Format = SPF32) -> bool:
    return int(word) == int(fmt.word_nan)


def _is_inf(word: int, fmt: Format = SPF32) -> bool:
    """Inf sentinel = L_CODE_INF with Θ payload ≠ NaN's all-ones."""
    mask = (1 << fmt.l_bits) - 1
    raw_l = (int(word) >> fmt.theta_bits) & mask
    raw_t = int(word) & ((1 << fmt.theta_bits) - 1)
    # Sign-extended compare.
    l_signed = raw_l - (1 << fmt.l_bits) if raw_l & (1 << (fmt.l_bits - 1)) else raw_l
    return l_signed == fmt.l_code_inf and raw_t != fmt.theta_nan_payload


@dataclass
class LazyAccum:
    """Deferred-collapse SPF accumulator keyed on a fixed-capacity queue.

    Parameters
    ----------
    k_cap : int
        Queue capacity ``K_MAX``. Paper-recommended tradeoff points:
        ``{4, 8, 16}``. Must be in ``[1, ACCUM_K_MAX_HARD]``.
    fmt : Format
        SPF format; defaults to ``SPF32`` since the C port only covers
        SPF-32. SPF-16/64 flow through the same algorithm by passing a
        different ``fmt``.

    Notes
    -----
    The public behaviour matches the C struct bit-for-bit when the same
    input sequence is pushed and the same tropical-drop threshold applies.
    See ``tests/sb/test_sb_test_4_lazy_multiphasor.py::test_c_python_parity``
    for the invariant.
    """

    k_cap: int = ACCUM_K_MAX_DEFAULT
    fmt: Format = SPF32
    queue: list[int] = field(default_factory=list)
    n_sb_calls: int = 0
    n_drops: int = 0

    def __post_init__(self) -> None:
        if self.k_cap <= 0:
            self.k_cap = ACCUM_K_MAX_DEFAULT
        if self.k_cap > ACCUM_K_MAX_HARD:
            self.k_cap = ACCUM_K_MAX_HARD

    @property
    def k_len(self) -> int:
        return len(self.queue)

    def reset(self) -> None:
        """Discard all queued phasors and statistics counters."""
        self.queue.clear()
        self.n_sb_calls = 0
        self.n_drops = 0

    # ── Core primitives ────────────────────────────────────────────────

    def _tropical_drop(self) -> None:
        """Discard queue entries with L < (max_L − drop_threshold)."""
        if len(self.queue) < 2:
            return
        finite_ls = [
            _l_code(w, self.fmt)
            for w in self.queue
            if not _is_zero(w, self.fmt)
            and not _is_nan(w, self.fmt)
            and not _is_inf(w, self.fmt)
        ]
        if not finite_ls:
            return
        max_l = max(finite_ls)
        # Per-format drop threshold — SPF-16/32/64 each scale their L-code
        # units differently (Q3.4 / Q7.8 / Q15.16), but the physical
        # log₂-magnitude cutoff is the same 32 far-tail decades across
        # all three. Computing from self.fmt here (not the import-time
        # SPF-32 constant) keeps SPF-16 accumulators from over-retaining
        # and SPF-64 accumulators from over-dropping.
        threshold = max_l - accum_tropical_drop_code(self.fmt)
        kept: list[int] = []
        for w in self.queue:
            if _is_zero(w, self.fmt):
                self.n_drops += 1
                continue
            if _is_nan(w, self.fmt) or _is_inf(w, self.fmt):
                kept.append(w)
                continue
            if _l_code(w, self.fmt) < threshold:
                self.n_drops += 1
                continue
            kept.append(w)
        self.queue = kept

    def _sb_add_pair(self, a: int, b: int) -> int:
        """Single Sb-based addition. Bumps the Sb-call counter."""
        if self.fmt is SPF32:
            from spf.scalar import spf32

            out = spf32.add(
                np.asarray([a], dtype=np.uint32),
                np.asarray([b], dtype=np.uint32),
            )
        else:
            out = spf_add(
                np.asarray([a], dtype=self.fmt.word_dtype),
                np.asarray([b], dtype=self.fmt.word_dtype),
                self.fmt,
            )
        self.n_sb_calls += 1
        return int(out[0])

    def _collapse(self) -> int:
        """Pairwise tree reduction on the queue. Returns the single fused word.

        Layer 0 merges (0,1), (2,3), (4,5), ...; layer 1 merges the layer-0
        outputs; etc. Odd tails carry forward. Each pair merge fires one Sb
        call. Mirrors ``accum_collapse`` in the C port.
        """
        if not self.queue:
            return int(self.fmt.word_zero)
        if len(self.queue) == 1:
            w = self.queue.pop(0)
            return int(w)
        scratch = list(self.queue)
        while len(scratch) > 1:
            half = len(scratch) // 2
            odd = len(scratch) & 1
            merged: list[int] = []
            for i in range(half):
                merged.append(self._sb_add_pair(scratch[2 * i], scratch[2 * i + 1]))
            if odd:
                merged.append(scratch[-1])
            scratch = merged
        self.queue.clear()
        return int(scratch[0])

    # ── Public API ────────────────────────────────────────────────────

    def push(self, word: int) -> None:
        """Enqueue a phasor; collapse on memory pressure."""
        w_int = int(word)

        # Drop zero sentinels — contribute nothing to the sum.
        if _is_zero(w_int, self.fmt):
            return

        # Sentinel poison: collapse and combine now.
        if _is_nan(w_int, self.fmt) or _is_inf(w_int, self.fmt):
            head = self._collapse()
            head = self._sb_add_pair(head, w_int)
            self.queue = [head]
            return

        if len(self.queue) < self.k_cap:
            self.queue.append(w_int)
            return

        # Queue full — try tropical drop first.
        self._tropical_drop()
        if len(self.queue) < self.k_cap:
            self.queue.append(w_int)
            return

        # Still full — must collapse.
        head = self._collapse()
        if self.k_cap > 1:
            self.queue = [head, w_int]
        else:
            # k_cap = 1 → every push collapses: equivalent to sequential Sb.
            self.queue = [self._sb_add_pair(head, w_int)]

    def finalize(self) -> int:
        """Collapse the queue and return the packed SPF word."""
        if not self.queue:
            return int(self.fmt.word_zero)
        return self._collapse()


# ── Convenience wrappers ────────────────────────────────────────────────


def sum_array(
    words: npt.ArrayLike,
    k_cap: int = ACCUM_K_MAX_DEFAULT,
    fmt: Format = SPF32,
) -> int:
    """Sum an array of SPF words through a lazy accumulator.

    Returns a single packed SPF word (Python int). The C counterpart
    ``spf32_accum_sum`` returns ``uint32_t``; both round-trip through the
    same bit pattern.
    """
    arr = np.ascontiguousarray(np.asarray(words, dtype=fmt.word_dtype))
    acc = LazyAccum(k_cap=k_cap, fmt=fmt)
    for w in arr.tolist():
        acc.push(int(w))
    return acc.finalize()


def inner_product(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    k_cap: int = ACCUM_K_MAX_DEFAULT,
    fmt: Format = SPF32,
) -> int:
    """SPF inner product ``Σ a[i] · b[i]`` through a lazy accumulator.

    Multiplication is free (integer add on L, Θ); the win comes from
    amortized Sb collapses on the running sum.

    Shapes must match — no NumPy broadcasting. An explicit shape check
    catches caller bugs that would otherwise silently broadcast a scalar
    or length-1 array into the full inner product, returning a plausible
    but wrong value. Mirrors ``spf.scalar._ops.dot``.
    """
    a_arr = np.ascontiguousarray(np.asarray(a, dtype=fmt.word_dtype))
    b_arr = np.ascontiguousarray(np.asarray(b, dtype=fmt.word_dtype))
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"inner_product: shape mismatch, got {a_arr.shape} and {b_arr.shape}"
        )
    products = spf_mul(a_arr, b_arr, fmt)
    return sum_array(products, k_cap=k_cap, fmt=fmt)
