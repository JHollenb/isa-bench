"""Observe array boundaries without calling logical traffic physical DRAM traffic."""

from __future__ import annotations

import contextlib
import sys
import weakref
from collections import Counter
from collections.abc import Iterator
from types import FrameType
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy import fft  # type: ignore[import-untyped]

#: Function names that cross the SPF-word <-> Cartesian boundary. An observed
#: call to any of these is one fallback conversion (design v0.2 §4 invariant 8).
CONVERSION_FUNCTIONS = frozenset(
    {
        "encode_words",
        "decode_words",
        "encode_complex",
        "decode_complex",
        "materialize_words",
        "reconstruct",
    }
)


def conversions_from_calls(calls: Any) -> int:
    """Count observed word<->Cartesian conversions in an :class:`ExecutionTrace` call map.

    ``calls`` is ``ExecutionTrace.summary()["calls"]`` — qualified name to call
    count.  The result is a measured lower bound for that arm's
    ``fallback_conversions``: it sees Python-level dispatch only, never a
    conversion buried inside a native kernel.
    """
    total = 0
    for key, count in dict(calls).items():
        if str(key).rsplit(".", 1)[-1] in CONVERSION_FUNCTIONS:
            total += int(count)
    return total


class ExecutionTrace:
    """Actual Python call edges and structural consumers, outside timed runs.

    Captures calls in SPF and SciPy FFT on the calling thread. This proves
    dispatch, not native instruction counts, memory traffic, or performance.
    Expressions and addressed sample counts are observed at their consumers;
    constructing an SPF object alone is not evidence of structured execution.
    """

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.edges: Counter[tuple[str, str]] = Counter()
        self.consumers: list[dict[str, Any]] = []

    @staticmethod
    def _name(frame: FrameType) -> str:
        return f"{frame.f_globals.get('__name__', '')}.{frame.f_code.co_qualname}"

    def _profile(self, frame: FrameType, event: str, arg: Any) -> None:
        if event != "call":
            return
        module = str(frame.f_globals.get("__name__", ""))
        if not module.startswith(("spf.renderer.", "spf.tensor.", "spf.dsp.", "scipy.fft.")):
            return
        name = self._name(frame)
        self.calls[name] += 1
        if frame.f_back is not None:
            self.edges[self._name(frame.f_back), name] += 1
        receiver = frame.f_locals.get("self")
        method = frame.f_code.co_name
        if receiver is not None and method in {"sample_words", "materialize_words", "reconstruct", "execute"}:
            row: dict[str, Any] = {"function": name, "receiver": type(receiver).__name__}
            expr = getattr(receiver, "expr", None)
            if expr is not None:
                row["expression"] = expr.op
            if method == "sample_words":
                query = frame.f_locals.get("linear")
                row["requested_elements"] = None if query is None else int(np.size(query))
                row["phase_requested"] = frame.f_locals.get("phase", True)
            if type(receiver).__name__ == "ExecutionPlan":
                row["plan_ops"] = [step.op for step in receiver.steps]
                row["reference_only"] = receiver.reference_only
            self.consumers.append(row)

    @contextlib.contextmanager
    def installed(self) -> Iterator[ExecutionTrace]:
        previous = sys.getprofile()
        if previous is not None:
            raise RuntimeError("execution trace requires an unprofiled calling thread")
        sys.setprofile(self._profile)
        try:
            yield self
        finally:
            sys.setprofile(previous)

    def summary(self) -> dict[str, Any]:
        return {
            "instrument": "Python call events on the calling thread; excludes native internals; not timed",
            "calls": dict(sorted(self.calls.items())),
            "edges": [{"caller": a, "callee": b, "count": n} for (a, b), n in sorted(self.edges.items())],
            "consumers": self.consumers,
        }


class BoundaryTrace:
    """Lower bound over named FFT/codec/transfer/multiply boundaries.

    Does not see internal FFT stages, allocator traffic, or every NumPy ufunc
    temporary. Timing and RSS profiling run separately without these hooks.
    """

    def __init__(self, existing: tuple[np.ndarray, ...] = ()) -> None:
        self.known = {}
        for x in existing:
            root = self.storage_root(x)
            self.known[id(root)] = weakref.ref(root)
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def storage_root(a: np.ndarray) -> np.ndarray:
        while isinstance(a.base, np.ndarray):
            a = a.base
        return a

    def observe(self, a: Any, name: str) -> None:
        if not isinstance(a, np.ndarray) or a.size == 0:
            return
        root = self.storage_root(a)
        old = self.known.get(id(root))
        if old is not None and old() is root:
            return
        # Also handle independently wrapped views of the same underlying buffer.
        for ref in self.known.values():
            other = ref()
            if other is not None and np.shares_memory(a, other):
                return
        self.known[id(root)] = weakref.ref(root)
        self.events.append(
            {
                "event": "materialize",
                "boundary": name,
                "elements": int(a.size),
                "bytes": int(a.nbytes),
                "dtype": str(a.dtype),
                "shape": list(a.shape),
            }
        )

    def wrap(self, function: Any, name: str) -> Any:
        def measured(*args: Any, **kwargs: Any) -> Any:
            arrays = [a for a in args if isinstance(a, np.ndarray)]
            for a in arrays:
                self.observe(a, name + ":input")
            result = function(*args, **kwargs)
            value = result.transfer if hasattr(result, "transfer") else result
            self.observe(value, name + ":output")
            self.events.append(
                {
                    "event": "operation",
                    "boundary": name,
                    "logical_bytes_read": sum(a.nbytes for a in arrays),
                    "logical_bytes_written": value.nbytes if isinstance(value, np.ndarray) else 0,
                }
            )
            return result

        return measured

    @contextlib.contextmanager
    def installed(self) -> Iterator[BoundaryTrace]:
        from spf.renderer import filter_chain
        from spf.tensor import core

        from . import fields

        with contextlib.ExitStack() as stack:
            for module, names in [
                (fft, ("fftn", "ifftn", "irfftn")),
                (np.fft, ("fftn", "ifftn")),
                (filter_chain, ("encode_words", "decode_words")),
                (core, ("spf_mul_words",)),
                (filter_chain.FilterChain, ("_single_transfer", "fused_transfer")),
                (fields, ("transfer_fp32",)),
            ]:
                for name in names:
                    original = getattr(module, name)
                    stack.enter_context(patch.object(module, name, self.wrap(original, name)))
            yield self

    def summary(self) -> dict[str, Any]:
        operations = [x for x in self.events if x["event"] == "operation"]
        return {
            "materialized_elements": sum(x["elements"] for x in self.events if x["event"] == "materialize"),
            "logical_bytes_read": sum(x["logical_bytes_read"] for x in operations),
            "logical_bytes_written": sum(x["logical_bytes_written"] for x in operations),
            "fft_count": sum(x["boundary"] in {"fftn", "ifftn", "irfftn"} for x in operations),
            # Design v0.2 §4 invariant 8 / audit item E36: every arm reports its
            # Cartesian conversion count. ``encode_words`` / ``decode_words`` are
            # the only word<->Cartesian boundaries this trace patches, so their
            # observed call count *is* the conversion count for this arm; it
            # agrees with ``RenderReport.metrics.fallback_conversions`` (2 for
            # the DENSE_SPF variant: one encode, one decode).
            "fallback_conversions": sum(x["boundary"] in {"encode_words", "decode_words"} for x in operations),
            "fallback_conversion_instrument": (
                "observed encode_words/decode_words boundary crossings on the untimed instrumented replay; "
                "lower bound, excludes conversions inside native kernels"
            ),
            "materialization_instrument": (
                "observed named array boundaries; lower bound, excludes internal ufunc/FFT temporaries"
            ),
            "bytes_read": None,
            "bytes_written": None,
            "dram_instrument": "unavailable; logical byte counts are not hardware DRAM transfers",
            "events": self.events,
        }
