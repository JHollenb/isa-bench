"""Strict/reference execution policy for SPF kernel programs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .core import (
    AccumulationDomain,
    PrecisionMode,
    ReducerPolicy,
    ReducerSpec,
    RoutingPolicy,
    SPFError,
)

POLICY_SCHEMA_VERSION = "spf-execution-policy-v1"


@dataclass(frozen=True)
class SPFExecutionPolicy:
    """Shared policy for reference, native, and strict SPF graph lowering.

    The policy is intentionally independent of PAN's ``ExecutionContext`` so SPF
    kernels can enforce the same contract without importing PAN.
    """

    name: str
    forbid_decode: bool
    forbid_cartesian_reducer: bool
    forbid_scores_f32: bool
    forbid_dense_fallback: bool
    require_words_io: bool
    require_named_reducers: bool
    require_proof: bool
    allow_materialized_reference: bool

    def __post_init__(self) -> None:
        if self.name not in {"reference", "native", "strict"}:
            raise SPFError("SPFExecutionPolicy name must be reference, native, or strict")

    @classmethod
    def reference(cls) -> SPFExecutionPolicy:
        return cls(
            name="reference",
            forbid_decode=False,
            forbid_cartesian_reducer=False,
            forbid_scores_f32=False,
            forbid_dense_fallback=False,
            require_words_io=False,
            require_named_reducers=False,
            require_proof=False,
            allow_materialized_reference=True,
        )

    @classmethod
    def native(cls) -> SPFExecutionPolicy:
        return cls(
            name="native",
            forbid_decode=True,
            forbid_cartesian_reducer=False,
            forbid_scores_f32=True,
            forbid_dense_fallback=False,
            require_words_io=True,
            require_named_reducers=True,
            require_proof=True,
            allow_materialized_reference=False,
        )

    @classmethod
    def strict(cls) -> SPFExecutionPolicy:
        return cls(
            name="strict",
            forbid_decode=True,
            forbid_cartesian_reducer=True,
            forbid_scores_f32=True,
            forbid_dense_fallback=True,
            require_words_io=True,
            require_named_reducers=True,
            require_proof=True,
            allow_materialized_reference=False,
        )

    @classmethod
    def for_name(cls, name: str) -> SPFExecutionPolicy:
        if name == "reference":
            return cls.reference()
        if name == "native":
            return cls.native()
        if name == "strict":
            return cls.strict()
        raise SPFError("SPFExecutionPolicy name must be reference, native, or strict")

    @classmethod
    def from_context(cls, context: Any) -> SPFExecutionPolicy:
        """Build from PAN-style context objects without a PAN dependency."""

        name = getattr(context, "name", context)
        return cls.for_name(str(name))

    def default_reducer(self) -> ReducerSpec:
        if self.name == "reference":
            return ReducerSpec(
                policy=ReducerPolicy.AUTO,
                routing=RoutingPolicy.PINNED,
                domain=AccumulationDomain.CARTESIAN,
                precision_mode=PrecisionMode.ACCURATE,
            )
        return ReducerSpec(
            policy=ReducerPolicy.AUTO,
            routing=RoutingPolicy.PINNED,
            domain=AccumulationDomain.LOGPOLAR,
            precision_mode=PrecisionMode.ACCURATE if self.name == "strict" else PrecisionMode.BALANCED,
        )

    @property
    def require_proof_metadata(self) -> bool:
        return self.require_proof

    def policy_payload(self) -> dict[str, Any]:
        return {
            "schema": POLICY_SCHEMA_VERSION,
            "name": self.name,
            "forbid_decode": self.forbid_decode,
            "forbid_cartesian_reducer": self.forbid_cartesian_reducer,
            "forbid_scores_f32": self.forbid_scores_f32,
            "forbid_dense_fallback": self.forbid_dense_fallback,
            "require_words_io": self.require_words_io,
            "require_named_reducers": self.require_named_reducers,
            "require_proof": self.require_proof,
            "allow_materialized_reference": self.allow_materialized_reference,
        }

    @property
    def policy_fingerprint(self) -> str:
        payload = json.dumps(self.policy_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return f"{POLICY_SCHEMA_VERSION}:{self.name}:{digest}"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema": POLICY_SCHEMA_VERSION,
            "policy_fingerprint": self.policy_fingerprint,
            "forbid_decode": self.forbid_decode,
            "forbid_cartesian_reducer": self.forbid_cartesian_reducer,
            "forbid_scores_f32": self.forbid_scores_f32,
            "forbid_dense_fallback": self.forbid_dense_fallback,
            "require_words_io": self.require_words_io,
            "require_named_reducers": self.require_named_reducers,
            "require_proof_metadata": self.require_proof,
            "require_proof": self.require_proof,
            "allow_materialized_reference": self.allow_materialized_reference,
        }

    def validate_reducer(self, reducer: ReducerSpec | None, *, purpose: str = "reducer") -> None:
        if reducer is None:
            return
        if self.forbid_cartesian_reducer and reducer.domain == AccumulationDomain.CARTESIAN:
            raise SPFError(f"{self.name} policy forbids CARTESIAN {purpose}")

    def validate_kernel(self, kernel: Any, *, purpose: str = "kernel") -> None:
        cost = kernel.cost() if hasattr(kernel, "cost") else {}
        proof = getattr(getattr(kernel, "expr", None), "proof", None)
        if self.require_proof and proof is None:
            raise SPFError(f"{self.name} policy requires proof metadata for {purpose}")
        if self.forbid_cartesian_reducer and int(cost.get("cartesian_fallback_count", 0)) > 0:
            raise SPFError(f"{self.name} policy forbids Cartesian fallback in {purpose}")
        reducer_domain = cost.get("reducer_domain")
        if self.forbid_cartesian_reducer and str(reducer_domain).upper() == "CARTESIAN":
            raise SPFError(f"{self.name} policy forbids Cartesian reducer in {purpose}")
        reference_ops = tuple(str(op) for op in cost.get("lowered_reference_ops", ()) or ())
        bad_reference_ops = tuple(
            op
            for op in reference_ops
            if op != "decoded_materialized_reference"
            and ("decoded" in op or "cartesian" in op.lower())
        )
        if self.forbid_decode and bad_reference_ops:
            raise SPFError(f"{self.name} policy forbids decoded reference ops in {purpose}: {bad_reference_ops}")


__all__ = ["SPFExecutionPolicy"]
