"""Runtime containers for SPF-resident product pipelines.

The runtime keeps packed SPF words as the canonical data plane and carries
confidence, erasure, and provenance as sidecar metadata. These types are
intentionally lightweight: they do not schedule kernels or own memory beyond
the arrays passed into them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt


class Layout(StrEnum):
    """Physical layout of an SPF word buffer."""

    CONTIGUOUS = "contiguous"
    BATCHED = "batched"
    ANTENNA_MAJOR = "antenna_major"
    SUBCARRIER_MAJOR = "subcarrier_major"


class Domain(StrEnum):
    """Signal-processing domain represented by the SPF words."""

    TIME = "time"
    FREQUENCY = "frequency"
    ANTENNA = "antenna"
    SLOW_TIME = "slow_time"
    SUBCARRIER = "subcarrier"
    DECISION = "decision"


@dataclass(frozen=True, slots=True)
class Rate:
    """Sampling or symbol-rate metadata for a runtime buffer."""

    sample_rate_hz: float | None = None
    symbol_rate_hz: float | None = None
    subcarrier_spacing_hz: float | None = None


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a runtime buffer came from and which backend produced it."""

    source: str = "unknown"
    backend: str = "native"
    stages: tuple[str, ...] = ()

    def with_stage(self, stage: str, *, backend: str | None = None) -> Provenance:
        return Provenance(
            source=self.source,
            backend=self.backend if backend is None else backend,
            stages=(*self.stages, stage),
        )


@dataclass(slots=True)
class SPFMeta:
    """Sidecar metadata for packed SPF words."""

    confidence: npt.NDArray[np.integer[Any]] | None = None
    erasure: npt.NDArray[np.uint8] | None = None
    noise_l: npt.NDArray[np.integer[Any]] | None = None
    timing_phase: int | None = None
    cfo_theta_per_sample: int | None = None
    cpe_theta: int | None = None


@dataclass(slots=True)
class SPFBuffer:
    """Packed SPF word buffer plus runtime metadata."""

    words: npt.NDArray[np.uint32]
    meta: SPFMeta = field(default_factory=SPFMeta)
    layout: Layout = Layout.CONTIGUOUS
    domain: Domain = Domain.TIME
    rate: Rate | None = None
    provenance: Provenance = field(default_factory=Provenance)
    backend: str = "native"

    def __post_init__(self) -> None:
        self.words = np.ascontiguousarray(np.asarray(self.words), dtype=np.uint32)
        if self.meta.erasure is not None:
            self.meta.erasure = np.ascontiguousarray(
                np.asarray(self.meta.erasure), dtype=np.uint8
            )
            if self.meta.erasure.size != self.words.size:
                raise ValueError(
                    f"erasure length {self.meta.erasure.size} != words length {self.words.size}"
                )
        if self.meta.confidence is not None and self.meta.confidence.size != self.words.size:
            raise ValueError(
                f"confidence length {self.meta.confidence.size} != words length {self.words.size}"
            )


@dataclass(frozen=True, slots=True)
class RuntimeTelemetry:
    """Execution metadata reported by product runtime calls."""

    backend: str
    stages: tuple[str, ...]
    materialized_equalized: bool = False
    diagnostics: bool = False
    details: dict[str, Any] = field(default_factory=dict)
