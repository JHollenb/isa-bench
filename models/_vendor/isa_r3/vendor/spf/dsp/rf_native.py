"""RF-native polar source adapters for SPF runtime simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from spf.runtime import Domain, Layout, Provenance, SPFBuffer, SPFMeta


@dataclass(frozen=True, slots=True)
class PolarRFBuffer:
    """Simulation-side RF-native polar samples before SPF runtime ingestion."""

    l_code: npt.NDArray[np.int16]
    theta: npt.NDArray[np.uint16]
    sample_rate_hz: float | None = None
    source: str = "polar_rf_sim"

    @classmethod
    def from_l_theta(
        cls,
        l_code: npt.ArrayLike,
        theta: npt.ArrayLike,
        *,
        sample_rate_hz: float | None = None,
        source: str = "polar_rf_sim",
    ) -> PolarRFBuffer:
        l_arr, theta_arr = np.broadcast_arrays(
            np.asarray(l_code, dtype=np.int16),
            np.asarray(theta, dtype=np.uint16),
        )
        return cls(
            l_code=np.ascontiguousarray(l_arr, dtype=np.int16),
            theta=np.ascontiguousarray(theta_arr, dtype=np.uint16),
            sample_rate_hz=sample_rate_hz,
            source=source,
        )

    @property
    def words(self) -> npt.NDArray[np.uint32]:
        return np.ascontiguousarray(
            (self.l_code.astype(np.uint16).astype(np.uint32) << 16)
            | self.theta.astype(np.uint32),
            dtype=np.uint32,
        )

    def to_spf_buffer(self, *, backend: str = "native") -> SPFBuffer:
        return SPFBuffer(
            words=self.words,
            meta=SPFMeta(),
            layout=Layout.CONTIGUOUS,
            domain=Domain.TIME,
            provenance=Provenance(source=self.source, backend=backend, stages=("polar_ingress",)),
            backend=backend,
        )
