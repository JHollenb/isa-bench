"""Operator wrapper for SPF-native QAM16 log-projection demap."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from spf.native import spf32 as c_spf32
from spf.tensor import QAM16LogProjMetadata


@dataclass(frozen=True, slots=True)
class QAM16LogProjResult:
    """Hard decisions and side outputs from QAM16 log-projection demap."""

    bits: npt.NDArray[np.uint8]
    confidence: npt.NDArray[np.uint8]
    erasure: npt.NDArray[np.uint8] | None
    metadata: QAM16LogProjMetadata


@dataclass(frozen=True, slots=True)
class QAM16LogProjOperator:
    """OperatorBlock-compatible QAM16 log-projection demapper."""

    metadata: QAM16LogProjMetadata = field(
        default_factory=lambda: QAM16LogProjMetadata(qam_projection_threshold_L=256)
    )

    def projection_planes(
        self,
        y: npt.ArrayLike,
        h_inv: npt.ArrayLike,
    ) -> tuple[
        npt.NDArray[np.int16],
        npt.NDArray[np.uint16],
        npt.NDArray[np.int16],
        npt.NDArray[np.int16],
    ]:
        """Return ``L_eq, theta_eq, L_I, L_Q`` for diagnostics/tests."""
        return self.metadata.equalized_projection_planes(y, h_inv)

    def demap(
        self,
        y: npt.ArrayLike,
        h_inv: npt.ArrayLike,
        *,
        erasure_mask: npt.ArrayLike | None = None,
    ) -> QAM16LogProjResult:
        """Call the SPF-native log-projection hard demapper."""
        erasure = _erasure(erasure_mask, np.asarray(y).size)
        bits, confidence = c_spf32.qam16_logproj_eq_demod_fused(
            y,
            h_inv,
            agc_L_offset=self.metadata.agc_L_offset,
            cpe_theta=self.metadata.cpe_theta,
            threshold_L=self.metadata.qam_projection_threshold_L,
            phase_bits=self.metadata.qam_projection_phase_bits,
            erasure_mask=erasure,
        )
        return QAM16LogProjResult(
            bits=bits,
            confidence=confidence,
            erasure=erasure,
            metadata=self.metadata,
        )

    def to_metadata(self) -> dict[str, int | str | bool]:
        return self.metadata.to_metadata()

    @classmethod
    def from_metadata(cls, data: dict[str, object]) -> QAM16LogProjOperator:
        return cls(QAM16LogProjMetadata.from_metadata(data))


def _erasure(erasure_mask: npt.ArrayLike | None, n: int) -> npt.NDArray[np.uint8] | None:
    if erasure_mask is None:
        return None
    erasure = np.ascontiguousarray(np.asarray(erasure_mask), dtype=np.uint8).ravel()
    if erasure.size != n:
        raise ValueError(f"erasure_mask length {erasure.size} != symbol count {n}")
    return erasure


__all__ = [
    "QAM16LogProjOperator",
    "QAM16LogProjResult",
]
