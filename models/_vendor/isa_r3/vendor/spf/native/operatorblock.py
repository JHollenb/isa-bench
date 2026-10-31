"""Compatibility import for OperatorBlock ctypes helpers."""

from __future__ import annotations

from spf.tensor.native import (
    LIBSPF_AVAILABLE,
    LIBSPF_PATH,
    CBandRange,
    COperatorHeader,
    CProfileTermV2,
    CSynthPlan,
    CSynthStats,
    CTensorBlockHeaderV2,
    NativeOperatorBlock,
    NativeSynthResult,
    from_c_tensorblock_header_v2,
    to_c_bands,
    to_c_header,
    to_c_profile_term_v2,
    to_c_synth_plan,
    to_c_tensorblock_header_v2,
)

__all__ = [
    "LIBSPF_AVAILABLE",
    "LIBSPF_PATH",
    "CBandRange",
    "COperatorHeader",
    "CProfileTermV2",
    "CSynthPlan",
    "CSynthStats",
    "CTensorBlockHeaderV2",
    "NativeOperatorBlock",
    "NativeSynthResult",
    "from_c_tensorblock_header_v2",
    "to_c_bands",
    "to_c_header",
    "to_c_profile_term_v2",
    "to_c_synth_plan",
    "to_c_tensorblock_header_v2",
]
