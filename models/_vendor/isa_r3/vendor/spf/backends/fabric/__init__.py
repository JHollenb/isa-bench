"""SPU (design v0.2 §38 / Appendix C) to PAN-fabric (``pan-fabricd``) lowering."""

from .panfg1 import DAEMON_REGISTERED_OPS, FabricGraph, FabricOp, encode_request
from .spu_to_fabric import (
    PL_DISPATCH_JOULES,
    PL_DISPATCH_SECONDS,
    LoweringResult,
    SPUToFabricLowering,
    Unsupported,
    pack_word,
    simulate_graph,
)

__all__ = [
    "DAEMON_REGISTERED_OPS",
    "PL_DISPATCH_JOULES",
    "PL_DISPATCH_SECONDS",
    "FabricGraph",
    "FabricOp",
    "LoweringResult",
    "SPUToFabricLowering",
    "Unsupported",
    "encode_request",
    "pack_word",
    "simulate_graph",
]
