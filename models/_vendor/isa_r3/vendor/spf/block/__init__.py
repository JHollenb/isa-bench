"""Block formats: spectral block float for activation tensors (SBF-*)."""

from spf.block.sbf128 import BLOCK_LENGTH, BLOCK_SIZE, decode, encode, get_spectrum

__all__ = [
    "BLOCK_LENGTH",
    "BLOCK_SIZE",
    "decode",
    "encode",
    "get_spectrum",
]
