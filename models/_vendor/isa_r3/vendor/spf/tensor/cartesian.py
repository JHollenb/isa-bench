"""Cartesian sibling blocks for the Level-4 structured tensor surface.

SPFTensor exposes structure in log-magnitude/phase coordinates.  This module
adds the complementary Cartesian chart: patterns that are most visible in
real/imaginary coordinates, such as sparsity, low rank, conjugate symmetry, and
destructive-cancellation regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt

from .core import SPFError

CARTESIAN_TENSOR_MAJOR = 1
CARTESIAN_TENSOR_MINOR = 0
CARTESIAN_TENSOR_DTYPE = "CART_TB_v1"
CARTESIAN_HEADER_BYTES = 64
# Byte accounting reflects the precision the block actually stores and
# reconstructs (complex128 values / float64 components / int64 coordinates).
# Counting half-precision here while storing/reconstructing at double precision
# inflated the compression ratios and made them physically inconsistent with the
# (near-exact) rel_recon_err the same block reports.
COMPLEX128_BYTES = 16
FLOAT64_BYTES = 8
INT64_BYTES = 8


class CartesianProfile(StrEnum):
    """Structure profile selected in the Cartesian coordinate chart."""

    DENSE = "dense"
    REAL_ONLY = "real_only"
    IMAG_ONLY = "imag_only"
    SPARSE_COO = "sparse_coo"
    LOW_RANK = "low_rank"
    CONJUGATE_SYMMETRIC = "conjugate_symmetric"


@dataclass(frozen=True, slots=True)
class CartesianBlockHeader:
    """Metadata for one CartesianTensor block."""

    dims: tuple[int, ...]
    profile: CartesianProfile | str = CartesianProfile.DENSE
    version: int = CARTESIAN_TENSOR_MAJOR
    rank: int = 0
    zero_tol: float = 1e-12
    rel_tol: float = 1e-9
    metadata: dict[str, int | float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dims = tuple(int(d) for d in self.dims)
        if not dims:
            raise SPFError("CartesianBlockHeader requires at least one dimension")
        if any(d <= 0 for d in dims):
            raise SPFError("CartesianBlockHeader dims must be positive")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "profile", CartesianProfile(self.profile))
        if int(self.version) != CARTESIAN_TENSOR_MAJOR:
            raise SPFError(f"unsupported CartesianTensor version {self.version}")
        if int(self.rank) < 0:
            raise SPFError("CartesianTensor rank must be non-negative")
        if float(self.zero_tol) < 0.0 or float(self.rel_tol) < 0.0:
            raise SPFError("CartesianTensor tolerances must be non-negative")

    @property
    def total_elements(self) -> int:
        return int(np.prod(self.dims, dtype=np.int64))

    def to_metadata(self) -> dict[str, Any]:
        return {
            "cartesian_tensor_major": CARTESIAN_TENSOR_MAJOR,
            "cartesian_tensor_minor": CARTESIAN_TENSOR_MINOR,
            "dtype": CARTESIAN_TENSOR_DTYPE,
            "dims": list(self.dims),
            "profile": self.profile.value,
            "rank": int(self.rank),
            "zero_tol": float(self.zero_tol),
            "rel_tol": float(self.rel_tol),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CartesianBlockAnalysis:
    """Observed Cartesian-domain meta-patterns for a dense tensor block."""

    dims: tuple[int, ...]
    best_profile: CartesianProfile
    bytes_dense: int
    bytes_cartesian: int
    compression_ratio: float
    rel_recon_err: float
    sparsity: float
    real_only: bool
    imag_only: bool
    conjugate_symmetry_error: float
    rank1_rel_l2: float
    rank2_rel_l2: float
    cancellation_coherence: float
    dynamic_range: float

    def to_metadata(self) -> dict[str, int | float | str | bool | list[int]]:
        return {
            "dims": list(self.dims),
            "best_profile": self.best_profile.value,
            "bytes_dense": self.bytes_dense,
            "bytes_cartesian": self.bytes_cartesian,
            "compression_ratio": self.compression_ratio,
            "rel_recon_err": self.rel_recon_err,
            "sparsity": self.sparsity,
            "real_only": self.real_only,
            "imag_only": self.imag_only,
            "conjugate_symmetry_error": self.conjugate_symmetry_error,
            "rank1_rel_l2": self.rank1_rel_l2,
            "rank2_rel_l2": self.rank2_rel_l2,
            "cancellation_coherence": self.cancellation_coherence,
            "dynamic_range": self.dynamic_range,
        }


@dataclass(slots=True)
class CartesianTensorBlock:
    """Structured tensor block in real/imaginary coordinates."""

    header: CartesianBlockHeader
    dense_values: npt.ArrayLike | None = None
    values: npt.ArrayLike | None = None
    indices: npt.ArrayLike | None = None
    left: npt.ArrayLike | None = None
    right: npt.ArrayLike | None = None
    half_spectrum: npt.ArrayLike | None = None

    @classmethod
    def from_dense(
        cls,
        values: npt.ArrayLike,
        *,
        profile: CartesianProfile | str | None = None,
        max_rank: int = 2,
        zero_tol: float = 1e-12,
        rel_tol: float = 1e-9,
        max_sparse_fraction: float = 0.20,
    ) -> CartesianTensorBlock:
        """Fit a CartesianTensor block, selecting the smallest valid profile."""

        arr = _as_complex(values)
        if profile is not None:
            return _block_for_profile(
                arr,
                CartesianProfile(profile),
                max_rank=max_rank,
                zero_tol=zero_tol,
                rel_tol=rel_tol,
            )

        candidates: list[CartesianTensorBlock] = [
            _dense_block(arr, zero_tol=zero_tol, rel_tol=rel_tol)
        ]
        if _is_real_only(arr, zero_tol):
            candidates.append(_real_only_block(arr, zero_tol=zero_tol, rel_tol=rel_tol))
        if _is_imag_only(arr, zero_tol):
            candidates.append(_imag_only_block(arr, zero_tol=zero_tol, rel_tol=rel_tol))
        sparse_fraction = _sparsity(arr, zero_tol)
        if sparse_fraction <= max_sparse_fraction:
            candidates.append(_sparse_block(arr, zero_tol=zero_tol, rel_tol=rel_tol))
        if arr.ndim == 1 and _conjugate_symmetry_error(arr) <= rel_tol:
            candidates.append(_conjugate_symmetric_block(arr, zero_tol=zero_tol, rel_tol=rel_tol))
        if arr.ndim == 2:
            low_rank = _low_rank_block(
                arr,
                max_rank=max_rank,
                zero_tol=zero_tol,
                rel_tol=rel_tol,
                require_tol=True,
            )
            if low_rank is not None:
                candidates.append(low_rank)

        return min(candidates, key=lambda b: (b.bytes_estimate(), b.reconstruction_error(arr)))

    def reconstruct(self) -> np.ndarray:
        profile = self.header.profile
        dims = self.header.dims

        if profile == CartesianProfile.DENSE:
            return _require_array(self.dense_values, "dense_values", np.complex128).reshape(dims)
        if profile == CartesianProfile.REAL_ONLY:
            real = _require_array(self.values, "values", np.float64).reshape(dims)
            return real.astype(np.complex128)
        if profile == CartesianProfile.IMAG_ONLY:
            imag = _require_array(self.values, "values", np.float64).reshape(dims)
            return 1j * imag.astype(np.complex128)
        if profile == CartesianProfile.SPARSE_COO:
            out = np.zeros(dims, dtype=np.complex128)
            idx = _require_array(self.indices, "indices", np.int64)
            vals = _require_array(self.values, "values", np.complex128)
            _validate_sparse(idx, vals, dims)
            if idx.size:
                flat_idx = np.ravel_multi_index(idx.T, dims)
                flat = out.reshape(-1)
                if np.unique(flat_idx).size == flat_idx.size:
                    flat[flat_idx] = vals
                else:
                    for linear, value in zip(flat_idx, vals, strict=True):
                        flat[int(linear)] = value
            return out
        if profile == CartesianProfile.LOW_RANK:
            left = _require_array(self.left, "left", np.complex128)
            right = _require_array(self.right, "right", np.complex128)
            if len(dims) != 2:
                raise SPFError("LOW_RANK CartesianTensor requires order-2 dims")
            if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
                raise SPFError("LOW_RANK factors have incompatible shapes")
            return np.asarray(left @ right, dtype=np.complex128).reshape(dims)
        if profile == CartesianProfile.CONJUGATE_SYMMETRIC:
            if len(dims) != 1:
                raise SPFError("CONJUGATE_SYMMETRIC CartesianTensor requires a vector")
            half = _require_array(self.half_spectrum, "half_spectrum", np.complex128)
            return _reconstruct_conjugate_symmetric(half, dims[0])
        raise SPFError(f"unsupported Cartesian profile {profile!r}")

    def reconstruction_error(self, reference: npt.ArrayLike) -> float:
        return _rel_l2(self.reconstruct(), _as_complex(reference))

    def bytes_estimate(self) -> int:
        return estimate_cartesian_bytes(self)

    def to_metadata(self) -> dict[str, Any]:
        meta = self.header.to_metadata()
        meta["bytes_estimate"] = self.bytes_estimate()
        return meta


def analyze_cartesian_patterns(
    values: npt.ArrayLike,
    *,
    max_rank: int = 2,
    zero_tol: float = 1e-12,
    rel_tol: float = 1e-9,
) -> CartesianBlockAnalysis:
    """Return Cartesian-visible structure metrics and selected block profile."""

    arr = _as_complex(values)
    block = CartesianTensorBlock.from_dense(
        arr,
        max_rank=max_rank,
        zero_tol=zero_tol,
        rel_tol=rel_tol,
    )
    bytes_dense = dense_cartesian_bytes(arr.shape)
    bytes_cart = block.bytes_estimate()
    rank1 = _rank_rel_l2(arr, 1) if arr.ndim == 2 else float("nan")
    rank2 = _rank_rel_l2(arr, 2) if arr.ndim == 2 else float("nan")
    return CartesianBlockAnalysis(
        dims=tuple(int(d) for d in arr.shape),
        best_profile=block.header.profile,
        bytes_dense=bytes_dense,
        bytes_cartesian=bytes_cart,
        compression_ratio=float(bytes_dense / bytes_cart) if bytes_cart else float("inf"),
        rel_recon_err=block.reconstruction_error(arr),
        sparsity=_sparsity(arr, zero_tol),
        real_only=_is_real_only(arr, zero_tol),
        imag_only=_is_imag_only(arr, zero_tol),
        conjugate_symmetry_error=_conjugate_symmetry_error(arr) if arr.ndim == 1 else float("nan"),
        rank1_rel_l2=rank1,
        rank2_rel_l2=rank2,
        cancellation_coherence=_cancellation_coherence(arr),
        dynamic_range=_dynamic_range(arr, zero_tol),
    )


def dense_cartesian_bytes(shape: tuple[int, ...] | npt.ArrayLike) -> int:
    n = int(np.prod(shape, dtype=np.int64)) if isinstance(shape, tuple) else int(np.asarray(shape).size)
    return n * COMPLEX128_BYTES


def estimate_cartesian_bytes(block: CartesianTensorBlock) -> int:
    profile = block.header.profile
    n = block.header.total_elements
    order = len(block.header.dims)
    if profile == CartesianProfile.DENSE:
        return n * COMPLEX128_BYTES
    if profile in {CartesianProfile.REAL_ONLY, CartesianProfile.IMAG_ONLY}:
        return CARTESIAN_HEADER_BYTES + n * FLOAT64_BYTES
    if profile == CartesianProfile.SPARSE_COO:
        idx = _require_array(block.indices, "indices", np.int64)
        nnz = int(idx.shape[0])
        return CARTESIAN_HEADER_BYTES + nnz * (order * INT64_BYTES + COMPLEX128_BYTES)
    if profile == CartesianProfile.LOW_RANK:
        left = _require_array(block.left, "left", np.complex128)
        right = _require_array(block.right, "right", np.complex128)
        entries = int(left.size + right.size)
        return CARTESIAN_HEADER_BYTES + entries * COMPLEX128_BYTES
    if profile == CartesianProfile.CONJUGATE_SYMMETRIC:
        half = _require_array(block.half_spectrum, "half_spectrum", np.complex128)
        return CARTESIAN_HEADER_BYTES + int(half.size) * COMPLEX128_BYTES
    raise SPFError(f"unsupported Cartesian profile {profile!r}")


def _block_for_profile(
    arr: np.ndarray,
    profile: CartesianProfile,
    *,
    max_rank: int,
    zero_tol: float,
    rel_tol: float,
) -> CartesianTensorBlock:
    if profile == CartesianProfile.DENSE:
        return _dense_block(arr, zero_tol=zero_tol, rel_tol=rel_tol)
    if profile == CartesianProfile.REAL_ONLY:
        if not _is_real_only(arr, zero_tol):
            raise SPFError("requested REAL_ONLY profile for tensor with imaginary content")
        return _real_only_block(arr, zero_tol=zero_tol, rel_tol=rel_tol)
    if profile == CartesianProfile.IMAG_ONLY:
        if not _is_imag_only(arr, zero_tol):
            raise SPFError("requested IMAG_ONLY profile for tensor with real content")
        return _imag_only_block(arr, zero_tol=zero_tol, rel_tol=rel_tol)
    if profile == CartesianProfile.SPARSE_COO:
        return _sparse_block(arr, zero_tol=zero_tol, rel_tol=rel_tol)
    if profile == CartesianProfile.LOW_RANK:
        block = _low_rank_block(arr, max_rank=max_rank, zero_tol=zero_tol, rel_tol=rel_tol, require_tol=False)
        if block is None:
            raise SPFError("LOW_RANK profile requires an order-2 tensor")
        return block
    if profile == CartesianProfile.CONJUGATE_SYMMETRIC:
        if arr.ndim != 1:
            raise SPFError("CONJUGATE_SYMMETRIC profile requires a vector")
        return _conjugate_symmetric_block(arr, zero_tol=zero_tol, rel_tol=rel_tol)
    raise SPFError(f"unsupported Cartesian profile {profile!r}")


def _dense_block(arr: np.ndarray, *, zero_tol: float, rel_tol: float) -> CartesianTensorBlock:
    return CartesianTensorBlock(
        header=CartesianBlockHeader(arr.shape, CartesianProfile.DENSE, zero_tol=zero_tol, rel_tol=rel_tol),
        dense_values=np.asarray(arr, dtype=np.complex128).copy(),
    )


def _real_only_block(arr: np.ndarray, *, zero_tol: float, rel_tol: float) -> CartesianTensorBlock:
    return CartesianTensorBlock(
        header=CartesianBlockHeader(arr.shape, CartesianProfile.REAL_ONLY, zero_tol=zero_tol, rel_tol=rel_tol),
        values=np.asarray(arr.real, dtype=np.float64).copy(),
    )


def _imag_only_block(arr: np.ndarray, *, zero_tol: float, rel_tol: float) -> CartesianTensorBlock:
    return CartesianTensorBlock(
        header=CartesianBlockHeader(arr.shape, CartesianProfile.IMAG_ONLY, zero_tol=zero_tol, rel_tol=rel_tol),
        values=np.asarray(arr.imag, dtype=np.float64).copy(),
    )


def _sparse_block(arr: np.ndarray, *, zero_tol: float, rel_tol: float) -> CartesianTensorBlock:
    mask = np.abs(arr) > zero_tol
    idx = np.argwhere(mask).astype(np.int64)
    vals = arr[mask].astype(np.complex128)
    return CartesianTensorBlock(
        header=CartesianBlockHeader(
            arr.shape,
            CartesianProfile.SPARSE_COO,
            zero_tol=zero_tol,
            rel_tol=rel_tol,
            metadata={"nnz": int(vals.size)},
        ),
        indices=idx,
        values=vals,
    )


def _low_rank_block(
    arr: np.ndarray,
    *,
    max_rank: int,
    zero_tol: float,
    rel_tol: float,
    require_tol: bool,
) -> CartesianTensorBlock | None:
    if arr.ndim != 2:
        return None
    rank = max(1, min(int(max_rank), min(arr.shape)))
    u, s, vh = np.linalg.svd(arr, full_matrices=False)
    best_rank = rank
    for r in range(1, rank + 1):
        approx = (u[:, :r] * s[:r]) @ vh[:r, :]
        if _rel_l2(approx, arr) <= rel_tol:
            best_rank = r
            break
    approx = (u[:, :best_rank] * s[:best_rank]) @ vh[:best_rank, :]
    if require_tol and _rel_l2(approx, arr) > rel_tol:
        return None
    return CartesianTensorBlock(
        header=CartesianBlockHeader(
            arr.shape,
            CartesianProfile.LOW_RANK,
            rank=best_rank,
            zero_tol=zero_tol,
            rel_tol=rel_tol,
        ),
        left=np.asarray(u[:, :best_rank] * s[:best_rank], dtype=np.complex128),
        right=np.asarray(vh[:best_rank, :], dtype=np.complex128),
    )


def _conjugate_symmetric_block(arr: np.ndarray, *, zero_tol: float, rel_tol: float) -> CartesianTensorBlock:
    if arr.ndim != 1:
        raise SPFError("conjugate-symmetric block requires a vector")
    n = int(arr.shape[0])
    half = arr[: n // 2 + 1].astype(np.complex128)
    return CartesianTensorBlock(
        header=CartesianBlockHeader(
            arr.shape,
            CartesianProfile.CONJUGATE_SYMMETRIC,
            zero_tol=zero_tol,
            rel_tol=rel_tol,
        ),
        half_spectrum=half,
    )


def _as_complex(values: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex128)
    if arr.size == 0:
        raise SPFError("CartesianTensorBlock cannot be empty")
    return arr


def _require_array(values: npt.ArrayLike | None, name: str, dtype: Any) -> np.ndarray:
    if values is None:
        raise SPFError(f"CartesianTensorBlock missing {name}")
    return np.asarray(values, dtype=dtype)


def _validate_sparse(indices: np.ndarray, values: np.ndarray, dims: tuple[int, ...]) -> None:
    if indices.ndim != 2 or indices.shape[1] != len(dims):
        raise SPFError("sparse indices must have shape (nnz, order)")
    if values.ndim != 1 or values.shape[0] != indices.shape[0]:
        raise SPFError("sparse values must have shape (nnz,)")
    if indices.size:
        lo_ok = bool(np.all(indices >= 0))
        hi_ok = bool(np.all(indices < np.asarray(dims, dtype=np.int64)))
        if not (lo_ok and hi_ok):
            raise SPFError("sparse indices out of bounds")


def _is_real_only(arr: np.ndarray, zero_tol: float) -> bool:
    return bool(np.max(np.abs(arr.imag)) <= zero_tol)


def _is_imag_only(arr: np.ndarray, zero_tol: float) -> bool:
    return bool(np.max(np.abs(arr.real)) <= zero_tol)


def _sparsity(arr: np.ndarray, zero_tol: float) -> float:
    return float(np.count_nonzero(np.abs(arr) > zero_tol) / arr.size)


def _conjugate_symmetry_error(arr: np.ndarray) -> float:
    if arr.ndim != 1:
        return float("nan")
    n = int(arr.shape[0])
    if n == 1:
        return 0.0 if abs(arr[0].imag) == 0.0 else float("inf")
    ref = np.conj(np.roll(arr[::-1], 1))
    return _rel_l2(arr, ref)


def _reconstruct_conjugate_symmetric(half: np.ndarray, n: int) -> np.ndarray:
    expected = n // 2 + 1
    if int(half.size) != expected:
        raise SPFError(f"half_spectrum length {half.size} does not match N={n}")
    if n == 1:
        return np.asarray([half[0]], dtype=np.complex128)
    tail = np.conj(half[1:-1][::-1]) if n % 2 == 0 else np.conj(half[1:][::-1])
    return np.concatenate([half, tail]).astype(np.complex128)


def _rank_rel_l2(arr: np.ndarray, rank: int) -> float:
    if arr.ndim != 2:
        return float("nan")
    r = max(1, min(int(rank), min(arr.shape)))
    u, s, vh = np.linalg.svd(arr, full_matrices=False)
    approx = (u[:, :r] * s[:r]) @ vh[:r, :]
    return _rel_l2(approx, arr)


def _cancellation_coherence(arr: np.ndarray) -> float:
    denom = float(np.sum(np.abs(arr)))
    if denom == 0.0:
        return 1.0
    return float(abs(np.sum(arr)) / denom)


def _dynamic_range(arr: np.ndarray, zero_tol: float) -> float:
    mag = np.abs(arr)
    nz = mag[mag > zero_tol]
    if nz.size == 0:
        return 0.0
    return float(np.log2(float(np.max(nz)) / float(np.min(nz))))


def _rel_l2(got: npt.ArrayLike, ref: npt.ArrayLike) -> float:
    g = np.asarray(got, dtype=np.complex128)
    r = np.asarray(ref, dtype=np.complex128)
    denom = float(np.linalg.norm(r.ravel()))
    if denom == 0.0:
        return float(np.linalg.norm(g.ravel()))
    return float(np.linalg.norm((g - r).ravel()) / denom)
