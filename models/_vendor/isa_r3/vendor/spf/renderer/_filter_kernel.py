"""Native compact spectral consumer. Build with python -m ..._filter_kernel.

The library is optional; absence selects the declared NumPy consumer. It is
never compiled inside a timed rendering call. Coefficients are chart-transformed
Gaussian/chirp atoms, not a sampled transfer plane.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


def library_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    suffix = "dylib" if sys.platform == "darwin" else "so"
    return Path(os.environ.get("SPF_FILTER_LIBRARY", root / "native" / f"libspf_filter.{suffix}"))


@lru_cache(maxsize=1)
def library() -> Any:
    path = library_path()
    if not path.is_file():
        if os.environ.get("SPF_FILTER_LIBRARY"):
            raise RuntimeError(f"requested filter library does not exist: {path}")
        return None
    lib = ctypes.CDLL(str(path))
    if lib.spf_sfr_filter_abi_version() != 3:
        raise RuntimeError("native filter ABI mismatch")
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.spf_sfr_filter_grid.argtypes = [ptr, ptr] + [integer] * 5 + [ptr, integer, ptr, integer, ptr]
    lib.spf_sfr_filter_grid.restype = integer
    lib.spf_sfr_filter_packed_grid.argtypes = [ptr, ptr] + [integer] * 5 + [ptr, integer, ptr, integer, ptr]
    lib.spf_sfr_filter_packed_grid.restype = integer
    lib.spf_sfr_filter_seed_size.restype = ctypes.c_size_t
    lib.spf_sfr_filter_prepare.argtypes = [ptr, integer, integer, integer, ptr, integer]
    lib.spf_sfr_filter_prepare.restype = integer
    lib.spf_sfr_filter_words.argtypes = [
        ptr,
        ptr,
        ptr,
        ctypes.c_size_t,
        integer,
        ptr,
        ptr,
        integer,
        ptr,
        integer,
        integer,
        integer,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.spf_sfr_filter_words.restype = integer
    return lib


def ptr(array: np.ndarray) -> ctypes.c_void_p:
    return ctypes.c_void_p(array.ctypes.data)


class NativeTransfer:
    def __init__(self, transfer: Any, chart: Any, *, periodic: bool) -> None:
        from spf.renderer.block_filter import chart_matrix

        self.lib = library()
        self.supported = (
            self.lib is not None
            and transfer.atom.dimension == 2
            and not transfer.phase_residuals
            and not transfer.windows
        )
        self.periodic = periodic
        self.seeds = np.empty(0, dtype=np.uint8)
        self.seed_shape: tuple[int, ...] = ()
        self.coefficients = np.empty((0, 12), float)
        self.groups = np.asarray([len(m.atoms) for m in transfer.mixtures], dtype=np.int32)
        if not self.supported:
            return
        matrix = chart_matrix(chart, 2)
        a = np.linalg.inv(matrix).T if periodic else matrix
        b = np.zeros(2) if periodic or not chart.dimension else np.asarray(chart.origin)
        coeff = []
        for atom in (transfer.atom, *(a for m in transfer.mixtures for a in m.atoms)):
            c, h = np.asarray(atom.C), np.asarray(atom.h)
            k = atom.kappa + h @ b - 0.5 * b @ c @ b
            h = a.T @ (h - c @ b)
            c = a.T @ c @ a
            coeff.append(np.asarray([k, h[0], h[1], c[0, 0], c[0, 1], c[1, 1]], complex).view(float))
        self.coefficients = np.ascontiguousarray(coeff)
        self.coefficients.setflags(write=False)
        self.groups.setflags(write=False)
        self.supported = len(coeff) <= 4096

    @property
    def nbytes(self) -> int:
        return self.coefficients.nbytes + self.groups.nbytes + self.seeds.nbytes

    def prepare(self, shape: tuple[int, ...]) -> None:
        if not self.supported or self.seed_shape == shape:
            return
        h, w = shape
        if h <= 0 or w <= 0:
            raise ValueError("native filter grid dimensions must be positive")
        self.seeds = np.empty(0, dtype=np.uint8)
        self.seed_shape = ()
        first = (w + 1) // 2
        blocks = (first + 127) // 128 + (w - first + 127) // 128 if self.periodic else (w + 127) // 128
        size = h * blocks * len(self.coefficients) * self.lib.spf_sfr_filter_seed_size()
        # Huge mixtures keep bounded recurrence state instead of allocating an
        # unexpectedly large command cache. Semantics are unchanged.
        if size > 16 * h * w:
            return
        self.seeds = np.empty(size, dtype=np.uint8)
        status = self.lib.spf_sfr_filter_prepare(
            ptr(self.seeds), h, w, int(self.periodic), ptr(self.coefficients), len(self.coefficients)
        )
        if status:
            raise RuntimeError(f"native filter preparation failed: {status}")
        self.seeds.setflags(write=False)
        self.seed_shape = shape

    def grid(self, spectrum: np.ndarray, *, intensity: bool = False) -> np.ndarray:
        if (
            not self.supported
            or spectrum.ndim != 2
            or spectrum.dtype not in (np.dtype("complex128"), np.dtype("complex64"))
        ):
            raise ValueError("unsupported native spectral grid")
        spectrum = np.ascontiguousarray(spectrum)
        single = spectrum.dtype == np.complex64
        dtype = (np.float32 if single else np.float64) if intensity else spectrum.dtype
        output = np.empty(spectrum.shape, dtype=dtype)
        status = self.lib.spf_sfr_filter_grid(
            ptr(spectrum),
            ptr(output),
            int(single),
            int(intensity),
            *spectrum.shape,
            int(self.periodic),
            ptr(self.coefficients),
            len(self.coefficients),
            ptr(self.groups),
            len(self.groups),
            ptr(self.seeds) if self.seeds.size and self.seed_shape == spectrum.shape else None,
        )
        if status:
            raise RuntimeError(f"native compact filter failed: {status}")
        return output

    def packed_grid(
        self,
        words: np.ndarray,
        *,
        dtype: Any = "complex128",
        intensity: bool = False,
    ) -> np.ndarray:
        """Apply the compact periodic transfer directly to packed SPF32 words.

        The result is the declared Cartesian terminal carrier for the platform
        inverse FFT.  Source projection and transfer evaluation both happen in
        the native consumer; Python never constructs a decoded source plane or
        a sampled transfer plane.
        """

        target = np.dtype(dtype)
        if (
            not self.supported
            or words.ndim != 2
            or words.dtype != np.dtype("uint32")
            or target not in (np.dtype("complex128"), np.dtype("complex64"))
        ):
            raise ValueError("unsupported native packed spectral grid")
        words = np.ascontiguousarray(words)
        single = target == np.dtype("complex64")
        output_dtype = (np.float32 if single else np.float64) if intensity else target
        output = np.empty(words.shape, dtype=output_dtype)
        status = self.lib.spf_sfr_filter_packed_grid(
            ptr(words),
            ptr(output),
            int(single),
            int(intensity),
            *words.shape,
            int(self.periodic),
            ptr(self.coefficients),
            len(self.coefficients),
            ptr(self.groups),
            len(self.groups),
            ptr(self.seeds) if self.seeds.size and self.seed_shape == words.shape else None,
        )
        if status:
            raise RuntimeError(f"native packed compact filter failed: {status}")
        return output

    def words(
        self, words: np.ndarray, addresses: np.ndarray, destination: np.ndarray, output: np.ndarray, view: Any
    ) -> None:
        status = self.lib.spf_sfr_filter_words(
            ptr(words),
            ptr(addresses),
            ptr(destination),
            len(words),
            view.shape[1],
            ptr(output),
            ptr(self.coefficients),
            len(self.coefficients),
            ptr(self.groups),
            len(self.groups),
            int(view.source.observable == "intensity"),
            int(view.observable == "intensity"),
            view.log_offset / (128 if view.observable == "intensity" else 256),
            view.phase_offset * 2 * np.pi / 65536,
        )
        if status:
            raise RuntimeError(f"native packed filter failed: {status}")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    target = library_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "cc",
        "-shared",
        "-fPIC",
        "-O3",
        "-ffp-contract=off",
        str(root / "native/src/scalar/spf32/sfr_filter.c"),
        "-o",
        str(target),
        "-lm",
    ]
    subprocess.run(command, check=True)
    print(target)


if __name__ == "__main__":
    main()
