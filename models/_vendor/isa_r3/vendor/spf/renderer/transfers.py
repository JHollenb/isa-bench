"""Analytic spectral diagonals: full quadratic log/phase plus positive mixtures.

Coordinates are physical cycles per spatial unit, in AXIS ORDER: component 0 of
every frequency vector, offset and covariance matrix belongs to array axis 0
(rows).  This is the design v0.2 §17.1 convention fixed 2026-09-18; user-facing
``dx``/``dy`` names are converted at the `FilterChain` boundary.  No transfer
plane is fitted, sampled, or quantized during composition. SPF32 code units
exist only on the paired jets; the Gaussian atom remains in natural-log/radian
units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from spf.tensor import v02


@dataclass(frozen=True)
class SpectralTransfer:
    atom: v02.GaussianChirpAtom
    mixtures: tuple[v02.AtomMixture, ...] = ()
    operations: tuple[str, ...] = ()
    phase_residuals: tuple[np.ndarray, ...] = ()
    windows: tuple[tuple[tuple[int, ...], tuple[float, float], str, bool], ...] = ()
    # Accumulated §17.2/§17.3 source-frequency remap: the transfer multiplies a
    # source spectrum read at ``coordinate @ k`` instead of at ``k``.  ``None``
    # is the identity, i.e. a purely diagonal transfer.
    coordinate: np.ndarray | None = None

    @classmethod
    def identity(cls, dimension: int) -> SpectralTransfer:
        return cls(
            v02.GaussianChirpAtom.from_arrays(
                0j, np.zeros(dimension, complex), np.zeros((dimension, dimension), complex), allow_nondecaying=True
            )
        )

    @property
    def nbytes(self) -> int:
        return (
            sum(a.nbytes for a in self.phase_residuals)
            + 64 * len(self.windows)
            + sum(
                16 * (1 + a.dimension + a.dimension**2)
                for a in (self.atom, *(a for m in self.mixtures for a in m.atoms))
            )
        )

    @property
    def jets(self) -> tuple[v02.Jet2D, v02.Jet2D]:
        if self.atom.dimension != 2:
            raise ValueError("paired Jet2D requires two coordinates")
        result = []
        for part, scale, kind in (("real", 256 / math.log(2), "log_pos"), ("imag", 65536 / (2 * math.pi), "phase")):
            k = getattr(self.atom.kappa, part)
            h, c = getattr(self.atom.h, part), getattr(self.atom.C, part)
            result.append(
                v02.Jet2D(
                    k * scale,
                    h[0] * scale,
                    h[1] * scale,
                    -c[0, 0] * scale,
                    -c[0, 1] * scale,
                    -c[1, 1] * scale,
                    guard_L=0,
                    guard_T=0,
                    value_kind=kind,
                )
            )
        return result[0], result[1]

    def transformed_jets(self, matrix: Any, offset: Any = None) -> tuple[v02.Jet2D, v02.Jet2D]:
        """The paired jets after a §17.2/§17.3 remap, through :func:`v02.jet_transform`.

        The exact jet-representation route for a coordinate transform: the log
        and phase jets are quadratics in ``k``, so the substitution
        ``k -> A^-T k`` is :func:`spf.tensor.v02.jet_transform`.  It agrees
        exactly with the atom route used by :meth:`apply`, which composes the
        same substitution with :func:`spf.tensor.v02.atom_affine`; the §17.3
        ``ΔL = -256 log2|det A|`` shift is not included here because it is a
        constant added to the log jet's value, not part of the substitution.
        """

        remap = np.linalg.inv(np.asarray(matrix, dtype=float)).T
        shift = np.zeros(2) if offset is None else np.asarray(offset, dtype=float).reshape(2)
        log, phase = self.jets
        return v02.jet_transform(log, remap, shift), v02.jet_transform(phase, remap, shift)

    def apply(self, operation: Any) -> SpectralTransfer:
        op, params, n = str(operation.op), operation.params, self.atom.dimension
        kappa, h, c = 0j, np.zeros(n, complex), np.zeros((n, n), complex)
        if op == "translate":
            offset = params.get("offset")
            if offset is None:
                # Legacy spelling: ``dx``/``dy``/``dz`` name the reader's axes, so
                # reverse them into axis order (design §17.1 convention).
                offset = [params.get(name, 0) for name in ("dx", "dy", "dz")[:n]][::-1]
            offset = np.asarray(offset, float)
            if offset.shape != (n,) or not np.all(np.isfinite(offset)):
                raise ValueError("translation dimension/finite-value mismatch")
            h = -2j * math.pi * offset
        elif op == "gaussian_filter":
            covariance = np.asarray(params["covariance"], float)
            if (
                covariance.shape != (n, n)
                or not np.all(np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-14)
                or np.linalg.eigvalsh(covariance).min() < -1e-12
            ):
                raise ValueError("Gaussian covariance must be finite, symmetric and positive semidefinite")
            c = 4 * math.pi**2 * covariance.astype(complex)
        elif op in {"rotate", "affine"}:
            # §17.2 G(k) = F(Rk) and §17.3 G(k) = |det A|^-1 F(A^-T k).  A
            # coordinate remap is not diagonal, so it composes by remapping every
            # factor accumulated so far and recording the source-frequency map;
            # the |det A|^-1 amplitude (ΔL = -256 log2|det A|) and the linear
            # phase exp(i 2 pi (A^-1 b) . k) stay on the diagonal part.
            from spf.renderer.normalization import _affine_params

            if self.phase_residuals or self.windows:
                raise ValueError("a sampled spectral factor cannot be remapped by a coordinate transform")
            matrix, shift = _affine_params(params, n)
            inverse = np.linalg.inv(matrix)
            remap = inverse.T
            zero = np.zeros(n)
            remapped = v02.atom_affine(self.atom, remap, zero)
            mixtures = tuple(
                v02.AtomMixture(tuple(v02.atom_affine(a, remap, zero) for a in m.atoms)) for m in self.mixtures
            )
            kappa = -math.log(abs(float(np.linalg.det(matrix))))
            h = 2j * math.pi * (inverse @ shift)
            factor = v02.GaussianChirpAtom.from_arrays(kappa + 0j, h, c, allow_nondecaying=True)
            return replace(
                self,
                atom=v02.atom_mul(remapped, factor),
                mixtures=mixtures,
                coordinate=remap if self.coordinate is None else self.coordinate @ remap,
                operations=(*self.operations, op),
            )
        elif op == "exposure":
            ev = float(params.get("ev", params.get("exposure", 0)))
            if not math.isfinite(ev):
                raise ValueError("exposure must be finite")
            kappa = ev * math.log(2)
        elif op in {"phase", "phase_shift"}:
            phase = np.asarray(params.get("dT", params.get("phase", params.get("delta", 0))), float)
            if phase.ndim:
                phase = np.broadcast_to(phase, tuple(params["shape"])).copy()
                if not np.all(np.isfinite(phase)):
                    raise ValueError("phase must be finite")
                if params.get("phase_units") not in {"radians", "rad"}:
                    phase *= 2 * math.pi / 65536
                phase.setflags(write=False)
                return replace(self, phase_residuals=(*self.phase_residuals, phase), operations=(*self.operations, op))
            scalar_phase = float(phase)
            if not math.isfinite(scalar_phase):
                raise ValueError("phase must be finite")
            kappa = 1j * scalar_phase * (1 if params.get("phase_units") in {"radians", "rad"} else 2 * math.pi / 65536)
        elif op == "lod":
            shape = tuple(params["shape"])
            cutoffs = np.broadcast_to(np.asarray(params["cutoff"], float), (2,))
            name = str(params.get("window", "raised_cosine")).lower().replace("-", "_").replace(" ", "_")
            if len(shape) != 2 or np.any(cutoffs <= 0) or not np.all(np.isfinite(cutoffs)):
                raise ValueError("LOD requires a 2D shape and positive finite cutoffs")
            if name not in {"hard", "box", "ideal", "hann", "cosine", "raised_cosine", "raised", "tukey"}:
                raise ValueError(f"unsupported LOD window {name!r}")
            window = (shape, (float(cutoffs[0]), float(cutoffs[1])), name, bool(params.get("centered", False)))
            return replace(self, windows=(*self.windows, window), operations=(*self.operations, op))
        elif op == "glare":
            if n != 2:
                raise ValueError("star glare requires two dimensions")
            strength, length, width = (float(params[k]) for k in ("strength", "length", "width"))
            if not (
                0 <= strength <= 1
                and length > 0
                and width > 0
                and all(math.isfinite(v) for v in (strength, length, width))
            ):
                raise ValueError("invalid glare parameters")
            atoms = []
            if strength < 1:
                atoms.append(
                    v02.GaussianChirpAtom.from_arrays(
                        math.log1p(-strength), np.zeros(2), np.zeros((2, 2)), allow_nondecaying=True
                    )
                )
            if strength:
                for angle in (0, math.pi / 3, 2 * math.pi / 3):
                    # Ray angles are read in the reader's frame (x = columns =
                    # axis 1), so the axis-order direction is (sin, cos).
                    u = np.array([math.sin(angle), math.cos(angle)])
                    v = np.array([-u[1], u[0]])
                    covariance = length**2 * np.outer(u, u) + width**2 * np.outer(v, v)
                    atoms.append(
                        v02.GaussianChirpAtom.from_arrays(
                            math.log(strength / 3), np.zeros(2), 4 * math.pi**2 * covariance
                        )
                    )
            return replace(
                self, mixtures=(*self.mixtures, v02.AtomMixture(tuple(atoms))), operations=(*self.operations, op)
            )
        elif op in {"fresnel", "spectral_lens"}:
            wavelength, distance = float(params["wavelength"]), float(params["distance"])
            if not (wavelength > 0 and distance != 0 and math.isfinite(wavelength) and math.isfinite(distance)):
                raise ValueError("invalid propagation parameters")
            c = 2j * math.pi * wavelength * distance * np.eye(n)
            kappa = 2j * math.pi * distance / wavelength if op == "fresnel" else 0j
        else:
            raise ValueError(f"{op!r} has no analytic spectral transfer")
        multiplier = v02.GaussianChirpAtom.from_arrays(kappa, h, c, allow_nondecaying=True)
        return replace(self, atom=v02.atom_mul(self.atom, multiplier), operations=(*self.operations, op))

    def log_phase(
        self, points: Any, addresses: Any = None, *, include_phase: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate only requested coordinates; return log2 magnitude and radians."""
        points = np.asarray(points, float)
        if points.shape[-1] != self.atom.dimension:
            raise ValueError("transfer coordinate dimension mismatch")
        if self.atom.dimension == 2:
            log, phase = self.jets
            x, y = points[..., 0], points[..., 1]

            def evaluate(jet: v02.Jet2D) -> np.ndarray:
                return np.asarray(
                    jet.value
                    + jet.grad_x * x
                    + jet.grad_y * y
                    + 0.5 * jet.hess_xx * x * x
                    + jet.hess_xy * x * y
                    + 0.5 * jet.hess_yy * y * y
                )

            logs = evaluate(log) / 256
            phases = evaluate(phase) * (2 * math.pi / 65536) if include_phase else np.zeros_like(logs)
        else:
            exponent = (
                self.atom.kappa
                + points @ self.atom.h
                - 0.5 * np.einsum("...i,ij,...j->...", points, np.asarray(self.atom.C), points)
            )
            logs, phases = exponent.real / math.log(2), exponent.imag
        for mixture in self.mixtures:
            # Positive Gaussian mixture: stable real log-sum-exp, no phase
            # reduction and no invalid single-quadratic approximation to glare.
            total = np.full(points.shape[:-1], -np.inf)
            for atom in mixture.atoms:
                term = complex(atom.kappa).real - 0.5 * np.einsum(
                    "...i,ij,...j->...", points, np.asarray(atom.C).real, points
                )
                total = np.logaddexp(total, term)
            logs = logs + total / math.log(2)
        if (self.phase_residuals or self.windows) and addresses is None:
            raise ValueError("sampled spectral factors require explicit source addresses")
        for residual in self.phase_residuals:
            if include_phase:
                phases = phases + residual.reshape(-1)[addresses]
        for shape, cutoffs, name, centered in self.windows:
            index = np.asarray(np.unravel_index(addresses, shape)).T
            dims = np.asarray(shape)
            if centered:
                index = (index - dims // 2) % dims
            frequency = np.where(index >= (dims + 1) // 2, index - dims, index) / dims
            radius = np.sqrt((frequency[..., 1] / cutoffs[0]) ** 2 + (frequency[..., 0] / cutoffs[1]) ** 2)
            if name in {"hard", "box", "ideal"}:
                weight = (radius <= 1).astype(float)
            elif name in {"hann", "cosine"}:
                weight = np.where(radius <= 1, 0.5 * (1 + np.cos(math.pi * radius)), 0)
            else:
                weight = np.where(
                    radius <= 0.8, 1, np.where(radius < 1, 0.5 * (1 + np.cos(math.pi * (radius - 0.8) / 0.2)), 0)
                )
            with np.errstate(divide="ignore"):
                logs = logs + np.log2(weight)
        return np.asarray(logs), np.asarray(phases)
