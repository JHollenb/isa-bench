"""Quadratic jet tiles, residuals, and adaptive subdivision for SFR v0.2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spf.renderer.carriers import decode_words
from spf.renderer.residuals import ResidualComponent, SparseResidual, residual_quantization_bits
from spf.tensor import v02


def guard_bits_for_tile(
    extent: int | tuple[int, int],
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    target_ticks: float = 0.5,
    minimum: int = 8,
    maximum: int = 30,
) -> int:
    """Guard bits that keep a jet's coefficient rounding under ``target_ticks``.

    Reference §13.4 stores header coefficients in units of ``2**-G`` output
    ticks and recommends ``G_L = G_T = 8`` "for experimentation", adding that
    backends may tune them. ``G`` is *not* tile-size independent: rounding each
    of the six coefficients onto the guard grid by up to half a guard unit moves
    the emitted value at the far corner of the addressed region by at most

    ``w / 2**(G + 1)`` ticks, with ``w = 1 + x + y + x²/2 + x·y + y²/2``

    (the same coefficient weight :func:`spf.renderer.native._jet_rounding_code_bound`
    charges). At ``G = 8`` that is 0.30 tick on a 16×16 tile, 2.7 on 32×32 and
    36.5 on 128×128 — measured, audit item F37 — so one fixed default silently
    loses the low bits of every large tile.

    ``origin`` is ``(x0, y0)`` in the *enclosing field's* coordinates, the same
    convention :class:`SpectralJetTile2D` and
    :func:`spf.tensor.v02.jet_forward_differences` use. The jet polynomial is
    evaluated at absolute coordinates, so what sets the requirement is the
    largest ``|x|``/``|y|`` the region reaches, not its extent: a 16×16 tile at
    column 48 needs the guard bits of a 64-wide field, and subdividing a field
    therefore does not by itself buy guard precision.

    Returns the smallest ``G >= minimum`` whose worst case is ``<= target_ticks``,
    clamped at ``maximum`` (30 keeps ``L_ticks · 2**G`` exact in float64 across
    the whole SPF32 L range; a region needing more than that wants recentring).
    """

    if isinstance(extent, (int, np.integer)):
        rows = columns = int(extent)
    else:
        rows, columns = int(extent[0]), int(extent[1])
    target = float(target_ticks)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("guard target_ticks must be positive and finite")
    if minimum < 0 or maximum < minimum:
        raise ValueError("guard bit bounds must satisfy 0 <= minimum <= maximum")
    x0, y0 = float(origin[0]), float(origin[1])
    x = max(abs(x0), abs(x0 + max(columns - 1, 0)))
    y = max(abs(y0), abs(y0 + max(rows - 1, 0)))
    weight = 1.0 + x + y + 0.5 * x * x + x * y + 0.5 * y * y
    guard = int(minimum)
    while guard < int(maximum) and 0.5 * weight > target * float(1 << guard):
        guard += 1
    return guard


def _design_matrix(shape: tuple[int, int], origin: tuple[float, float]) -> np.ndarray:
    rows, columns = np.meshgrid(
        origin[1] + np.arange(shape[0], dtype=np.float64),
        origin[0] + np.arange(shape[1], dtype=np.float64),
        indexing="ij",
    )
    x = columns.reshape(-1)
    y = rows.reshape(-1)
    return np.stack((np.ones_like(x), x, y, 0.5 * x * x, x * y, 0.5 * y * y), axis=1)


def _fit_plane(values: np.ndarray, origin: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares quadratic fit, snapped onto the §13.4 guard grid.

    ``values`` already carry the ``2**G`` guard factor, so the header coefficient
    unit is 1 and snapping is a plain round-to-nearest. Returning the *snapped*
    fit (not the real-valued one) keeps the residual honest: whatever the snap
    costs is measured and carried by the residual stream instead of silently
    appearing in the emitted plane.
    """

    matrix = _design_matrix(values.shape, origin)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, values.reshape(-1), rcond=None)
    coefficients = np.rint(coefficients)
    fitted = (matrix @ coefficients).reshape(values.shape)
    return coefficients, fitted


def _jet_from_coefficients(coefficients: np.ndarray, *, guard: int, kind: str) -> v02.Jet2D:
    # Integers in units of 2**-guard ticks (Reference §13.4); ``_fit_plane`` has
    # already snapped, and ``np.rint`` here is idempotent for those inputs.
    snapped = np.rint(np.asarray(coefficients, dtype=np.float64))
    return v02.Jet2D(
        value=float(snapped[0]),
        grad_x=float(snapped[1]),
        grad_y=float(snapped[2]),
        hess_xx=float(snapped[3]),
        hess_xy=float(snapped[4]),
        hess_yy=float(snapped[5]),
        guard_L=guard if kind == "log_pos" else 0,
        guard_T=guard if kind == "phase" else 0,
        value_kind=kind,
    )


@dataclass(frozen=True, slots=True)
class JetFitDiagnostics:
    max_abs_log_error: float = 0.0
    max_phase_error: float = 0.0
    max_complex_error: float = 0.0
    residual_energy: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class SpectralJetTile2D:
    """A paired log-magnitude/phase jet plus optional narrow residuals."""

    log_magnitude: v02.Jet2D
    phase: v02.Jet2D
    shape: tuple[int, int]
    origin: tuple[float, float] = (0.0, 0.0)
    log_residual: SparseResidual | None = None
    phase_residual: SparseResidual | None = None
    diagnostics: JetFitDiagnostics = field(default_factory=JetFitDiagnostics)
    source_values: np.ndarray | None = field(default=None, repr=False, compare=False)
    zero_mask: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        shape = (int(self.shape[0]), int(self.shape[1]))
        origin = (float(self.origin[0]), float(self.origin[1]))
        if any(value <= 0 for value in shape):
            raise ValueError("jet tile shape must be positive")
        if self.log_magnitude.value_kind not in {"log", "log_pos"}:
            raise TypeError("log_magnitude jet must use LOG_POS units")
        if self.phase.value_kind != "phase":
            raise TypeError("phase jet must use PHASE units")
        for residual in (self.log_residual, self.phase_residual):
            if residual is not None and residual.shape != shape:
                raise ValueError("jet residual shape must match tile")
        source = None if self.source_values is None else np.asarray(self.source_values, dtype=np.complex128).copy()
        if source is not None:
            if source.shape != shape:
                raise ValueError("jet source_values shape must match tile")
            source.setflags(write=False)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "origin", origin)
        zeros = self.zero_mask
        if source is not None:
            zeros = source == 0
        if zeros is not None:
            zeros = np.asarray(zeros, dtype=bool).copy()
            if zeros.shape != shape:
                raise ValueError("jet zero mask shape must match tile")
            zeros.setflags(write=False)
            if not np.any(zeros):
                zeros = None
        object.__setattr__(self, "zero_mask", zeros)
        object.__setattr__(self, "source_values", None)
        if self.phase_residual is not None and self.log_residual is not None:
            object.__setattr__(self, "phase_residual", self.phase_residual.share_layout(self.log_residual))

    @classmethod
    def fit(
        cls,
        values: Any,
        *,
        origin: tuple[float, float] = (0.0, 0.0),
        guard_L: int | None = None,
        guard_T: int | None = None,
        residual_bits: int | None = 12,
        residual_threshold: float = 0.0,
        precision: Any = None,
    ) -> SpectralJetTile2D:
        """Fit one paired log/phase jet tile.

        ``guard_L`` / ``guard_T`` default to :func:`guard_bits_for_tile` on this
        tile's own extent *and origin*, so coefficient snapping stays under half
        an output tick however large or far from the origin the tile is (§13.4;
        audit F37). Pass an explicit int to pin them.

        ``precision`` is a §52 :class:`~spf.renderer.model.PrecisionMode` (or a
        ``PrecisionContract``): it supplies ``residual_log_bits`` for the log
        residual and ``residual_phase_bits`` for the phase residual, the two
        widths §52 names.  EXACT keeps both residuals in float64.  When it is
        given it replaces ``residual_bits`` (which pins one width for both).
        """

        log_bits = residual_bits
        phase_bits = residual_bits
        if precision is not None:
            log_bits = residual_quantization_bits(precision, component=ResidualComponent.LOG)
            phase_bits = residual_quantization_bits(precision, component=ResidualComponent.PHASE)

        source = np.asarray(values, dtype=np.complex128)
        if source.ndim != 2 or not source.size:
            raise ValueError("jet fitting requires a non-empty 2D field")
        if np.any(~np.isfinite(source.real)) or np.any(~np.isfinite(source.imag)):
            raise ValueError("jet fitting requires finite values")
        shape = (int(source.shape[0]), int(source.shape[1]))
        guard_L = guard_bits_for_tile(shape, origin=origin) if guard_L is None else int(guard_L)
        guard_T = guard_bits_for_tile(shape, origin=origin) if guard_T is None else int(guard_T)
        magnitude = np.abs(source)
        safe = np.maximum(magnitude, np.finfo(np.float64).tiny)
        log_units = np.log2(safe) * v02.SPF32_L_SCALE * (1 << int(guard_L))
        phase_radians = np.unwrap(np.unwrap(np.angle(source), axis=0), axis=1)
        phase_units = phase_radians * (v02.SPF32_PHASE_MOD / (2.0 * math.pi)) * (1 << int(guard_T))
        log_coefficients, log_fitted = _fit_plane(log_units, origin)
        phase_coefficients, phase_fitted = _fit_plane(phase_units, origin)
        log_residual_values = log_units - log_fitted
        phase_residual_values = phase_units - phase_fitted
        log_residual = SparseResidual.from_dense(
            log_residual_values,
            threshold=residual_threshold,
            quantization_bits=log_bits,
            precision=precision,
            component=ResidualComponent.LOG,
        )
        phase_residual = SparseResidual.from_dense(
            phase_residual_values,
            threshold=residual_threshold,
            quantization_bits=phase_bits,
            precision=precision,
            component=ResidualComponent.PHASE,
        )
        log_jet = _jet_from_coefficients(log_coefficients, guard=int(guard_L), kind="log_pos")
        phase_jet = _jet_from_coefficients(phase_coefficients, guard=int(guard_T), kind="phase")
        emitted_log = v02.jet_forward_differences(log_jet, source.shape, origin=origin) + log_residual.to_dense()
        emitted_phase = v02.jet_forward_differences(phase_jet, source.shape, origin=origin) + phase_residual.to_dense()
        log_error = float(np.max(np.abs(emitted_log - log_units))) / (v02.SPF32_L_SCALE * (1 << guard_L))
        phase_error = float(np.max(np.abs(emitted_phase - phase_units))) * (
            2.0 * math.pi / (v02.SPF32_PHASE_MOD * (1 << guard_T))
        )
        words = cls._pack_planes(emitted_log, emitted_phase, guard_L, guard_T, magnitude == 0.0)
        reconstructed = np.asarray(decode_words(words), dtype=np.complex128)
        complex_error = float(np.max(np.abs(reconstructed - source)))
        residual_energy = float(
            np.linalg.norm(log_residual_values.reshape(-1)) ** 2
            + np.linalg.norm(phase_residual_values.reshape(-1)) ** 2
        )
        return cls(
            log_jet,
            phase_jet,
            source.shape,
            origin,
            log_residual,
            phase_residual,
            JetFitDiagnostics(log_error, phase_error, complex_error, residual_energy),
            source,
        )

    @staticmethod
    def _pack_planes(
        log_values: np.ndarray,
        phase_values: np.ndarray,
        guard_L: int,
        guard_T: int,
        zero_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        logs = np.rint(np.asarray(log_values) / (1 << int(guard_L))).astype(np.int64)
        phases = np.rint(np.asarray(phase_values) / (1 << int(guard_T))).astype(np.int64)
        logs = np.clip(logs, v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE)
        if zero_mask is not None:
            logs = np.where(zero_mask, v02.SPF32_L_ZERO, logs)
            phases = np.where(zero_mask, v02.SPF32_T_ZERO, phases)
        return np.asarray(v02.pack_spf32(logs, phases & v02.SPF32_PHASE_MASK), dtype=np.uint32)

    def planes(self) -> tuple[np.ndarray, np.ndarray]:
        log_values = np.asarray(
            v02.jet_forward_differences(self.log_magnitude, self.shape, origin=self.origin),
            dtype=np.float64,
        )
        phase_values = np.asarray(
            v02.jet_forward_differences(self.phase, self.shape, origin=self.origin),
            dtype=np.float64,
        )
        if self.log_residual is not None:
            log_values += self.log_residual.to_dense()
        if self.phase_residual is not None:
            phase_values += self.phase_residual.to_dense()
        return log_values, phase_values

    def materialize_words(self) -> np.ndarray:
        log_values, phase_values = self.planes()
        zero_mask = self.zero_mask
        return self._pack_planes(
            log_values,
            phase_values,
            self.log_magnitude.guard_L,
            self.phase.guard_T,
            zero_mask,
        )

    def materialize(self) -> np.ndarray:
        return np.asarray(decode_words(self.materialize_words()), dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class AdaptiveJetField:
    shape: tuple[int, int]
    tiles: tuple[SpectralJetTile2D, ...]
    slices: tuple[tuple[slice, slice], ...]
    error_bound: float

    def __post_init__(self) -> None:
        shape = (int(self.shape[0]), int(self.shape[1]))
        if len(self.tiles) != len(self.slices):
            raise ValueError("adaptive jet tiles and slices must align")
        bound = float(self.error_bound)
        if not math.isfinite(bound) or bound < 0.0:
            raise ValueError("adaptive jet error bound must be finite and non-negative")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "tiles", tuple(self.tiles))
        object.__setattr__(self, "slices", tuple(self.slices))
        object.__setattr__(self, "error_bound", bound)

    @classmethod
    def fit(
        cls,
        values: Any,
        *,
        error_budget: float,
        max_depth: int = 6,
        min_tile: int = 2,
        guard_L: int | None = None,
        guard_T: int | None = None,
        residual_bits: int | None = None,
    ) -> AdaptiveJetField:
        """Subdivide until each tile meets ``error_budget``.

        ``guard_L`` / ``guard_T`` left at ``None`` let every tile pick its own
        guard bits from its own extent and origin (:func:`guard_bits_for_tile`),
        so every tile is sub-tick wherever it sits in the field; an explicit int
        pins the same value for every tile.
        """

        source = np.asarray(values, dtype=np.complex128)
        if source.ndim != 2 or not source.size:
            raise ValueError("adaptive jet fitting requires a non-empty 2D field")
        budget = float(error_budget)
        if not math.isfinite(budget) or budget < 0.0:
            raise ValueError("adaptive jet error budget must be finite and non-negative")
        depth_limit = int(max_depth)
        tiles: list[SpectralJetTile2D] = []
        slices: list[tuple[slice, slice]] = []

        def recurse(row0: int, row1: int, column0: int, column1: int, depth: int) -> None:
            block = source[row0:row1, column0:column1]
            tile = SpectralJetTile2D.fit(
                block,
                origin=(float(column0), float(row0)),
                guard_L=guard_L,
                guard_T=guard_T,
                residual_bits=residual_bits,
            )
            can_split = depth < depth_limit and max(block.shape) > int(min_tile) and min(block.shape) > 1
            if tile.diagnostics.max_complex_error <= budget or not can_split:
                tiles.append(tile)
                slices.append((slice(row0, row1), slice(column0, column1)))
                return
            row_mid = row0 + (row1 - row0) // 2
            column_mid = column0 + (column1 - column0) // 2
            for next_rows, next_columns in (
                ((row0, row_mid), (column0, column_mid)),
                ((row0, row_mid), (column_mid, column1)),
                ((row_mid, row1), (column0, column_mid)),
                ((row_mid, row1), (column_mid, column1)),
            ):
                if next_rows[0] < next_rows[1] and next_columns[0] < next_columns[1]:
                    recurse(*next_rows, *next_columns, depth + 1)

        recurse(0, source.shape[0], 0, source.shape[1], 0)
        maximum = max((tile.diagnostics.max_complex_error for tile in tiles), default=0.0)
        return cls(source.shape, tuple(tiles), tuple(slices), maximum)

    def materialize_words(self) -> np.ndarray:
        result = np.empty(self.shape, dtype=np.uint32)
        for tile, target in zip(self.tiles, self.slices, strict=True):
            result[target] = tile.materialize_words()
        return result

    def materialize(self) -> np.ndarray:
        return np.asarray(decode_words(self.materialize_words()), dtype=np.complex128)


__all__ = ["AdaptiveJetField", "JetFitDiagnostics", "SpectralJetTile2D", "guard_bits_for_tile"]
