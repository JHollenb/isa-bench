"""Bounded-scratch terminal consumption of selected quadratic field blocks."""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np

from spf.renderer._jet_kernel import JetCommand, JetStats, batch_kernel, coefficients, pointer
from spf.renderer._jet_kernel import kernel as kernel
from spf.renderer.jets import AdaptiveJetField, SpectralJetTile2D
from spf.tensor import v02


@dataclass(frozen=True)
class Emission:
    values: np.ndarray
    logs: np.ndarray
    phases: np.ndarray | None
    log_error: float
    phase_error: float
    stats: JetStats
    allocations: tuple[tuple[str, int, int], ...]
    tiles: int
    backend: str


def _upper(c: np.ndarray, x: float, y: float, rx: float, ry: float) -> float:
    center = c[0] + c[1] * x + c[2] * y + 0.5 * c[3] * x * x + c[4] * x * y + 0.5 * c[5] * y * y
    variation = (
        abs(c[1] + c[3] * x + c[4] * y) * rx
        + abs(c[2] + c[4] * x + c[5] * y) * ry
        + 0.5 * max(c[3], 0) * rx * rx
        + abs(c[4]) * rx * ry
        + 0.5 * max(c[5], 0) * ry * ry
    )
    rounding = 64 * np.finfo(float).eps * (1 + float(np.abs(c).sum()) * (1 + abs(x) + abs(y) + rx + ry) ** 2)
    return float(center + variation + rounding)


def _local_values(jet: v02.Jet2D, shape: tuple[int, int], origin: tuple[float, float]) -> np.ndarray:
    from spf.renderer.native import jet_forward_differences_vectorized

    local = v02.jet_transform(jet, np.eye(2), np.asarray(origin))
    return np.asarray(jet_forward_differences_vectorized(local, shape), dtype=np.float64)


@lru_cache(maxsize=4)
def _display_lut(squared: bool) -> np.ndarray:
    codes = np.arange(v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE + 1, dtype=float)
    values = np.exp2((2 if squared else 1) * codes / v02.SPF32_L_SCALE)
    mapped = values / (1 + values)
    srgb = np.where(mapped <= 0.0031308, 12.92 * mapped, 1.055 * mapped ** (1 / 2.4) - 0.055)
    return np.asarray(np.rint(np.clip(srgb, 0, 1) * 255))


def emit_selected(
    payload: Any,
    shape: tuple[int, int],
    origin: tuple[float, float],
    support_slices: tuple[slice, slice],
    *,
    semantic: str,
    squared: bool,
    epsilon: float,
    magnitude_lut: np.ndarray,
    intensity_lut: np.ndarray,
    retain_codes: bool,
    output: np.ndarray | None = None,
    _commands: list[Any] | None = None,
) -> Emission | None:
    """Return None when the optional kernel or a supported bounded header is absent."""
    fn = kernel()
    if isinstance(payload, SpectralJetTile2D):
        if payload.shape != shape:
            raise ValueError("selected jet shape does not match output")
        tiles = [(payload, (slice(0, shape[0]), slice(0, shape[1])))]
    elif isinstance(payload, AdaptiveJetField):
        if payload.shape != shape:
            raise ValueError("adaptive jet shape does not match output")
        tiles = list(zip(payload.tiles, payload.slices, strict=True))
    else:
        return None
    real = semantic in {"real_field", "complex"}
    prepared = []
    for tile, target in tiles:
        lc = coefficients(tile.log_magnitude)
        pc = coefficients(tile.phase) if real else None
        # The double carrier retains guarded integer/half-integer state exactly
        # in this range. Larger or nonfinite state stays on the reference path.
        extent = max(*tile.shape, *(abs(v) for v in tile.origin), 1)
        if any(
            not np.all(np.isfinite(c)) or float(np.abs(c).sum()) * (2 * extent + 1) ** 2 >= 2**50
            for c in (lc, pc)
            if c is not None
        ):
            return None
        prepared.append((tile, target, lc, pc))
    display = semantic == "display_rgb"
    dtype = np.dtype(np.uint8 if display else np.complex128 if semantic == "complex" else np.float32)
    if output is not None and (
        output.shape != shape
        or output.dtype != dtype
        or not output.flags.writeable
        or any(stride <= 0 or stride % dtype.itemsize for stride in output.strides)
    ):
        raise ValueError("terminal output must be a writable, positively strided array with the requested shape/dtype")
    values = np.empty(shape, dtype=dtype) if output is None else output
    values.fill(-math.inf if semantic == "log_radiance" else 0)
    logs = np.full(shape, v02.SPF32_L_ZERO, dtype=np.int32) if retain_codes else np.empty(0, dtype=np.int32)
    phases = np.zeros(shape, dtype=np.uint16) if real and retain_codes else None
    allocations: list[tuple[str, int, int]] = []
    if retain_codes:
        allocations.append(("log_codes", logs.size, logs.nbytes))
    if phases is not None:
        allocations.append(("phase_codes", phases.size, phases.nbytes))
    threshold = (
        -math.inf
        if epsilon == 0
        else v02.SPF32_L_SCALE * math.log2(max(math.sqrt(epsilon) if squared else epsilon, np.finfo(float).tiny))
    )
    lut = _display_lut(squared) if display else intensity_lut if squared else magnitude_lut
    semantic_code = (
        3 if display else 2 if semantic == "log_radiance" else 4 if semantic == "complex" else 1 if squared else 0
    )
    stats = JetStats(max_log=v02.SPF32_L_ZERO)
    log_error = phase_error = 0.0
    sy, sx = support_slices
    for tile, (ty, tx), lc, pc in prepared:
        allocations.append(
            ("jet_header", lc.size + (0 if pc is None else pc.size), lc.nbytes + (0 if pc is None else pc.nbytes))
        )
        error = float(tile.diagnostics.max_abs_log_error) * v02.SPF32_L_SCALE + 0.5
        log_error = max(log_error, error)
        if real:
            phase_error = max(
                phase_error, float(tile.diagnostics.max_phase_error) * v02.SPF32_PHASE_MOD / (2 * math.pi) + 0.5
            )
        y0, y1 = max(0, sy.start - ty.start), min(tile.shape[0], sy.stop - ty.start)
        x0, x1 = max(0, sx.start - tx.start), min(tile.shape[1], sx.stop - tx.start)
        if y0 >= y1 or x0 >= x1:
            continue
        lr, pr = tile.log_residual, tile.phase_residual if real else None
        # Nonempty residuals are decoded only for bounded patches. A header-only
        # field enters C once, and C performs tile culling and pixel emission.
        has_residual = any(r is not None and r.max_abs_value > 0 for r in (lr, pr))
        bounded_patches = has_residual or tile.zero_mask is not None or fn is None
        patches = (
            ((y0, y1, x0, x1),)
            if not bounded_patches
            else ((y, min(y + 32, y1), x, min(x + 32, x1)) for y in range(y0, y1, 32) for x in range(x0, x1, 32))
        )
        for by, ey, bx, ex in patches:
            if bounded_patches:
                residual_bound = 0 if lr is None else lr.max_abs_value
                bound = _upper(
                    lc,
                    tile.origin[0] + 0.5 * (bx + ex - 1),
                    tile.origin[1] + 0.5 * (by + ey - 1),
                    0.5 * (ex - bx - 1),
                    0.5 * (ey - by - 1),
                )
                if (bound + residual_bound) / (1 << tile.log_magnitude.guard_L) + 0.5 + error <= threshold:
                    stats.tiles += 1
                    stats.culled += 1
                    continue
                yy, xx = np.ogrid[by:ey, bx:ex]
                linear = yy * tile.shape[1] + xx
                lpatch = None if lr is None else np.ascontiguousarray(lr.sample_flat(linear), dtype=np.float64)
                ppatch = None if pr is None else np.ascontiguousarray(pr.sample_flat(linear), dtype=np.float64)
                if tile.zero_mask is not None:
                    if lpatch is None:
                        lpatch = np.zeros((ey - by, ex - bx))
                    lpatch[tile.zero_mask[by:ey, bx:ex]] = -np.inf
                h, w = ey - by, ex - bx
                ox, oy = tile.origin[0] + bx, tile.origin[1] + by
                ranges = (0, w, 0, h)
            else:
                lpatch = ppatch = None
                h, w = tile.shape
                ox, oy = tile.origin
                ranges = (bx, ex, by, ey)
            for name, data in (("log_residual_tile", lpatch), ("phase_residual_tile", ppatch)):
                if data is not None:
                    allocations.append((name, data.size, data.nbytes))
            base_y = ty.start + (by if bounded_patches else 0)
            base_x = tx.start + (bx if bounded_patches else 0)
            if fn is None:
                log_values = _local_values(tile.log_magnitude, (h, w), (ox, oy))
                if lpatch is not None:
                    log_values += lpatch
                raw = np.rint(log_values / (1 << tile.log_magnitude.guard_L))
                live = raw + error > threshold
                stats.evaluated += raw.size
                stats.retained += int(np.count_nonzero(live))
                stats.low += int(np.count_nonzero(live & (raw < v02.SPF32_L_MIN_FINITE)))
                stats.high += int(np.count_nonzero(live & (raw > v02.SPF32_L_MAX_FINITE)))
                stats.tiles += 1
                codes = np.clip(raw, v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE).astype(np.int32)
                if np.any(live):
                    stats.max_log = max(stats.max_log, int(codes[live].max()))
                projected = codes / v02.SPF32_L_SCALE if semantic_code == 2 else lut[codes - v02.SPF32_L_MIN_FINITE]
                phase_codes = None
                if real:
                    phase_values = _local_values(tile.phase, (h, w), (ox, oy))
                    if ppatch is not None:
                        phase_values += ppatch
                    phase_codes = (np.rint(phase_values / (1 << tile.phase.guard_T)).astype(np.int64) & 65535).astype(
                        np.uint16
                    )
                    angle = phase_codes.astype(float) * (2 * math.pi / 65536)
                    projected = projected * (np.exp(1j * angle) if semantic_code == 4 else np.cos(angle))
                stats.fractional_samples += raw.size
                target = (slice(base_y, base_y + h), slice(base_x, base_x + w))
                # SPF intensity's finite range exceeds float32. Match the C
                # cast and let the caller's output contract inspect infinity.
                with np.errstate(over="ignore", under="ignore"):
                    values[target] = np.where(live, projected, -math.inf if semantic_code == 2 else 0)
                stats.max_output = max(
                    stats.max_output, float(np.max(np.abs(np.where(live, values[target], 0)), initial=0))
                )
                if retain_codes:
                    logs[target] = np.where(live, codes, v02.SPF32_L_ZERO)
                    if phases is not None:
                        assert phase_codes is not None
                        phases[target] = np.where(live, phase_codes, 0)
                # Upper bound includes recurrence, quantization, projection,
                # masks and NumPy expression temporaries; scratch is <=32x32.
                allocations.append(("portable_tile_workspace_upper_bound", raw.size * 32, raw.size * 32 * 8))
                continue

            def at(array: np.ndarray | None, base_y: int = base_y, base_x: int = base_x) -> Any:
                if array is None or not array.size:
                    return None
                return ctypes.c_void_p(array.ctypes.data + base_y * array.strides[0] + base_x * array.strides[1])

            arguments = (
                pointer(lc),
                pointer(pc),
                h,
                w,
                shape[1],
                values.strides[0] // values.itemsize,
                values.strides[1] // values.itemsize,
                *ranges,
                ox,
                oy,
                1.0 / (1 << tile.log_magnitude.guard_L),
                1.0 / (1 << tile.phase.guard_T),
                threshold,
                error,
                0.0 if lpatch is None else float(np.max(lpatch, initial=0)),
                pointer(lpatch),
                pointer(ppatch),
                pointer(lut),
                semantic_code,
                at(values),
                at(logs),
                at(phases),
                ctypes.byref(stats),
            )
            if _commands is None:
                fn(*arguments)
            else:
                _commands.append((arguments, (lc, pc, lpatch, ppatch, lut), base_y, base_x))
    return Emission(
        values,
        logs,
        phases,
        log_error,
        phase_error,
        stats,
        tuple(allocations),
        len(tiles),
        "c_fused_jet_terminal" if fn is not None else "numpy_bounded_tile_terminal",
    )


@dataclass
class PreparedEmission:
    """Owned commands and static error terms. Executions only bind output bases.

    The plan owns all pointer carriers, independent of the source's lifetime.
    Execution buffers/stats are per call, so simultaneous calls do not alias.
    """

    template: Emission
    records: tuple[Any, ...]
    native_commands: Any
    function: Any
    fill_value: float
    prepared_strides: tuple[int, int]

    def commands_for(self, output: np.ndarray) -> Any:
        if output.strides == self.prepared_strides:
            return self.native_commands
        commands = (JetCommand * len(self.records))()
        ctypes.memmove(commands, self.native_commands, ctypes.sizeof(commands))
        for command, record in zip(commands, self.records, strict=True):
            _, _, y, x = record
            command.out_row_stride = output.strides[0] // output.itemsize
            command.out_pixel_stride = output.strides[1] // output.itemsize
            command.out_offset = y * output.strides[0] + x * output.strides[1]
        return commands

    def execute(self, output: np.ndarray | None = None, *, clear: bool = True) -> Emission:
        reference = self.template
        if output is None:
            output = np.empty_like(reference.values)
        if (
            output.shape != reference.values.shape
            or output.dtype != reference.values.dtype
            or not output.flags.writeable
            or any(s <= 0 or s % output.itemsize for s in output.strides)
        ):
            raise ValueError("prepared output has incompatible shape, dtype or strides")
        if clear:
            output.fill(self.fill_value)
        logs = np.full_like(reference.logs, v02.SPF32_L_ZERO)
        phases = None if reference.phases is None else np.zeros_like(reference.phases)
        stats = JetStats(max_log=v02.SPF32_L_ZERO, tiles=reference.stats.tiles, culled=reference.stats.culled)
        self.function(
            self.commands_for(output),
            len(self.records),
            pointer(output),
            pointer(logs) if logs.size else None,
            pointer(phases),
            ctypes.byref(stats),
        )
        return replace(reference, values=output, logs=logs, phases=phases, stats=stats)


def prepare_emission(*args: Any, **kwargs: Any) -> PreparedEmission | None:
    """Compile static coefficients, support ranges, residual tiles and LUT bindings."""
    if kernel() is None:
        return None
    fn = batch_kernel()
    if fn is None:
        return None
    records: list[Any] = []
    template = emit_selected(*args, **kwargs, _commands=records)
    if template is None:
        return None
    commands = (JetCommand * len(records))()
    owned_records = []
    for command, (arguments, owners, y, x) in zip(commands, records, strict=True):
        # Copy pointer-bearing input arrays once: caller mutation must not alter
        # a prepared execution or leave dangling residual-tile pointers.
        owned = (*(None if a is None else np.array(a, copy=True) for a in owners[:4]), owners[4])
        for array in owned:
            if array is not None:
                array.setflags(write=False)
        arguments = list(arguments)
        for index, array in zip((0, 1, 18, 19, 20), owned, strict=True):
            arguments[index] = pointer(array)
        for (name, _), argument in zip(JetCommand._fields_[:22], arguments[:22], strict=True):
            setattr(command, name, argument)
        command.out_offset = y * template.values.strides[0] + x * template.values.strides[1]
        command.log_offset = (y * template.values.shape[1] + x) * 4
        command.phase_offset = (y * template.values.shape[1] + x) * 2
        owned_records.append((None, owned, y, x))
    prepared_strides = (template.values.strides[0], template.values.strides[1])
    # Prepared plans need layout descriptors, not a retained image allocation.
    template = replace(
        template, values=np.broadcast_to(np.array(0, dtype=template.values.dtype), template.values.shape)
    )
    return PreparedEmission(
        template,
        tuple(owned_records),
        commands,
        fn,
        -math.inf if kwargs["semantic"] == "log_radiance" else 0.0,
        prepared_strides,
    )
