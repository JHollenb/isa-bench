"""Device splat compositor on integer SPF32 log planes (design §27, §27.7).

The word-plane compositor :func:`spf.renderer.splat_terminal.rasterize_planes`
is the CPU oracle.  This module runs the same three stages on a torch device
with integer tensors only:

1. log-alpha codes per tile entry, either from the tile's
   ``TensorBlockHeaderV2`` log polynomial (ABI v23 ``band_exp`` /
   ``log_quadratic`` / ``log_mixed``) when the widened jet is header-exact, or
   from the v0.2 forward-difference recurrence on the widened jet coefficients
   (IEEE float64 adds in the same order as :func:`spf.tensor.v02.jet_forward_differences`,
   rounded to whole L ticks);
2. alpha complement through the ``LOGADD_MINUS`` ROM, uploaded once per device
   from :mod:`spf.tensor.logadd_rom` (the table ``LogPosPlane.log1m`` uses);
3. colour multiply as saturated L adds with sign planes and the ordered
   front-to-back transport scan ``C' = C + T·c``, ``T' = T·t`` over the
   depth-sorted entries of every tile, batched over tiles of equal shape.

Output words are identical to the CPU plane path.  The float image is the
magnitude-ROM display projection at the one output boundary.

The device is ``cuda`` when :func:`torch.cuda.is_available`, else ``cpu``
(``SPF_SFR_SPLAT_DEVICE`` overrides; ``SPF_SFR_SPLAT_CUDA_DISABLE`` turns
the module off).  torch is optional (``uv sync --extra sfr-cuda``); every
entry point fails closed with a reason when it is absent.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from spf.tensor import v02
from spf.tensor.logadd_rom import LOGADD_G_ZERO, LOGADD_MINUS, LOGADD_PLUS

if TYPE_CHECKING:
    from spf.renderer.splats import SplatRenderResult

_L_ZERO = int(v02.SPF32_L_ZERO)
_NAN_L = int(v02.SPF32_NAN_L)
_L_MIN = int(v02.SPF32_L_MIN_FINITE)
_L_MAX = int(v02.SPF32_L_MAX_FINITE)
_G_ZERO = int(LOGADD_G_ZERO)

CONSUMER = "spf.renderer._splat_cuda.rasterize"


def _import_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found,unused-ignore]
    except Exception as exc:  # pragma: no cover - depends on host extras
        raise ImportError("the device splat compositor requires the optional torch runtime (extra 'sfr-cuda')") from exc
    return torch


def availability() -> tuple[bool, str | None]:
    """Whether a torch device (CUDA or CPU) can run the compositor, and why not."""

    if str(os.environ.get("SPF_SFR_SPLAT_CUDA_DISABLE", "")).strip().lower() in {"1", "true", "yes"}:
        return False, "SPF_SFR_SPLAT_CUDA_DISABLE requested"
    try:
        _import_torch()
    except ImportError as exc:
        return False, str(exc)
    return True, None


def available() -> bool:
    return availability()[0]


def cuda_available() -> bool:
    """True only when torch reports a CUDA device (auto-selection criterion)."""

    ok, _ = availability()
    if not ok:
        return False
    try:
        return bool(_import_torch().cuda.is_available())
    except Exception:  # pragma: no cover - vendor runtime failure
        return False


#: Terminal names that select the fused single-launch CUDA kernel
#: (:mod:`spf.renderer._splat_cuda_fused`, design §36.3 / Appendix B).
FUSED_TERMINALS = ("cuda_fused", "fused", "cuda_kernel", "device_fused")

#: Terminal names that select this device compositor.  ``"cuda"`` is the historic
#: spelling and ``"torch"`` the honest one (the module runs on CPU torch when no
#: CUDA runtime is present); ``spf.renderer.splats.render_splats`` routes both
#: through :func:`resolve_terminal`.
DEVICE_TERMINALS = ("cuda", "torch", "torch_cpu", "torch_cuda", "device", *FUSED_TERMINALS)


def fused_availability() -> tuple[bool, str | None]:
    """Whether the fused one-launch CUDA kernel is built and usable here."""

    from spf.renderer import _splat_cuda_fused

    return _splat_cuda_fused.extension_ready()


def fused_available() -> bool:
    """True when the fused CUDA tile kernel compiles and a CUDA device is present."""

    return fused_availability()[0]


def resolve_terminal(terminal: str) -> str | None:
    """Map a ``terminal=`` request to this compositor, ``"auto"``, or ``None``.

    ``"device"``/``"torch"``/``"cuda"`` all select :func:`rasterize`;
    ``"cuda_fused"``/``"fused"``/``"cuda_kernel"`` select the fused single-launch
    CUDA kernel and return ``"device_fused"``; ``"auto"`` returns ``"auto"`` so
    the caller keeps its own auto policy; anything else is ``None`` (not a device
    terminal).  ``"torch_cuda"`` and the fused spellings additionally require a
    CUDA runtime (the fused one also requires the kernel to build) and raise when
    there is none, so a pinned request never silently degrades.
    """

    name = str(terminal).strip().lower().replace("-", "_")
    if name == "auto":
        return "auto"
    if name not in DEVICE_TERMINALS:
        return None
    if name in FUSED_TERMINALS:
        ready, reason = fused_availability()
        if not ready:
            raise RuntimeError(f"terminal={terminal!r} requested but {reason}")
        return "device_fused"
    if name == "torch_cuda" and not cuda_available():
        raise RuntimeError("terminal='torch_cuda' requested but torch reports no CUDA runtime")
    ok, reason = availability()
    if not ok:
        raise RuntimeError(f"terminal={terminal!r} requested but {reason}")
    return "device"


def select_device(device: str | None = None) -> Any:
    """Resolve the torch device: explicit > ``SPF_SFR_SPLAT_DEVICE`` > cuda-if-available > cpu."""

    torch = _import_torch()
    name = device or os.environ.get("SPF_SFR_SPLAT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(name)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SPF_SFR_SPLAT_DEVICE/device asked for cuda but torch reports no CUDA runtime")
    return resolved


def device_info(device: Any | None = None) -> dict[str, Any]:
    torch = _import_torch()
    resolved = select_device(None if device is None else str(device))
    info: dict[str, Any] = {
        "device": str(resolved),
        "device_type": resolved.type,
        "torch_version": str(torch.__version__),
        "cuda": resolved.type == "cuda",
    }
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        info["device_name"] = str(torch.cuda.get_device_name(index))
        info["cuda_version"] = str(getattr(torch.version, "cuda", None))
    return info


# ---------------------------------------------------------------------------
# ROM tables and the integer plane rules (each mirrors one v02_planes helper)
# ---------------------------------------------------------------------------

_ROM_CACHE: dict[str, tuple[Any, Any]] = {}


def _rom(torch: Any, device: Any) -> tuple[Any, Any]:
    """``(LOGADD_PLUS, LOGADD_MINUS)`` int64 tensors resident on ``device``, built once."""

    key = str(device)
    cached = _ROM_CACHE.get(key)
    if cached is None:
        plus = torch.tensor(np.asarray(LOGADD_PLUS, dtype=np.int64), dtype=torch.int64, device=device)
        minus = torch.tensor(np.asarray(LOGADD_MINUS, dtype=np.int64), dtype=torch.int64, device=device)
        cached = (plus, minus)
        _ROM_CACHE[key] = cached
    return cached


class _Ops:
    """Integer plane arithmetic on one torch device (``v02_planes`` rules, tensorized)."""

    def __init__(self, torch: Any, device: Any) -> None:
        self.torch = torch
        self.device = device
        self.plus, self.minus = _rom(torch, device)

    def clamp_finite(self, x: Any) -> Any:
        return self.torch.clamp(x, _L_MIN, _L_MAX)

    def canonical_logpos(self, L: Any) -> Any:
        """``LogPosPlane.__post_init__``: NaN kept, zero kept, else clipped finite."""

        nan = L == _NAN_L
        zero = L == _L_ZERO
        out = self.clamp_finite(L)
        out = self.torch.where(zero, self.torch.full_like(out, _L_ZERO), out)
        return self.torch.where(nan, self.torch.full_like(out, _NAN_L), out)

    def canonical_signed(self, L: Any, sign: Any) -> tuple[Any, Any]:
        """``LogSignedPlane.__post_init__`` over broadcast ``(L, sign)`` int64 tensors."""

        L, sign = self.torch.broadcast_tensors(L, sign)
        nan = L == _NAN_L
        zero = (L == _L_ZERO) | (sign == 0)
        codes = self.clamp_finite(L)
        codes = self.torch.where(zero, self.torch.full_like(codes, _L_ZERO), codes)
        codes = self.torch.where(nan, self.torch.full_like(codes, _NAN_L), codes)
        signs = self.torch.where(sign > 0, self.torch.ones_like(sign), -self.torch.ones_like(sign))
        signs = self.torch.where(zero | nan, self.torch.zeros_like(signs), signs)
        return codes, signs

    def correction(self, gap: Any, *, minus: bool) -> Any:
        table = self.minus if minus else self.plus
        safe = self.torch.clamp(gap, max=_G_ZERO - 1)
        safe = self.torch.clamp(safe, min=0)
        value = table[safe]
        return self.torch.where(gap >= _G_ZERO, self.torch.zeros_like(value), value)

    def logpos_mul(self, a: Any, b: Any) -> Any:
        """``LogPosPlane.mul`` (``v02.spf_mul`` STRICT_V02 on unit-phase words)."""

        zero = (a == _L_ZERO) | (b == _L_ZERO)
        nan = (a == _NAN_L) | (b == _NAN_L)
        L = self.clamp_finite(a + b)
        L = self.torch.where(zero, self.torch.full_like(L, _L_ZERO), L)
        return self.torch.where(nan, self.torch.full_like(L, _NAN_L), L)

    def log1m(self, alpha: Any) -> Any:
        """``LogPosPlane.log1m``: ``logsub_pos(1, alpha)`` through the minus ROM."""

        nan = alpha == _NAN_L
        b_zero = alpha == _L_ZERO
        gap = -alpha
        result = self.clamp_finite(self.correction(self.torch.clamp(gap, min=0), minus=True))
        result = self.torch.where(gap == 0, self.torch.full_like(result, _L_ZERO), result)
        result = self.torch.where(b_zero, self.torch.zeros_like(result), result)
        return self.torch.where(nan, self.torch.full_like(result, _NAN_L), result)

    def as_signed(self, L: Any) -> tuple[Any, Any]:
        """``LogPosPlane.as_signed``: sign 1 unless zero/NaN."""

        sign = self.torch.where((L == _L_ZERO) | (L == _NAN_L), self.torch.zeros_like(L), self.torch.ones_like(L))
        return L, sign

    def signed_mul(self, a: tuple[Any, Any], b: tuple[Any, Any]) -> tuple[Any, Any]:
        """``LogSignedPlane.mul``: sign product, saturated log add, sentinels."""

        La, Sa = a
        Lb, Sb = b
        nan_a = La == _NAN_L
        nan_b = Lb == _NAN_L
        nan = nan_a | nan_b
        zero = (~nan_a & (Sa == 0)) | (~nan_b & (Sb == 0))
        L = self.clamp_finite(La + Lb)
        S = Sa * Sb
        L = self.torch.where(zero, self.torch.full_like(L, _L_ZERO), L)
        L = self.torch.where(nan, self.torch.full_like(L, _NAN_L), L)
        S = self.torch.where(zero | nan, self.torch.zeros_like(S), S)
        return L, S

    def signed_add(self, a: tuple[Any, Any], b: tuple[Any, Any]) -> tuple[Any, Any]:
        """``LogSignedPlane.add``: ``v02.logadd_signed`` through the plus/minus ROMs."""

        La, Sa = self.torch.broadcast_tensors(*a)
        Lb, Sb = self.torch.broadcast_tensors(*b)
        La, Lb = self.torch.broadcast_tensors(La, Lb)
        Sa, Sb = self.torch.broadcast_tensors(Sa, Sb)
        nan_a = La == _NAN_L
        nan_b = Lb == _NAN_L
        nan = nan_a | nan_b
        a_zero = ~nan_a & (Sa == 0)
        b_zero = ~nan_b & (Sb == 0)
        same = Sa == Sb
        big = self.torch.maximum(La, Lb)
        gap = self.torch.abs(La - Lb)
        plus = self.clamp_finite(big + self.correction(gap, minus=False))
        minus = self.clamp_finite(big + self.correction(gap, minus=True))
        zero_code = self.torch.full_like(big, _L_ZERO)
        L = self.torch.where(same, plus, self.torch.where(gap == 0, zero_code, minus))
        larger = self.torch.where(La > Lb, Sa, Sb)
        S = self.torch.where(same, Sa, self.torch.where(gap == 0, self.torch.zeros_like(Sa), larger))
        L = self.torch.where(b_zero, La, self.torch.where(a_zero, Lb, L))
        S = self.torch.where(b_zero, Sa, self.torch.where(a_zero, Sb, S))
        L = self.torch.where(nan, self.torch.full_like(L, _NAN_L), L)
        S = self.torch.where(nan, self.torch.zeros_like(S), S)
        return L, S


# ---------------------------------------------------------------------------
# Tile planning (host side): header ticks or widened-jet starting values
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _EntryPlan:
    header_route: bool
    ticks: tuple[int, int, int, int, int, int]  # c, gy, gx, hyy2, hxx2, hxy (header ticks)
    start: tuple[float, float, float]  # row_value, row_dx, row_dy at the tile origin (widened units)
    hess: tuple[float, float, float]  # hess_xx, hess_xy, hess_yy (widened units)
    guard: float


def _plan_entry(entry: Any, shape: tuple[int, int], *, tolerance_ticks: float) -> _EntryPlan:
    from spf.tensor.core import SPFError

    jet = entry.spf_log_alpha_jet
    guard = float(1 << int(jet.guard_L))
    ticks = (0, 0, 0, 0, 0, 0)
    header_route = False
    try:
        header, bound = entry.alpha_header(rounding="nearest")
    except (SPFError, ValueError):
        header = None
        bound = math.inf
    if header is not None and bound <= tolerance_ticks:
        header_route = True
        ticks = (
            int(header.global_L_bias),
            int(header.band_exp[0]),
            int(header.band_exp[1]),
            int(header.log_quadratic[0]),
            int(header.log_quadratic[1]),
            int(header.log_mixed[0]),
        )
    # The three starting values of the v0.2 recurrence, evaluated exactly as
    # ``jet_forward_differences`` does (same ``jet_eval`` expression order).
    row_value = v02.jet_eval(jet, 0.0, 0.0)
    row_dx = v02.jet_eval(jet, 1.0, 0.0) - row_value
    row_dy = v02.jet_eval(jet, 0.0, 1.0) - row_value
    return _EntryPlan(
        header_route=header_route,
        ticks=ticks,
        start=(_real(row_value), _real(row_dx), _real(row_dy)),
        hess=(_real(jet.hess_xx), _real(jet.hess_xy), _real(jet.hess_yy)),
        guard=guard,
    )


def _real(value: Any) -> float:
    """The CPU path takes ``np.real`` of the recurrence; real adds of a complex add are identical."""

    return float(np.real(value))


def _alpha_codes_device(
    ops: _Ops,
    plans: list[list[_EntryPlan]],
    shape: tuple[int, int],
    nmax: int,
) -> tuple[Any, dict[str, int]]:
    """``(G, nmax, h, w)`` canonical log-alpha codes for one group of equal-shape tiles."""

    torch = ops.torch
    device = ops.device
    G = len(plans)
    h, w = shape
    routes = {"header_log_jet": 0, "jet_forward_differences": 0}
    header_mask = np.zeros((G, nmax), dtype=bool)
    valid = np.zeros((G, nmax), dtype=bool)
    ticks = np.zeros((G, nmax, 6), dtype=np.int64)
    start = np.zeros((G, nmax, 3), dtype=np.float64)
    hess = np.zeros((G, nmax, 3), dtype=np.float64)
    guard = np.ones((G, nmax), dtype=np.float64)
    for g, tile_plans in enumerate(plans):
        for i, plan in enumerate(tile_plans):
            valid[g, i] = True
            header_mask[g, i] = plan.header_route
            ticks[g, i] = plan.ticks
            start[g, i] = plan.start
            hess[g, i] = plan.hess
            guard[g, i] = plan.guard
            routes["header_log_jet" if plan.header_route else "jet_forward_differences"] += 1

    # Header polynomial (exact integer arithmetic): dims (row y, col x).
    ys = torch.arange(h, dtype=torch.int64, device=device).view(1, 1, h, 1)
    xs = torch.arange(w, dtype=torch.int64, device=device).view(1, 1, 1, w)
    t = torch.tensor(ticks, dtype=torch.int64, device=device)
    c, gy, gx, hyy2, hxx2, hxy = (t[..., k].view(G, nmax, 1, 1) for k in range(6))
    poly = c + gy * ys + gx * xs + hyy2 * ys * ys + hxx2 * xs * xs + hxy * ys * xs
    poly = torch.clamp(poly, max=0)

    # v0.2 forward-difference recurrence in float64, same add order as the CPU path.
    s = torch.tensor(start, dtype=torch.float64, device=device)
    hs = torch.tensor(hess, dtype=torch.float64, device=device)
    hess_xx, hess_xy, hess_yy = hs[..., 0], hs[..., 1], hs[..., 2]
    row_value, row_dx, row_dy = s[..., 0].clone(), s[..., 1].clone(), s[..., 2].clone()
    row_values = torch.empty((G, nmax, h), dtype=torch.float64, device=device)
    row_dxs = torch.empty((G, nmax, h), dtype=torch.float64, device=device)
    for row in range(h):
        row_values[..., row] = row_value
        row_dxs[..., row] = row_dx
        row_value = row_value + row_dy
        row_dy = row_dy + hess_yy
        row_dx = row_dx + hess_xy
    wide = torch.empty((G, nmax, h, w), dtype=torch.float64, device=device)
    value = row_values.clone()
    dx = row_dxs.clone()
    hxx = hess_xx.view(G, nmax, 1)
    for column in range(w):
        wide[..., column] = value
        value = value + dx
        dx = dx + hxx
    recurrence = torch.clamp(torch.round(wide / torch.tensor(guard, device=device).view(G, nmax, 1, 1)), max=0)
    recurrence = recurrence.to(torch.int64)

    mask = torch.tensor(header_mask, dtype=torch.bool, device=device).view(G, nmax, 1, 1)
    codes = torch.where(mask, poly, recurrence)
    validity = torch.tensor(valid, dtype=torch.bool, device=device).view(G, nmax, 1, 1)
    codes = torch.where(validity, codes, torch.full_like(codes, _L_ZERO))
    return ops.canonical_logpos(codes), routes


# ---------------------------------------------------------------------------
# The compositor
# ---------------------------------------------------------------------------


def rasterize(
    bins: Any,
    background: tuple[float, ...],
    depth_limit: Any,
    budget: float,
    early_terminate: bool,
    compare_reference: bool,
    *,
    device: str | None = None,
    header_tolerance_ticks: float = 0.0,
    fused: bool = False,
) -> SplatRenderResult | None:
    """Composite every tile on a torch device; words identical to ``rasterize_planes``.

    ``fused=True`` routes to the single-launch CUDA tile kernel
    (:func:`spf.renderer._splat_cuda_fused.rasterize`, design §36.3 /
    Appendix B), whose output words are the same.

    Returns ``None`` when early termination with a positive budget is
    requested (that path keeps per-pixel tail certificates on the typed CPU
    compositor) or when torch is unavailable.
    """

    if fused:
        from spf.renderer import _splat_cuda_fused

        return _splat_cuda_fused.rasterize(
            bins,
            background,
            depth_limit,
            budget,
            early_terminate,
            compare_reference,
            device=device,
            header_tolerance_ticks=header_tolerance_ticks,
        )

    from spf.renderer.splat_terminal import TypedWordView
    from spf.renderer.splats import (
        SplatRenderResult,
        _empty_outputs,
        _reference_core,
        _report,
        _splat_ingress_summary,
    )
    from spf.renderer.transport_terminal import display_values
    from spf.tensor.v02_planes import LogPosPlane, LogSignedPlane

    if early_terminate and budget > 0.0:
        return None
    ok, reason = availability()
    if not ok:
        return None
    torch = _import_torch()
    resolved = select_device(device)
    ops = _Ops(torch, resolved)
    started = time.perf_counter()

    H, W = bins.image_shape
    color, transmittance, alpha, depth, ids, coverage = _empty_outputs((H, W))
    background_plane = LogSignedPlane.from_values(np.asarray(background, dtype=np.float64))
    bg_L = torch.tensor(background_plane.L.astype(np.int64), dtype=torch.int64, device=resolved)
    bg_S = torch.tensor(background_plane.sign.astype(np.int64), dtype=torch.int64, device=resolved)
    color_L = bg_L.view(1, 1, 3).expand(H, W, 3).clone()
    color_S = bg_S.view(1, 1, 3).expand(H, W, 3).clone()
    t_L = torch.zeros((H, W), dtype=torch.int64, device=resolved)
    depth_t = torch.full((H, W), math.inf, dtype=torch.float64, device=resolved)
    ids_t = torch.full((H, W), -1, dtype=torch.int64, device=resolved)
    limit_t = (
        None
        if depth_limit is None
        else torch.tensor(np.asarray(depth_limit, dtype=np.float64), dtype=torch.float64, device=resolved)
    )

    counters = v02.PerformanceCounters()
    ingress = _splat_ingress_summary(bins)
    counters.fallback_conversions = int(ingress["source_conversion_count"])
    alpha_planes = 0
    alpha_routes = {"header_log_jet": 0, "jet_forward_differences": 0}

    # Group tiles by shape so the ordered scan runs once per group, batched.
    groups: dict[tuple[int, int], list[Any]] = {}
    for tile in bins.tiles:
        if not tile.entries:
            continue
        y0, x0, y1, x1 = tile.bounds
        groups.setdefault((y1 - y0, x1 - x0), []).append(tile)

    for shape, tiles in groups.items():
        h, w = shape
        G = len(tiles)
        nmax = max(len(t.entries) for t in tiles)
        plans = [[_plan_entry(e, shape, tolerance_ticks=header_tolerance_ticks) for e in t.entries] for t in tiles]
        codes, routes = _alpha_codes_device(ops, plans, shape, nmax)
        for key, count in routes.items():
            alpha_routes[key] += count
        for tile, tile_plans in zip(tiles, plans, strict=True):
            n = len(tile.entries)
            alpha_planes += n
            jet_entries = sum(1 for p in tile_plans if not p.header_route)
            counters.jet_samples += jet_entries * h * w
            counters.values_materialized += jet_entries * h * w
            counters.transport_compositions += int(n * h * w)
            counters.positive_logadds += int(n * h * w)
            counters.signed_logadds += int(3 * n * h * w)
            counters.log_ops += int(4 * n * h * w)
            counters.bytes_read += int(n * h * w * 8 + n * 3 * 2 * 8)

        # Per-entry attributes, padded with identity entries (alpha ZERO, colour zero).
        colL = np.full((G, nmax, 3), _L_ZERO, dtype=np.int64)
        colS = np.zeros((G, nmax, 3), dtype=np.int64)
        depths = np.full((G, nmax), math.inf, dtype=np.float64)
        pids = np.full((G, nmax), -1, dtype=np.int64)
        rows0 = np.zeros(G, dtype=np.int64)
        cols0 = np.zeros(G, dtype=np.int64)
        for g, tile in enumerate(tiles):
            rows0[g], cols0[g] = tile.bounds[0], tile.bounds[1]
            for i, entry in enumerate(tile.entries):
                codes_rgb = np.asarray(entry.projected.color_codes, dtype=np.int64)
                colL[g, i] = codes_rgb[:, 0]
                colS[g, i] = codes_rgb[:, 1]
                depths[g, i] = entry.depth
                pids[g, i] = entry.projected.primitive_id
        cL, cS = ops.canonical_signed(
            torch.tensor(colL, dtype=torch.int64, device=resolved),
            torch.tensor(colS, dtype=torch.int64, device=resolved),
        )
        depths_t = torch.tensor(depths, dtype=torch.float64, device=resolved)
        pids_t = torch.tensor(pids, dtype=torch.int64, device=resolved)
        ys = torch.tensor(rows0, dtype=torch.int64, device=resolved).view(G, 1, 1) + torch.arange(
            h, dtype=torch.int64, device=resolved
        ).view(1, h, 1)
        xs = torch.tensor(cols0, dtype=torch.int64, device=resolved).view(G, 1, 1) + torch.arange(
            w, dtype=torch.int64, device=resolved
        ).view(1, 1, w)
        ys = ys.expand(G, h, w)
        xs = xs.expand(G, h, w)
        if limit_t is not None:
            occluded = depths_t.view(G, nmax, 1, 1) >= limit_t[ys, xs].view(G, 1, h, w)
            codes = torch.where(occluded, torch.full_like(codes, _L_ZERO), codes)

        # Ordered front-to-back scan over the depth-sorted entries.
        total_t = torch.zeros((G, h, w), dtype=torch.int64, device=resolved)  # LOG_POS 0 == one
        total_c = (
            torch.full((G, h, w, 3), _L_ZERO, dtype=torch.int64, device=resolved),
            torch.zeros((G, h, w, 3), dtype=torch.int64, device=resolved),
        )
        for i in range(nmax):
            a = codes[:, i]  # (G, h, w)
            step_t = ops.log1m(a)
            alpha_signed = ops.as_signed(a.unsqueeze(-1))
            step_c = ops.signed_mul(alpha_signed, (cL[:, i].view(G, 1, 1, 3), cS[:, i].view(G, 1, 1, 3)))
            prefix = ops.as_signed(total_t.unsqueeze(-1))
            total_c = ops.signed_add(total_c, ops.signed_mul(step_c, prefix))
            total_t = ops.logpos_mul(total_t, step_t)
        bg_term = ops.signed_mul((bg_L.view(1, 1, 1, 3), bg_S.view(1, 1, 1, 3)), ops.as_signed(total_t.unsqueeze(-1)))
        out = ops.signed_add(total_c, bg_term)

        live = codes != _L_ZERO  # (G, nmax, h, w)
        any_live = live.any(dim=1)
        first = torch.argmax(live.to(torch.int64), dim=1)  # first maximal index
        first_depth = torch.gather(depths_t.view(G, nmax, 1, 1).expand(G, nmax, h, w), 1, first.unsqueeze(1)).squeeze(1)
        first_pid = torch.gather(pids_t.view(G, nmax, 1, 1).expand(G, nmax, h, w), 1, first.unsqueeze(1)).squeeze(1)

        color_L[ys, xs] = out[0]
        color_S[ys, xs] = out[1]
        t_L[ys, xs] = total_t
        depth_t[ys, xs] = torch.where(any_live, first_depth, depth_t[ys, xs])
        ids_t[ys, xs] = torch.where(any_live, first_pid, ids_t[ys, xs])

    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    device_ms = 1000.0 * (time.perf_counter() - started)

    color_plane = LogSignedPlane(color_L.cpu().numpy(), color_S.cpu().numpy())
    t_plane = LogPosPlane(t_L.cpu().numpy())
    depth[...] = depth_t.cpu().numpy()
    ids[...] = ids_t.cpu().numpy()
    color_words = color_plane.to_words()
    twords = t_plane.to_words()
    color[...] = display_values(color_plane)
    transmittance[...] = display_values(t_plane)
    alpha[:] = 1.0 - transmittance
    coverage[:] = alpha
    counters.bytes_written = int(color_words.nbytes + twords.nbytes)
    reference = (
        _reference_core(bins, background=cast(tuple[float, float, float], background), opaque_depth=depth_limit)[0]
        if compare_reference
        else None
    )
    info = device_info(resolved)
    backend = "cuda" if resolved.type == "cuda" else "torch_cpu"
    report = _report(
        backend=backend,
        bins=bins,
        counters=counters,
        error_budget=budget,
        color=color,
        reference=reference,
        certificates=(),
        representation="spf_log_alpha_device_transport",
        background=background,
    )
    report = replace(
        report,
        diagnostics={
            **report.diagnostics,
            "execution_class": f"device_{resolved.type}",
            "actual_execution_class": f"device_{resolved.type}",
            "actual_callable": CONSUMER,
            "consumer": CONSUMER,
            "hardware_native": bool(resolved.type == "cuda"),
            "device": info,
            "device_ms": device_ms,
            "block_native_arithmetic": bool(ingress["source_conversion_count"] == 0),
            "typed_objects": "lazy_diagnostic_view",
            "alpha_planes": alpha_planes,
            "alpha_routes": dict(alpha_routes),
            "tile_groups": {f"{h}x{w}": len(tiles) for (h, w), tiles in groups.items()},
            "display_projection": "magnitude_rom",
            "oracle": "spf.renderer.splat_terminal.rasterize_planes",
            "unavailable_reason": reason,
        },
    )
    return SplatRenderResult(
        color,
        transmittance,
        alpha,
        depth,
        ids,
        coverage,
        report,
        bins,
        TypedWordView(color_words),
        TypedWordView(twords),
        color_words,
        twords,
        (),
    )


__all__ = [
    "CONSUMER",
    "DEVICE_TERMINALS",
    "FUSED_TERMINALS",
    "availability",
    "available",
    "cuda_available",
    "device_info",
    "fused_availability",
    "fused_available",
    "rasterize",
    "resolve_terminal",
    "select_device",
]
