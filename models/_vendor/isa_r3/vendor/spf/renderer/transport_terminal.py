"""Ordered LogPos channel-array consumer using normative integer LUT operations.

Plane arithmetic (``multiply``/``add``/``complement``/ordered composition)
delegates to :mod:`spf.tensor.v02_planes`; the only tables here are the two
terminal ROMs: the ``2**(L/256)`` magnitude table used for error envelopes
and display projection, and the optical-depth-to-transmittance code table.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from spf.tensor import v02
from spf.tensor import v02_planes as planes


def _magnitude_table() -> np.ndarray:
    """The shared magnitude ROM (:func:`spf.renderer.carriers.magnitude_table`)."""

    from spf.renderer.carriers import magnitude_table

    return magnitude_table()


@lru_cache(maxsize=1)
def _logadd_table() -> np.ndarray:
    table = np.asarray(v02.LOGADD_PLUS)
    table.setflags(write=False)
    return table


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``logpos_mul`` on int log-code planes (``LogPosPlane.mul``)."""

    left = np.asarray(a, dtype=np.int64)
    right = np.asarray(b, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("positive-log multiply operands must have matching shapes")
    return planes.LogPosPlane(left).mul(planes.LogPosPlane(right)).L.astype(np.int64)


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``logadd_pos`` on int log-code planes (``LogPosPlane.add``)."""

    left = np.asarray(a, dtype=np.int64)
    right = np.asarray(b, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("positive-log add operands must have matching shapes")
    return planes.LogPosPlane(left).add(planes.LogPosPlane(right)).L.astype(np.int64)


def complement(a: np.ndarray) -> np.ndarray:
    """``1 - x`` (``log1m_pos``) on an int log-code plane of values in ``[0, 1]``."""

    return planes.LogPosPlane(np.asarray(a, dtype=np.int64)).log1m().L.astype(np.int64)


@lru_cache(maxsize=1)
def _transmittance_code_table() -> np.ndarray:
    """``L_T[L_tau] = clamp(rne(-256 * 2**(L_tau/256) / ln 2))`` for every int16 code.

    The optical depth ``tau`` is a positive log value; its transmittance
    ``exp(-tau)`` has log code ``-256 * tau / ln 2``.  Indexed by the unsigned
    16-bit code of ``tau``; the zero sentinel maps to unit transmittance and
    the NaN code to the NaN code.
    """

    magnitude = _magnitude_table()
    codes = np.rint(-float(v02.SPF32_L_SCALE) * magnitude / math.log(2.0))
    codes = np.where(np.isfinite(codes), codes, float(v02.SPF32_L_MIN_FINITE))
    codes = np.clip(codes, v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE).astype(np.int64)
    codes[v02.SPF32_L_ZERO & 0xFFFF] = 0
    codes[v02.SPF32_NAN_L & 0xFFFF] = v02.SPF32_NAN_L
    codes.setflags(write=False)
    return codes


def optical_depth_to_transmittance(tau: planes.LogPosPlane) -> planes.LogPosPlane:
    """Map an optical-depth LOG_POS plane to its transmittance LOG_POS plane (ROM lookup)."""

    return planes.LogPosPlane(_transmittance_code_table()[tau.L.astype(np.int64) & 0xFFFF])


def compose_front_to_back(
    transmittance: planes.LogPosPlane,
    contribution: planes.LogSignedPlane | planes.LogPosPlane,
) -> tuple[planes.LogPosPlane, planes.LogSignedPlane]:
    """Ordered affine composition ``(T, C)`` over a leading segment axis.

    ``transmittance`` has shape ``(n, ...)`` and ``contribution`` ``(n, ...)``
    (channels may trail).  Returns the composed ``T = prod t_i`` and
    ``C = sum_i (prod_{j<i} t_j) c_i`` as planes, using only
    ``LogPosPlane.mul``/``LogSignedPlane.add``: the same operations
    :func:`spf.tensor.v02.transport_compose` applies segment by segment.
    """

    n = transmittance.shape[0]
    contribution_signed = contribution.as_signed()
    total_t = planes.LogPosPlane.ones(transmittance.shape[1:])
    total_c = planes.LogSignedPlane.zeros(contribution_signed.shape[1:])
    for index in range(n):
        step_t = transmittance[index]
        step_c = contribution_signed[index]
        prefix = total_t
        if step_c.L.ndim > prefix.L.ndim:
            prefix = planes.LogPosPlane(prefix.L[..., None])
        total_c = total_c.add(step_c.mul(prefix))
        total_t = total_t.mul(step_t)
    return total_t, total_c


def display_values(plane: planes.LogPosPlane | planes.LogSignedPlane) -> np.ndarray:
    """Float64 display projection of a plane through the magnitude ROM (no decoder call)."""

    magnitude = np.asarray(_magnitude_table()[plane.L.astype(np.int64) & 0xFFFF], dtype=np.float64)
    if isinstance(plane, planes.LogSignedPlane):
        return np.asarray(magnitude * plane.sign.astype(np.float64), dtype=np.float64)
    return magnitude


def _product_error(bounds: np.ndarray) -> float:
    """Bound a product of factors whose absolute errors are ``bounds``."""

    values = np.asarray(bounds, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    if np.any(np.isnan(values)) or np.any(values < 0.0):
        raise ValueError("transport error bounds must be finite or positive infinity")
    if np.any(np.isinf(values)):
        return math.inf
    logarithmic = math.fsum(math.log1p(float(value)) for value in values)
    if logarithmic >= math.log(np.finfo(np.float64).max):
        return math.inf
    return float(math.expm1(logarithmic))


def _safe_product(left: Any, right: Any) -> np.ndarray:
    """Multiply non-negative envelopes without forming ``0 * inf``."""

    shape = np.broadcast_shapes(np.shape(left), np.shape(right))
    lhs = np.broadcast_to(np.asarray(left, dtype=np.float64), shape)
    rhs = np.broadcast_to(np.asarray(right, dtype=np.float64), shape)
    result = np.zeros(shape, dtype=np.float64)
    np.multiply(lhs, rhs, out=result, where=(lhs != 0.0) & (rhs != 0.0))
    return result


def reduce_guarded_optical_depth(
    tau: np.ndarray,
    c: np.ndarray,
    et: np.ndarray,
    ec: np.ndarray,
    stages: tuple[Any, ...] = (),
    ledgers: tuple[v02.ErrorLedger, ...] = (),
) -> Any:
    """Reduce certified optical-depth transports at one native terminal.

    ``tau`` is retained as the attenuation carrier until the terminal.  A
    pure-extinction ray uses one ``fsum`` and one exponential per distinct
    channel total.  Emissive rays use ordered prefix attenuation and a single
    vectorized contribution reduction, with absolute input/error envelopes
    propagated through the affine law.
    """

    from dataclasses import replace

    from spf.renderer.volume import TransportRGB

    if ledgers and len(ledgers) != int(np.asarray(tau).shape[0]):
        raise ValueError("guarded optical-depth ledgers must match the segment count")
    base_ledger = v02.ErrorLedger()
    for item in ledgers:
        if not isinstance(item, v02.ErrorLedger):
            raise TypeError("guarded optical-depth ledgers must be ErrorLedger values")
        base_ledger = base_ledger.plus(item)

    def reduction_ledger(total_error: float) -> v02.ErrorLedger:
        # Segment ledgers already carry source/representation/transform/etc.
        # categories.  Only add the unclassified gap required by the output
        # transport bound, avoiding a second copy of those categories.
        base_total = base_ledger.total_bound
        if math.isnan(base_total) or math.isnan(total_error):
            gap = math.nan
        elif math.isinf(base_total) or base_total >= total_error:
            gap = 0.0
        else:
            gap = total_error - base_total
        stage = v02.ErrorLedger(reducer=gap)
        return replace(
            base_ledger,
            reducer=base_ledger.reducer + gap,
            stages=(*base_ledger.stages, stage),
        )

    tau = np.asarray(tau, dtype=np.float64)
    # The contribution keeps the carrier the caller handed over: a real emission
    # spectrum stays a real plane through the whole terminal instead of being
    # promoted to a Cartesian pair plane.  ``np.isfinite`` covers both carriers.
    c = np.asarray(c)
    if c.dtype.kind not in "fc":
        c = np.asarray(c, dtype=np.float64)
    if tau.ndim != 2 or c.ndim != 2 or tau.shape != c.shape:
        raise ValueError("guarded optical-depth arrays must be two-dimensional and shape-matched")
    if tau.shape[1] <= 0:
        raise ValueError("guarded optical-depth arrays need at least one channel")
    if np.any(np.isnan(tau)) or np.any(tau < 0.0):
        raise ValueError("guarded optical depth must be non-negative and non-NaN")
    if np.any(~np.isfinite(c)):
        raise ValueError("guarded optical-depth contributions must be finite")
    count, channels = tau.shape
    et = np.asarray(et, dtype=np.float64).reshape(-1)
    ec = np.asarray(ec, dtype=np.float64).reshape(-1)
    if et.size == 1 and count > 1:
        et = np.repeat(et, count)
    if ec.size == 1 and count > 1:
        ec = np.repeat(ec, count)
    if et.shape != (count,) or ec.shape != (count,):
        raise ValueError("guarded optical-depth error bounds must match the segment count")
    if np.any(np.isnan(et)) or np.any(et < 0.0) or np.any(np.isnan(ec)) or np.any(ec < 0.0):
        raise ValueError("guarded optical-depth error bounds must be non-negative")
    if count == 0:
        return TransportRGB(
            (1.0,) * channels,
            (0.0,) * channels,
            ledger=v02.ErrorLedger(stages=stages),
            optical_depth=(0.0,) * channels,
        )

    eps = np.finfo(np.float64).eps
    gamma_denominator = 1.0 - count * eps
    gamma = math.inf if gamma_denominator <= 0.0 else count * eps / gamma_denominator
    pure_extinction = bool(np.all(c == 0.0))
    tau_totals = np.asarray(
        [math.fsum(float(value) for value in tau[:, channel]) for channel in range(channels)],
        dtype=np.float64,
    )
    if pure_extinction:
        # RGB scalar τ is broadcast frequently.  Share its one terminal
        # exponential instead of repeating identical libm calls per channel.
        if np.all(tau_totals == tau_totals[0]):
            terminal = 0.0 if math.isinf(float(tau_totals[0])) else math.exp(-float(tau_totals[0]))
            exact_t = np.full(channels, terminal, dtype=np.float64)
        else:
            exact_t = np.asarray(
                [0.0 if math.isinf(float(value)) else math.exp(-float(value)) for value in tau_totals],
                dtype=np.float64,
            )
        tau_round = 0.0 if np.any(np.isinf(tau_totals)) else gamma * max(
            1.0, float(np.max(tau_totals, initial=0.0))
        )
        error_t = _product_error(et) + tau_round
        error_c = math.fsum(float(value) for value in ec) + gamma * max(1, count)
        ledger = reduction_ledger(error_t + error_c)
        if stages and not ledgers:
            ledger = replace(ledger, stages=tuple(stages) + ledger.stages)
        return TransportRGB(
            tuple(float(value) for value in exact_t),
            (0.0,) * channels,
            error_t,
            error_c,
            ledger,
            optical_depth=tuple(float(value) for value in tau_totals),
        )

    # Prefix τ is ordered front-to-back.  Numpy evaluates all exponentials at
    # this one terminal rather than invoking the scalar pair compositor at
    # every tree node.
    prefix_tau = np.concatenate((np.zeros((1, channels), dtype=np.float64), tau[:-1]), axis=0)
    prefix_tau = np.cumsum(prefix_tau, axis=0, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        prefix_t = np.exp(-prefix_tau)
    # Same per-channel terminal exponential as the pure-extinction branch above,
    # so both branches read the transmittance off one scalar libm call each.
    exact_t = np.asarray(
        [0.0 if math.isinf(float(value)) else math.exp(-float(value)) for value in tau_totals],
        dtype=np.float64,
    )
    prefix_t = np.where(np.isfinite(prefix_t), prefix_t, 0.0)
    exact_t = np.where(np.isfinite(exact_t), exact_t, 0.0)
    terms = prefix_t * c
    exact_c = np.add.reduce(terms, axis=0)

    # Product envelopes bound every prefix's attenuation error.  Include the
    # τ summation/cumulative-rounding envelope in both attenuation and the
    # attenuated emission term; absolute magnitudes remain valid under
    # cancellation in C.
    with np.errstate(over="ignore", invalid="ignore"):
        # Per-segment scalar bounds, the same spelling ``_product_error`` uses:
        # the envelope is one value per segment, never a plane, so it needs no
        # vectorized transcendental.  Tiny positive bounds stay in log1p/expm1
        # space; adding them to 1.0 first would erase sub-epsilon certificates.
        log_bounds = np.asarray([math.log1p(float(value)) for value in et], dtype=np.float64)
        prefix_log = np.cumsum(log_bounds, dtype=np.float64)
        prefix_log_abs = np.cumsum(np.abs(log_bounds), dtype=np.float64)
        prefix_upper = prefix_log + gamma * prefix_log_abs
        prefix_product = np.asarray(
            [math.inf if float(value) >= 709.78 else math.expm1(float(value)) for value in prefix_upper],
            dtype=np.float64,
        )
    prefix_product = np.where(np.isfinite(prefix_product), prefix_product, math.inf)
    prefix_error = np.concatenate((np.zeros(1, dtype=np.float64), prefix_product[:-1]))
    finite_tau = tau_totals[np.isfinite(tau_totals)]
    tau_scale = float(np.max(finite_tau, initial=0.0)) if finite_tau.size else 1.0
    tau_round = gamma * max(1.0, tau_scale)
    error_t = _product_error(et) + tau_round
    term_error = _safe_product(np.abs(prefix_t), ec[:, None])
    term_error += _safe_product(np.abs(c), prefix_error[:, None])
    term_error += _safe_product(ec[:, None], prefix_error[:, None])
    term_error += _safe_product(np.abs(c), tau_round)
    error_c_by_channel = np.sum(term_error, axis=0)
    error_c_by_channel += gamma * np.sum(np.abs(terms), axis=0)
    error_c = float(np.max(error_c_by_channel, initial=0.0))
    error_t += np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(exact_t), initial=0.0)))
    ledger = reduction_ledger(error_t + error_c)
    if stages and not ledgers:
        ledger = replace(ledger, stages=tuple(stages) + ledger.stages)
    contribution: tuple[complex | float, ...] = tuple(
        value.real if abs(value.imag) <= 1.0e-15 else value for value in exact_c
    )
    return TransportRGB(
        tuple(float(value) for value in exact_t),
        contribution,
        error_t,
        error_c,
        ledger,
        optical_depth=tuple(float(value) for value in tau_totals),
    )


def reduce_logpos(t: np.ndarray, c: np.ndarray, et: np.ndarray, ec: np.ndarray, stages: tuple[Any, ...] = ()) -> Any:
    from spf.renderer.volume import TransportRGB

    magnitudes = _magnitude_table()

    t = np.asarray(t, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    if t.ndim != 2 or c.ndim != 2 or t.shape != c.shape:
        raise ValueError("positive-log transport arrays must be two-dimensional and shape-matched")
    if t.shape[1] <= 0:
        raise ValueError("positive-log transport arrays need at least one channel")
    valid_t = (t == v02.SPF32_L_ZERO) | ((t >= v02.SPF32_L_MIN_FINITE) & (t <= v02.SPF32_L_MAX_FINITE))
    valid_c = (c == v02.SPF32_L_ZERO) | ((c >= v02.SPF32_L_MIN_FINITE) & (c <= v02.SPF32_L_MAX_FINITE))
    if np.any(~valid_t):
        raise ValueError("positive-log transmittance contains an invalid log code")
    if np.any(~valid_c):
        raise ValueError("positive-log contribution contains an invalid log code")
    et = np.asarray(et, dtype=np.float64).reshape(-1)
    ec = np.asarray(ec, dtype=np.float64).reshape(-1)
    if et.size == 1 and t.shape[0] > 1:
        et = np.repeat(et, t.shape[0])
    if ec.size == 1 and t.shape[0] > 1:
        ec = np.repeat(ec, t.shape[0])
    if et.shape != (t.shape[0],) or ec.shape != (t.shape[0],):
        raise ValueError("positive-log transport error arrays must match the segment count")
    if np.any(np.isnan(et)) or np.any(et < 0.0) or np.any(np.isnan(ec)) or np.any(ec < 0.0):
        raise ValueError("positive-log transport error bounds must be non-negative")
    unbounded = bool(np.any(np.isinf(et)) or np.any(np.isinf(ec)))
    # Preserve the typed result while avoiding invalid 0*inf arithmetic in
    # intermediate node bounds.  The final certificate remains unbounded.
    if unbounded:
        et = np.where(np.isinf(et), 0.0, et)
        ec = np.where(np.isinf(ec), 0.0, ec)
    if t.shape[0] == 0:
        return TransportRGB(
            (v02.LogPos(0),) * t.shape[1],
            (v02.LogPos.zero(),) * t.shape[1],
            ledger=v02.ErrorLedger(stages=stages),
        )

    def values(codes: np.ndarray) -> np.ndarray:
        return np.asarray(magnitudes[codes & 65535])

    while len(t) > 1:
        pairs = len(t) // 2
        a, b = slice(0, 2 * pairs, 2), slice(1, 2 * pairs, 2)
        ta, tb, ca, cb = values(t[a]), values(t[b]), values(c[a]), values(c[b])
        nt, nc = multiply(t[a], t[b]), add(c[a], multiply(t[a], c[b]))
        own_t = np.max(np.abs(values(nt) - ta * tb), axis=1)
        own_c = np.max(np.abs(values(nc) - (ca + ta * cb)), axis=1)
        at, bt, bc = np.max(ta, axis=1), np.max(tb, axis=1), np.max(cb, axis=1)
        net = own_t + bt * et[a] + at * et[b] + et[a] * et[b]
        nec = own_c + ec[a] + bc * et[a] + at * ec[b] + et[a] * ec[b]
        if len(t) % 2:
            nt, nc = np.concatenate((nt, t[-1:])), np.concatenate((nc, c[-1:]))
            net, nec = np.r_[net, et[-1]], np.r_[nec, ec[-1]]
        t, c, et, ec = nt, nc, net, nec
    output_et = math.inf if unbounded else float(et[0])
    output_ec = math.inf if unbounded else float(ec[0])
    ledger = v02.ErrorLedger(reducer=output_et + output_ec, stages=stages)
    return TransportRGB(
        tuple(v02.LogPos(int(x)) for x in t[0]),
        tuple(v02.LogPos(int(x)) for x in c[0]),
        output_et,
        output_ec,
        ledger,
    )
