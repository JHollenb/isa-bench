"""Backend-aware cost model for SFR v0.2 planning (design §40).

§40.1 primitives are :class:`PrimitiveCosts`; ``C_fft(N)`` and ``C_sort(N)`` are
the two size-dependent ones and both scale as ``N log2 N``.  §40.2's plan cost is
:meth:`CostModel.estimate_ir`, which carries the four documented λ terms plus
``λ_M M(P)`` and ``λ_D D(P)`` — see :class:`CostWeights` for why.  §40.3 backend
profiles are :func:`reference_cost_model`, :func:`cpu_cost_model`,
:func:`gpu_cost_model` (structural only; no CUDA on this host) and
:func:`spu_cost_model`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from spf.renderer.model import FieldRepresentation, RepresentationKind, SampleLayout
from spf.renderer.v02 import FieldIR


@dataclass(frozen=True, slots=True)
class CostWeights:
    """The λ multipliers of design §40.2, plus two documented extensions.

    §40.2 states ``C(P) = Σ C(o_i) + λ_B B(P) + λ_S S(P) + λ_R R(P) + λ_E E(P)``.
    This model carries two further terms, both reported separately so a plan
    total can be read back against the design's four:

    * ``λ_M M(P)`` (:attr:`materialization`) prices the elements a plan
      materializes.  The design will be amended to list it; it is not implied by
      §40.2 as written.
    * ``λ_D D(P)`` (:attr:`divergence`) prices data-dependent branching, which
      §40.3 names as a GPU penalty but §40.2 has no term for.  It is 0 for every
      backend except ``gpu``, so no other profile changes value because of it.
    """

    bytes_moved: float = 1.0 / 64.0
    synchronization: float = 8.0
    reductions: float = 4.0
    materialization: float = 1.0
    error: float = 1.0e3
    divergence: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"cost weight {name} must be finite and non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class PrimitiveCosts:
    phase_add: float = 1.0
    log_add: float = 1.0
    logadd_rom: float = 3.0
    pair_complex: float = 12.0
    pivot_setup: float = 8.0
    pivot_term: float = 2.0
    pivot_finalize: float = 14.0
    jet_setup: float = 20.0
    jet_sample: float = 1.0
    atom_transform: float = 80.0
    atom_sample: float = 24.0
    fft_coefficient: float = 5.0
    sort_item: float = 8.0
    transport_compose: float = 6.0
    dense_sample: float = 2.0
    memory_byte: float = 1.0 / 64.0
    materialize_element: float = 2.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"primitive cost {name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, base: PrimitiveCosts | None = None) -> PrimitiveCosts:
        """Build costs from ``{name: value}``; names absent from ``values`` keep ``base``."""

        current = base if base is not None else cls()
        known = {name: float(values[name]) for name in cls.__dataclass_fields__ if name in values}
        return replace(current, **known)

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class PlanCost:
    """One plan's cost terms.  ``memory`` and ``divergence`` follow ``total`` so
    that existing positional constructions keep their meaning."""

    compute: float = 0.0
    bytes_moved: float = 0.0
    synchronization: float = 0.0
    reductions: float = 0.0
    materialization: float = 0.0
    error: float = 0.0
    total: float = 0.0
    # C_memory(bytes) (§40.1), already included in ``compute`` and ``total``.
    memory: float = 0.0
    # D(P): divergent (data-dependent branch) operation count, priced by λ_D.
    divergence: float = 0.0


_DEFAULT_OP_COSTS: dict[str, float] = {
    "v02_phase_add": 1.0,
    "v02_phase_sub": 1.0,
    "v02_phase_conj": 1.0,
    "v02_phase_compose": 1.0,
    "v02_phase_shift": 1.0,
    "v02_log_mul": 1.0,
    "v02_log_div": 1.0,
    "v02_complex_mul": 2.0,
    "v02_complex_div": 2.0,
    "v02_logadd_pos": 3.0,
    "v02_logsub_pos": 3.0,
    "v02_logadd_signed": 4.0,
    "v02_complex_pair_add": 12.0,
    "v02_reduce_positive": 18.0,
    "v02_reduce_signed": 22.0,
    "v02_reduce_complex": 32.0,
    "v02_reduce_transport": 18.0,
    "v02_transport_compose": 6.0,
    "v02_jet_materialize": 8.0,
    "v02_mixture_materialize": 24.0,
    "v02_materialize": 8.0,
    "v02_materialize_tile": 8.0,
    "v02_fft": 64.0,
    "v02_ifft": 64.0,
    "v02_fft_forward": 64.0,
    "v02_fft_inverse": 64.0,
    "v02_sort": 16.0,
}


_REDUCTION_OPS = {
    "v02_logadd_pos",
    "v02_logsub_pos",
    "v02_logadd_signed",
    "v02_complex_pair_add",
    "v02_reduce_positive",
    "v02_reduce_signed",
    "v02_reduce_complex",
    "v02_reduce_optical_depth",
    "v02_reduce_transport",
    "v02_reduce_tropical",
    "v02_transport_compose",
    "v02_transport_reduce",
}


_MATERIALIZATION_OPS = {
    "v02_jet_materialize",
    "v02_mixture_materialize",
    "v02_mixture_lower",
    "v02_separable_materialize",
    "v02_materialize",
    "v02_materialize_tile",
    "v02_display",
    "v02_display_convert",
}


# ``C_jet_setup`` (§40.1) is charged once per tile: one jet-producing operation
# is one tile setup, and the per-element work stays in ``C_jet_sample``.
_JET_SETUP_OPS = {
    "v02_jet2d",
    "v02_jet_add",
    "v02_jet_transform",
    "v02_jet_materialize",
    "v02_separable_materialize",
    "v02_atom_to_jet",
    "v02_mixture_to_jets",
}


# ``C_atom_transform`` (§40.1) is charged once per atom operation: the analytic
# product/affine/Fourier transforms of Ref §15, plus lowering a mixture.
_ATOM_TRANSFORM_OPS = {
    "v02_atom_affine",
    "v02_atom_convolve",
    "v02_atom_create",
    "v02_atom_derivative",
    "v02_atom_div",
    "v02_atom_fourier",
    "v02_atom_hessian",
    "v02_atom_integral",
    "v02_atom_inverse_fourier",
    "v02_atom_mul",
    "v02_atom_pixel_integral",
    "v02_atom_product",
    "v02_mixture_lower",
    "v02_mixture_materialize",
    "v02_spherical_gaussian_product",
}


# D(P) of §40.3: operations whose work is data dependent (sentinel/near-
# cancellation branches, pivot selection, ordering).  λ_D prices them; only the
# GPU profile sets λ_D > 0.
_DIVERGENT_OPS = {
    "v02_complex_pair_add",
    "v02_complex_pair_select",
    "v02_logadd_signed",
    "v02_mixture_cull",
    "v02_mixture_prune",
    "v02_reduce_complex",
    "v02_reduce_signed",
    "v02_reduce_tropical",
    "v02_sort",
    "v02_tile_cull",
    "v02_visibility_select",
}


_SORT_OPS = {"v02_sort"}


# --------------------------------------------------------------------------- #
# Measured calibration (blog Tier 2 item 8).  ``scripts/sfr_calibrate_costs.py``
# times the primitives above on this host and writes ``ci/sfr_cost_calibration.json``.
# The static constants remain the documented fallback; a model built from the
# file reports ``calibrated=True`` and keeps the measurement beside the estimate.
# --------------------------------------------------------------------------- #

CALIBRATION_PATH = Path(__file__).resolve().parents[3] / "ci" / "sfr_cost_calibration.json"
SIZE_CLASSES: tuple[int, ...] = (256, 4096, 65536, 1048576)
STATIC_SOURCE = "static_defaults"


def _op_costs_from_primitives(p: PrimitiveCosts) -> dict[str, float]:
    """Per-operation unit costs derived from measured primitives (same keys as the static table)."""

    return {
        "v02_phase_add": p.phase_add,
        "v02_phase_sub": p.phase_add,
        "v02_phase_conj": p.phase_add,
        "v02_phase_compose": p.phase_add,
        "v02_phase_shift": p.phase_add,
        "v02_log_mul": p.log_add,
        "v02_log_div": p.log_add,
        "v02_complex_mul": p.log_add + p.phase_add,
        "v02_complex_div": p.log_add + p.phase_add,
        "v02_logadd_pos": p.logadd_rom,
        "v02_logsub_pos": p.logadd_rom,
        "v02_logadd_signed": p.logadd_rom + p.phase_add,
        "v02_complex_pair_add": p.pair_complex,
        "v02_reduce_positive": p.pivot_setup + p.pivot_term + p.pivot_finalize,
        "v02_reduce_signed": p.pivot_setup + 2.0 * p.pivot_term + p.pivot_finalize,
        "v02_reduce_complex": p.pivot_setup + p.pair_complex + p.pivot_finalize,
        "v02_reduce_transport": p.transport_compose + p.pivot_setup,
        "v02_transport_compose": p.transport_compose,
        "v02_jet_materialize": p.jet_sample,
        "v02_mixture_materialize": p.atom_sample,
        "v02_materialize": p.materialize_element,
        "v02_materialize_tile": p.materialize_element,
        "v02_fft": p.fft_coefficient,
        "v02_ifft": p.fft_coefficient,
        "v02_fft_forward": p.fft_coefficient,
        "v02_fft_inverse": p.fft_coefficient,
        "v02_sort": p.sort_item,
    }


_KIND_PRIMITIVE: dict[RepresentationKind, str] = {
    RepresentationKind.GAUSSIAN_CHIRP_ATOM: "atom_sample",
    RepresentationKind.SPHERICAL_GAUSSIAN_ATOM: "atom_sample",
    RepresentationKind.ATOM_MIXTURE: "atom_sample",
    RepresentationKind.QUADRATIC_JET: "jet_sample",
    RepresentationKind.SEPARABLE: "jet_sample",
    RepresentationKind.SPARSE_RESIDUAL: "dense_sample",
    RepresentationKind.HERMITIAN_HALF: "fft_coefficient",
    RepresentationKind.DENSE_TYPED: "dense_sample",
    RepresentationKind.DENSE_CARTESIAN_REFERENCE: "materialize_element",
}


@dataclass(frozen=True, slots=True)
class CostCalibration:
    """Contents of ``ci/sfr_cost_calibration.json``: median ns per primitive unit per backend and size class."""

    source: str
    host: str
    date: str
    repeats: int
    warmups: int
    backends: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "backends", MappingProxyType({str(k): v for k, v in dict(self.backends).items()}))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source: str) -> CostCalibration:
        return cls(
            source=str(source),
            host=str(data.get("host", "")),
            date=str(data.get("date", "")),
            repeats=int(data.get("repeats", 0)),
            warmups=int(data.get("warmups", 0)),
            backends=dict(data.get("backends", {})),
        )

    def backend(self, name: str) -> Mapping[str, Any] | None:
        return self.backends.get(str(name).lower().replace("-", "_"))

    def unit_ns(self, name: str) -> float:
        entry = self.backend(name)
        if not entry:
            return 0.0
        return float(entry.get("unit_ns", 0.0) or 0.0)

    def primitives(self, name: str, *, size: int | None = None, base: PrimitiveCosts | None = None) -> PrimitiveCosts:
        """Primitive costs in ``phase_add`` units; ``size`` selects the nearest measured size class."""

        entry = self.backend(name)
        if not entry:
            return base if base is not None else PrimitiveCosts()
        unit = self.unit_ns(name)
        table: Mapping[str, Any] = entry.get("primitives_ns", {})
        if size is not None and entry.get("sizes_ns"):
            classes = sorted(int(k) for k in entry["sizes_ns"])
            nearest = min(classes, key=lambda c: abs(math.log2(max(c, 1)) - math.log2(max(int(size), 1))))
            table = {**table, **entry["sizes_ns"][str(nearest)]}
        if unit <= 0.0:
            # Cycle-model backends (spu_sim) store unit costs directly.
            return PrimitiveCosts.from_mapping(table, base=base)
        return PrimitiveCosts.from_mapping({k: float(v) / unit for k, v in table.items()}, base=base)

    def measured_names(self, name: str) -> tuple[str, ...]:
        entry = self.backend(name)
        if not entry:
            return ()
        return tuple(str(v) for v in entry.get("measured", ()))


def load_cost_calibration(path: str | Path | None = None) -> CostCalibration | None:
    """Load the committed calibration file; ``None`` when absent or unreadable."""

    location = Path(path) if path is not None else CALIBRATION_PATH
    try:
        data = json.loads(location.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, Mapping) or "backends" not in data:
        return None
    return CostCalibration.from_dict(data, source=str(location))


@dataclass(frozen=True, slots=True)
class CostModel:
    name: str
    primitives: PrimitiveCosts = field(default_factory=PrimitiveCosts)
    weights: CostWeights = field(default_factory=CostWeights)
    operation_overrides: Mapping[str, float] = field(default_factory=dict)
    representation_multipliers: Mapping[RepresentationKind, float] = field(default_factory=dict)
    calibration: CostCalibration | None = None
    calibrated: bool = False
    ns_per_unit: float = 0.0
    # A device this profile models but whose timings this host cannot measure.
    # Set for ``gpu``; ``describe()`` then reports the profile as structural.
    device: str | None = None

    def __post_init__(self) -> None:
        overrides = {str(key): float(value) for key, value in self.operation_overrides.items()}
        multipliers = {
            key if isinstance(key, RepresentationKind) else RepresentationKind(str(key).upper()): float(value)
            for key, value in self.representation_multipliers.items()
        }
        if any(value < 0.0 or not math.isfinite(value) for value in (*overrides.values(), *multipliers.values())):
            raise ValueError("cost overrides must be finite and non-negative")
        object.__setattr__(self, "name", str(self.name).lower().replace("-", "_"))
        object.__setattr__(self, "representation_multipliers", MappingProxyType(multipliers))
        calibration = self.calibration
        calibrated = bool(self.calibrated) and calibration is not None and calibration.backend(self.name) is not None
        object.__setattr__(self, "calibrated", calibrated)
        if calibrated:
            assert calibration is not None
            object.__setattr__(self, "primitives", calibration.primitives(self.name, base=self.primitives))
            object.__setattr__(self, "ns_per_unit", float(calibration.unit_ns(self.name)))
            derived = _op_costs_from_primitives(self.primitives)
            overrides = {**derived, **overrides}
        else:
            object.__setattr__(self, "ns_per_unit", float(self.ns_per_unit))
        object.__setattr__(self, "operation_overrides", MappingProxyType(overrides))

    # ---- calibration surface ------------------------------------------------

    @property
    def calibration_source(self) -> str:
        return self.calibration.source if self.calibrated and self.calibration is not None else STATIC_SOURCE

    @property
    def uncalibrated_reason(self) -> str | None:
        """Why this profile is structural, when it models a device we cannot time."""

        if self.calibrated or self.device is None:
            return None
        return (
            f"no '{self.name}' entry in {CALIBRATION_PATH.name}: no {self.device} device on this host, "
            f"so the {self.name} numbers are the design's §40.3 penalty structure, "
            "not measured timings and not the cpu measurements"
        )

    def describe(self) -> dict[str, Any]:
        """Report-ready provenance: which numbers priced this plan."""

        info: dict[str, Any] = {
            "cost_model": self.name,
            "calibrated": self.calibrated,
            "calibration_source": self.calibration_source,
            "cost_unit": "phase_add" if self.name != "spu_sim" else "spu_cycle",
            "ns_per_unit": self.ns_per_unit,
        }
        if self.calibrated and self.calibration is not None:
            info["calibration_host"] = self.calibration.host
            info["calibration_date"] = self.calibration.date
            info["calibration_repeats"] = self.calibration.repeats
            info["measured_primitives"] = self.calibration.measured_names(self.name)
        if self.device is not None:
            info["device"] = self.device
            info["device_calibrated"] = self.calibrated
        reason = self.uncalibrated_reason
        if reason is not None:
            info["uncalibrated_on_host"] = True
            info["uncalibrated_reason"] = reason
        return info

    def primitives_for(self, element_count: int) -> PrimitiveCosts:
        """Region-aware primitive costs: the measured size class nearest ``element_count``."""

        if not self.calibrated or self.calibration is None:
            return self.primitives
        return self.calibration.primitives(self.name, size=max(1, int(element_count)), base=self.primitives)

    def estimate_ms(self, cost: PlanCost | float) -> float | None:
        """Convert a unit cost to milliseconds when the model is calibrated; ``None`` otherwise."""

        if self.ns_per_unit <= 0.0:
            return None
        total = cost.total if isinstance(cost, PlanCost) else float(cost)
        if not math.isfinite(total):
            return None
        return total * self.ns_per_unit / 1.0e6

    def cost_estimate(self, cost: PlanCost | float, **extra: Any) -> dict[str, Any]:
        """The ``cost_estimate`` mapping carried by reports: unit total, provenance, optional ms."""

        total = cost.total if isinstance(cost, PlanCost) else float(cost)
        record: dict[str, Any] = {"total": float(total), **self.describe(), **extra}
        if isinstance(cost, PlanCost):
            record.update(
                compute=cost.compute,
                bytes_moved=cost.bytes_moved,
                synchronization=cost.synchronization,
                reductions=cost.reductions,
                materialization=cost.materialization,
                error=cost.error,
                memory=cost.memory,
                divergence=cost.divergence,
            )
        record["estimated_ms"] = self.estimate_ms(total)
        return record

    def operation_cost(self, operation: str, *, element_count: int = 1) -> float:
        """Unit cost of one operation on ``element_count`` elements (§40.1).

        ``C_fft(N)`` and ``C_sort(N)`` both scale as ``N log2 N`` from their
        calibrated per-item base.  ``C_jet_setup`` is added once per jet tile and
        ``C_atom_transform`` once per atom operation; the per-element remainder
        is ``C_jet_sample`` / ``C_atom_sample``.
        """

        name = str(operation)
        base = self.operation_overrides.get(name, _DEFAULT_OP_COSTS.get(name, 2.0))
        count = max(1, int(element_count))
        # Setup and per-element base both come from ``self.primitives`` (the
        # operation overrides are derived from it), so one operation is priced in
        # one size class; ``primitives_for`` is the region-aware path used by
        # :meth:`estimate_representation`.
        setup = 0.0
        if name in _JET_SETUP_OPS:
            setup += float(self.primitives.jet_setup)
        if name in _ATOM_TRANSFORM_OPS:
            setup += float(self.primitives.atom_transform)
        if name in {"v02_fft", "v02_ifft", "v02_fft_forward", "v02_fft_inverse"} or name in _SORT_OPS:
            return float(setup + base * count * max(math.log2(count), 1.0))
        if name in _MATERIALIZATION_OPS:
            return float(setup + base * count)
        return float(setup + base)

    def memory_cost(self, nbytes: float) -> float:
        """``C_memory(bytes)`` of §40.1: the measured per-byte movement cost.

        This is the primitive, priced beside the plan-level ``λ_B B(P)`` penalty
        rather than instead of it: ``λ_B`` is the backend's exchange rate for
        traffic (0 for the reference profile, which ignores it), while
        ``C_memory`` is the calibrated cost of the bytes themselves.
        """

        count = float(nbytes)
        if count <= 0.0 or not math.isfinite(count):
            return 0.0
        return float(self.primitives.memory_byte * count)

    def estimate_ir(
        self,
        ir: FieldIR,
        *,
        element_count: int = 1,
        bytes_moved: int = 0,
        error: float = 0.0,
    ) -> PlanCost:
        """Price one IR graph with the §40.2 plan cost.

        ``Σ_i C(o_i)`` comes from :meth:`operation_cost` and includes
        ``C_memory(bytes)`` for the traffic passed in ``bytes_moved``.  The
        weighted terms are ``λ_B B(P) + λ_S S(P) + λ_R R(P) + λ_E E(P)`` of
        §40.2 plus the two extensions documented on :class:`CostWeights`:
        ``λ_M M(P)`` (materialized elements) and ``λ_D D(P)`` (divergent
        operations, GPU only).  ``λ_M`` is a fifth term the design text does not
        list; the design will be amended to carry it.
        """

        compute = 0.0
        reductions = 0.0
        materialization = 0.0
        synchronization = 0.0
        divergence = 0.0
        for item in ir.nodes:
            cost = self.operation_cost(item.op, element_count=element_count)
            compute += cost
            if item.op in _REDUCTION_OPS:
                reductions += 1.0
                synchronization += max(math.log2(max(element_count, 1)), 1.0)
            if item.op in _MATERIALIZATION_OPS:
                materialization += float(element_count)
            if item.op in _DIVERGENT_OPS:
                divergence += 1.0
        memory = self.memory_cost(bytes_moved)
        compute += memory
        weights = self.weights
        total = (
            compute
            + weights.bytes_moved * float(bytes_moved)
            + weights.synchronization * synchronization
            + weights.reductions * reductions
            + weights.materialization * materialization
            + weights.error * float(error)
            + weights.divergence * divergence
        )
        return PlanCost(
            compute=compute,
            bytes_moved=float(bytes_moved),
            synchronization=synchronization,
            reductions=reductions,
            materialization=materialization,
            error=float(error),
            total=total,
            memory=memory,
            divergence=divergence,
        )

    def estimate_representation(
        self,
        representation: FieldRepresentation,
        layout: SampleLayout,
    ) -> PlanCost:
        """Price sampling one representation over one region.

        This is the *evaluation* side: ``C_jet_sample`` / ``C_atom_sample`` /
        ``C_dense_sample`` per element (region-aware through
        :meth:`primitives_for`), the caller's own ``transform_cost``,
        ``C_memory(bytes)`` for the stored bytes beside the ``λ_B B(P)`` traffic
        penalty, and ``λ_M M(P)`` for materialized elements (the documented fifth
        term).  The setup primitives ``C_jet_setup`` and ``C_atom_transform`` are
        charged on the operations that perform the setup — see
        :meth:`operation_cost` — and not again here, so representation ranking
        stays a comparison of per-sample work.
        """

        count = layout.element_count
        multiplier = self.representation_multipliers.get(cast(RepresentationKind, representation.kind), 1.0)
        unit = 1.0
        kind = cast(RepresentationKind, representation.kind)
        if self.calibrated:
            primitives = self.primitives_for(count)
            unit = float(getattr(primitives, _KIND_PRIMITIVE.get(kind, "dense_sample")))
            if kind is RepresentationKind.HERMITIAN_HALF:
                unit *= max(math.log2(max(count, 2)), 1.0)
        compute = multiplier * (
            representation.evaluation_cost * unit * count
            + representation.transform_cost
            + representation.materialization_cost * count
        )
        bytes_moved = float(representation.byte_size or 0)
        memory = self.memory_cost(bytes_moved)
        compute += memory
        materialization = (
            0.0
            if representation.kind
            not in {
                RepresentationKind.DENSE_TYPED,
                RepresentationKind.DENSE_CARTESIAN_REFERENCE,
            }
            else float(count)
        )
        error = float(representation.error.total_bound)
        if not math.isfinite(error):
            return PlanCost(error=error, total=math.inf)
        total = (
            compute
            + self.weights.bytes_moved * bytes_moved
            + self.weights.materialization * materialization
            + self.weights.error * error
        )
        return PlanCost(
            compute=compute,
            bytes_moved=bytes_moved,
            materialization=materialization,
            error=error,
            total=total,
            memory=memory,
        )


def _calibration_for(calibrate: bool | CostCalibration | None) -> CostCalibration | None:
    if calibrate is False:
        return None
    if isinstance(calibrate, CostCalibration):
        return calibrate
    return load_cost_calibration()


def reference_cost_model(calibrate: bool | CostCalibration | None = True) -> CostModel:
    calibration = _calibration_for(calibrate)
    return CostModel(
        "reference",
        weights=CostWeights(bytes_moved=0.0, synchronization=0.0, reductions=0.0, materialization=0.0, error=1.0e6),
        calibration=calibration,
        calibrated=calibration is not None,
    )


def cpu_cost_model(calibrate: bool | CostCalibration | None = True) -> CostModel:
    calibration = _calibration_for(calibrate)
    return CostModel(
        "cpu",
        primitives=PrimitiveCosts(atom_sample=18.0, jet_sample=0.6, memory_byte=1.0 / 48.0),
        weights=CostWeights(
            bytes_moved=1.0 / 48.0,
            synchronization=5.0,
            reductions=3.0,
            materialization=0.8,
            error=2.0e3,
        ),
        representation_multipliers={
            RepresentationKind.GAUSSIAN_CHIRP_ATOM: 0.8,
            RepresentationKind.QUADRATIC_JET: 0.45,
            RepresentationKind.DENSE_CARTESIAN_REFERENCE: 1.4,
        },
        calibration=calibration,
        calibrated=calibration is not None,
    )


def gpu_cost_model(calibrate: bool | CostCalibration | None = True) -> CostModel:
    """Design §40.3 GPU profile: **structural penalties, uncalibrated on this host.**

    §40.3 states, for a GPU, a high penalty for global passes and for divergence
    and a lower penalty for dense arithmetic.  That is spelled here as a large
    ``λ_S`` (a global pass is a synchronization point), a nonzero ``λ_D``
    (:class:`CostWeights` divergence, the only profile that sets it), cheap
    ``dense_sample``/``materialize_element`` primitives and a dense-Cartesian
    representation multiplier below 1, and an expensive
    ``pivot_setup``/``pair_complex``/``sort_item`` path.

    There is no CUDA device on this host, so ``ci/sfr_cost_calibration.json``
    carries no ``gpu`` backend: the model reports ``calibrated=False`` with
    :attr:`CostModel.uncalibrated_reason` rather than silently reusing the cpu
    measurements.  ``estimate_ms`` returns ``None`` for the same reason.
    """

    calibration = _calibration_for(calibrate)
    return CostModel(
        "gpu",
        primitives=PrimitiveCosts(
            # Lower penalty for dense arithmetic: wide lanes amortize the word ops.
            phase_add=0.25,
            log_add=0.25,
            logadd_rom=1.5,
            # High penalty for divergence and serial finalizers.
            pair_complex=24.0,
            pivot_setup=32.0,
            pivot_term=1.0,
            pivot_finalize=48.0,
            jet_setup=40.0,
            jet_sample=0.25,
            atom_transform=120.0,
            atom_sample=6.0,
            fft_coefficient=2.0,
            sort_item=48.0,
            transport_compose=3.0,
            dense_sample=0.25,
            memory_byte=1.0 / 16.0,
            materialize_element=0.5,
        ),
        weights=CostWeights(
            # Global passes dominate: bandwidth and every synchronization point.
            bytes_moved=1.0 / 16.0,
            synchronization=64.0,
            reductions=16.0,
            materialization=0.5,
            error=2.0e3,
            divergence=32.0,
        ),
        representation_multipliers={
            RepresentationKind.DENSE_TYPED: 0.6,
            RepresentationKind.DENSE_CARTESIAN_REFERENCE: 0.8,
            RepresentationKind.ATOM_MIXTURE: 1.5,
            RepresentationKind.QUADRATIC_JET: 0.6,
        },
        calibration=calibration,
        calibrated=calibration is not None,
        device="cuda",
    )


def spu_cost_model(calibrate: bool | CostCalibration | None = True) -> CostModel:
    calibration = _calibration_for(calibrate)
    return CostModel(
        "spu_sim",
        primitives=PrimitiveCosts(
            phase_add=0.1,
            log_add=0.1,
            logadd_rom=0.5,
            pair_complex=6.0,
            pivot_setup=2.0,
            pivot_term=0.5,
            pivot_finalize=5.0,
            jet_setup=3.0,
            jet_sample=0.08,
            atom_transform=24.0,
            atom_sample=8.0,
            fft_coefficient=4.0,
            sort_item=12.0,
            transport_compose=1.0,
            dense_sample=1.0,
            memory_byte=1.0 / 128.0,
        ),
        weights=CostWeights(
            bytes_moved=1.0 / 128.0,
            synchronization=2.0,
            reductions=1.0,
            materialization=2.0,
            error=2.0e3,
        ),
        representation_multipliers={
            RepresentationKind.GAUSSIAN_CHIRP_ATOM: 0.4,
            RepresentationKind.QUADRATIC_JET: 0.12,
            RepresentationKind.ATOM_MIXTURE: 0.5,
            RepresentationKind.DENSE_CARTESIAN_REFERENCE: 4.0,
        },
        calibration=calibration,
        calibrated=calibration is not None,
    )


__all__ = [
    "CALIBRATION_PATH",
    "SIZE_CLASSES",
    "STATIC_SOURCE",
    "CostCalibration",
    "CostModel",
    "CostWeights",
    "PlanCost",
    "PrimitiveCosts",
    "cpu_cost_model",
    "gpu_cost_model",
    "load_cost_calibration",
    "reference_cost_model",
    "spu_cost_model",
]
