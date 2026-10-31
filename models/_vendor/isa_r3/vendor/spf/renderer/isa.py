"""Typed Spectral Graphics ISA v0.2 metadata and SSA program validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

from spf.renderer.model import Coherence, Domain
from spf.renderer.v02 import FieldKind


class ISAOpcode(StrEnum):
    LOG_MUL = "LOG_MUL"
    LOG_DIV = "LOG_DIV"
    LOGADD_POS = "LOGADD_POS"
    LOGSUB_POS = "LOGSUB_POS"
    LOGADD_SIGNED = "LOGADD_SIGNED"
    PHASE_ADD = "PHASE_ADD"
    PHASE_SUB = "PHASE_SUB"
    PHASE_CONJ = "PHASE_CONJ"
    SPF_MUL = "SPF_MUL"
    SPF_DIV = "SPF_DIV"
    SPF_CONJ = "SPF_CONJ"
    SPF_NEG = "SPF_NEG"
    SPF_ABS2 = "SPF_ABS2"
    SPF_PAIR_ADDSUB = "SPF_PAIR_ADDSUB"
    JET_INIT = "JET_INIT"
    JET_AFFINE = "JET_AFFINE"
    JET_ADD = "JET_ADD"
    JET_FD_ROW = "JET_FD_ROW"
    JET_FD_TILE = "JET_FD_TILE"
    JET_EMIT_LOG = "JET_EMIT_LOG"
    JET_EMIT_PHASE = "JET_EMIT_PHASE"
    JET_EMIT_SPF = "JET_EMIT_SPF"
    ATOM_MUL = "ATOM_MUL"
    ATOM_DIV = "ATOM_DIV"
    ATOM_AFFINE = "ATOM_AFFINE"
    ATOM_FOURIER = "ATOM_FOURIER"
    ATOM_INTEGRAL = "ATOM_INTEGRAL"
    ATOM_DERIVATIVE = "ATOM_DERIVATIVE"
    ATOM_BOUND = "ATOM_BOUND"
    ATOM_TO_JET = "ATOM_TO_JET"
    MIX_CONCAT = "MIX_CONCAT"
    MIX_CULL = "MIX_CULL"
    MIX_BOUND = "MIX_BOUND"
    MIX_FACTOR = "MIX_FACTOR"
    MIX_TO_JETS = "MIX_TO_JETS"
    MIX_MATERIALIZE = "MIX_MATERIALIZE"
    REDUCE_POS_PIVOT = "REDUCE_POS_PIVOT"
    REDUCE_SIGNED_PIVOT = "REDUCE_SIGNED_PIVOT"
    REDUCE_COMPLEX_PIVOT = "REDUCE_COMPLEX_PIVOT"
    REDUCE_TROPICAL_CERT = "REDUCE_TROPICAL_CERT"
    REDUCE_TRANSPORT_ORDERED = "REDUCE_TRANSPORT_ORDERED"
    FFT_FORWARD = "FFT_FORWARD"
    FFT_INVERSE = "FFT_INVERSE"
    HERMITIAN_SYNTH = "HERMITIAN_SYNTH"
    SPARSE_SYNTH = "SPARSE_SYNTH"
    SAMPLE_POINT = "SAMPLE_POINT"
    SAMPLE_FOOTPRINT = "SAMPLE_FOOTPRINT"
    MATERIALIZE_TILE = "MATERIALIZE_TILE"
    AABB_TEST = "AABB_TEST"
    BVH_STEP = "BVH_STEP"
    DEPTH_MIN = "DEPTH_MIN"
    MASK_AND = "MASK_AND"
    MASK_OR = "MASK_OR"
    TILE_BIN = "TILE_BIN"
    SORT_KEY = "SORT_KEY"
    VIS_SELECT = "VIS_SELECT"
    DISPLAY_CONVERT = "DISPLAY_CONVERT"


@dataclass(frozen=True, slots=True)
class EffectInfo:
    input_kinds: tuple[FieldKind, ...]
    output_kind: FieldKind
    commutative: bool = False
    associative: bool = False
    ordering_required: bool = False
    phase_observable: bool = False
    support_effect: str = "preserve"
    error_effect: str = "preserve"
    backend_capability: str = "core"


def _effect(
    inputs: Sequence[FieldKind],
    output: FieldKind,
    **options: Any,
) -> EffectInfo:
    return EffectInfo(tuple(inputs), output, **options)


ISA_EFFECTS: Mapping[ISAOpcode, EffectInfo] = MappingProxyType(
    {
        ISAOpcode.LOG_MUL: _effect((FieldKind.LOG_POS, FieldKind.LOG_POS), FieldKind.LOG_POS, commutative=True),
        ISAOpcode.LOG_DIV: _effect((FieldKind.LOG_POS, FieldKind.LOG_POS), FieldKind.LOG_POS),
        ISAOpcode.LOGADD_POS: _effect(
            (FieldKind.LOG_POS, FieldKind.LOG_POS),
            FieldKind.LOG_POS,
            commutative=True,
            associative=True,
            error_effect="quantize",
        ),
        ISAOpcode.LOGSUB_POS: _effect(
            (FieldKind.LOG_POS, FieldKind.LOG_POS), FieldKind.LOG_POS, error_effect="quantize"
        ),
        ISAOpcode.LOGADD_SIGNED: _effect(
            (FieldKind.LOG_SIGNED, FieldKind.LOG_SIGNED),
            FieldKind.LOG_SIGNED,
            commutative=True,
            error_effect="quantize",
        ),
        ISAOpcode.PHASE_ADD: _effect(
            (FieldKind.PHASE, FieldKind.PHASE), FieldKind.PHASE, commutative=True, associative=True
        ),
        ISAOpcode.PHASE_SUB: _effect((FieldKind.PHASE, FieldKind.PHASE), FieldKind.PHASE),
        ISAOpcode.PHASE_CONJ: _effect((FieldKind.PHASE,), FieldKind.PHASE),
        ISAOpcode.SPF_MUL: _effect(
            (FieldKind.SPF_COMPLEX, FieldKind.SPF_COMPLEX),
            FieldKind.SPF_COMPLEX,
            commutative=True,
            associative=True,
            phase_observable=True,
        ),
        ISAOpcode.SPF_DIV: _effect(
            (FieldKind.SPF_COMPLEX, FieldKind.SPF_COMPLEX), FieldKind.SPF_COMPLEX, phase_observable=True
        ),
        ISAOpcode.SPF_CONJ: _effect((FieldKind.SPF_COMPLEX,), FieldKind.SPF_COMPLEX, phase_observable=True),
        ISAOpcode.SPF_NEG: _effect((FieldKind.SPF_COMPLEX,), FieldKind.SPF_COMPLEX, phase_observable=True),
        ISAOpcode.SPF_ABS2: _effect((FieldKind.SPF_COMPLEX,), FieldKind.LOG_POS, phase_observable=False),
        ISAOpcode.SPF_PAIR_ADDSUB: _effect(
            (FieldKind.SPF_COMPLEX, FieldKind.SPF_COMPLEX),
            FieldKind.COMPLEX_PAIR,
            commutative=False,
            phase_observable=True,
            error_effect="reduce",
        ),
        ISAOpcode.JET_INIT: _effect((), FieldKind.JET2D, backend_capability="jet"),
        ISAOpcode.JET_AFFINE: _effect((FieldKind.JET2D,), FieldKind.JET2D, backend_capability="jet"),
        ISAOpcode.JET_ADD: _effect(
            (FieldKind.JET2D, FieldKind.JET2D), FieldKind.JET2D, commutative=True, backend_capability="jet"
        ),
        ISAOpcode.JET_FD_ROW: _effect((FieldKind.JET2D,), FieldKind.DENSE, backend_capability="jet"),
        ISAOpcode.JET_FD_TILE: _effect((FieldKind.JET2D,), FieldKind.DENSE, backend_capability="jet"),
        ISAOpcode.JET_EMIT_LOG: _effect(
            (FieldKind.JET2D,), FieldKind.LOG_POS, error_effect="materialize", backend_capability="jet"
        ),
        ISAOpcode.JET_EMIT_PHASE: _effect(
            (FieldKind.JET2D,), FieldKind.PHASE, error_effect="materialize", backend_capability="jet"
        ),
        ISAOpcode.JET_EMIT_SPF: _effect(
            (FieldKind.JET2D,), FieldKind.WORDS, error_effect="materialize", backend_capability="jet"
        ),
        ISAOpcode.ATOM_MUL: _effect(
            (FieldKind.ATOM, FieldKind.ATOM),
            FieldKind.ATOM,
            commutative=True,
            associative=True,
            backend_capability="atom",
        ),
        ISAOpcode.ATOM_DIV: _effect((FieldKind.ATOM, FieldKind.ATOM), FieldKind.ATOM, backend_capability="atom"),
        ISAOpcode.ATOM_AFFINE: _effect(
            (FieldKind.ATOM,), FieldKind.ATOM, support_effect="transform", backend_capability="atom"
        ),
        ISAOpcode.ATOM_FOURIER: _effect(
            (FieldKind.ATOM,), FieldKind.ATOM, support_effect="transform", backend_capability="atom"
        ),
        ISAOpcode.ATOM_INTEGRAL: _effect((FieldKind.ATOM,), FieldKind.DENSE, backend_capability="atom"),
        ISAOpcode.ATOM_DERIVATIVE: _effect(
            (FieldKind.ATOM, FieldKind.DENSE), FieldKind.DENSE, backend_capability="atom"
        ),
        ISAOpcode.ATOM_BOUND: _effect(
            (FieldKind.ATOM,), FieldKind.DENSE, support_effect="bound", backend_capability="atom"
        ),
        ISAOpcode.ATOM_TO_JET: _effect(
            (FieldKind.ATOM,), FieldKind.JET2D, error_effect="fit", backend_capability="atom"
        ),
        ISAOpcode.MIX_CONCAT: _effect(
            (FieldKind.ATOM_MIXTURE, FieldKind.ATOM_MIXTURE),
            FieldKind.ATOM_MIXTURE,
            commutative=True,
            associative=True,
            backend_capability="mixture",
        ),
        ISAOpcode.MIX_CULL: _effect(
            (FieldKind.ATOM_MIXTURE,),
            FieldKind.ATOM_MIXTURE,
            support_effect="shrink",
            error_effect="prune",
            backend_capability="mixture",
        ),
        ISAOpcode.MIX_BOUND: _effect(
            (FieldKind.ATOM_MIXTURE,), FieldKind.DENSE, support_effect="bound", backend_capability="mixture"
        ),
        ISAOpcode.MIX_FACTOR: _effect(
            (FieldKind.ATOM_MIXTURE,), FieldKind.MIXTURE_FACTOR, backend_capability="mixture"
        ),
        ISAOpcode.MIX_TO_JETS: _effect(
            (FieldKind.ATOM_MIXTURE,), FieldKind.JET2D, error_effect="fit", backend_capability="mixture"
        ),
        ISAOpcode.MIX_MATERIALIZE: _effect(
            (FieldKind.ATOM_MIXTURE,), FieldKind.DENSE, error_effect="materialize", backend_capability="mixture"
        ),
        ISAOpcode.REDUCE_POS_PIVOT: _effect(
            (FieldKind.LOG_POS,),
            FieldKind.REDUCTION,
            commutative=True,
            associative=True,
            error_effect="reduce",
            backend_capability="reducer",
        ),
        ISAOpcode.REDUCE_SIGNED_PIVOT: _effect(
            (FieldKind.LOG_SIGNED,),
            FieldKind.REDUCTION,
            commutative=True,
            error_effect="reduce",
            backend_capability="reducer",
        ),
        ISAOpcode.REDUCE_COMPLEX_PIVOT: _effect(
            (FieldKind.SPF_COMPLEX,),
            FieldKind.REDUCTION,
            commutative=True,
            phase_observable=True,
            error_effect="reduce",
            backend_capability="reducer",
        ),
        ISAOpcode.REDUCE_TROPICAL_CERT: _effect(
            (FieldKind.TROPICAL_MIN,),
            FieldKind.REDUCTION,
            commutative=True,
            associative=True,
            error_effect="prune",
            backend_capability="reducer",
        ),
        ISAOpcode.REDUCE_TRANSPORT_ORDERED: _effect(
            (FieldKind.TRANSPORT,),
            FieldKind.REDUCTION,
            associative=True,
            ordering_required=True,
            error_effect="reduce",
            backend_capability="transport",
        ),
        ISAOpcode.FFT_FORWARD: _effect(
            (FieldKind.WORDS,),
            FieldKind.WORDS,
            phase_observable=True,
            error_effect="transform",
            backend_capability="fft",
        ),
        ISAOpcode.FFT_INVERSE: _effect(
            (FieldKind.WORDS,),
            FieldKind.WORDS,
            phase_observable=True,
            error_effect="transform",
            backend_capability="fft",
        ),
        ISAOpcode.HERMITIAN_SYNTH: _effect(
            (FieldKind.HERMITIAN,), FieldKind.LOG_SIGNED, error_effect="reduce", backend_capability="hermitian"
        ),
        ISAOpcode.SPARSE_SYNTH: _effect(
            (FieldKind.DENSE, FieldKind.DENSE), FieldKind.DENSE, error_effect="materialize", backend_capability="sparse"
        ),
        ISAOpcode.SAMPLE_POINT: _effect((FieldKind.ANY, FieldKind.DENSE), FieldKind.DENSE, error_effect="materialize"),
        ISAOpcode.SAMPLE_FOOTPRINT: _effect(
            (FieldKind.ANY, FieldKind.ATOM), FieldKind.DENSE, error_effect="materialize"
        ),
        ISAOpcode.MATERIALIZE_TILE: _effect(
            (FieldKind.ANY,), FieldKind.MATERIALIZATION_RESULT, error_effect="materialize"
        ),
        ISAOpcode.AABB_TEST: _effect(
            (FieldKind.DENSE, FieldKind.DENSE), FieldKind.DENSE, support_effect="test", backend_capability="visibility"
        ),
        ISAOpcode.BVH_STEP: _effect(
            (FieldKind.DENSE,), FieldKind.DENSE, support_effect="test", backend_capability="visibility"
        ),
        ISAOpcode.DEPTH_MIN: _effect(
            (FieldKind.TROPICAL_MIN,),
            FieldKind.REDUCTION,
            commutative=True,
            associative=True,
            backend_capability="visibility",
        ),
        ISAOpcode.MASK_AND: _effect(
            (FieldKind.DENSE, FieldKind.DENSE),
            FieldKind.DENSE,
            commutative=True,
            associative=True,
            support_effect="shrink",
            backend_capability="visibility",
        ),
        ISAOpcode.MASK_OR: _effect(
            (FieldKind.DENSE, FieldKind.DENSE),
            FieldKind.DENSE,
            commutative=True,
            associative=True,
            support_effect="expand",
            backend_capability="visibility",
        ),
        ISAOpcode.TILE_BIN: _effect(
            (FieldKind.DENSE,), FieldKind.DENSE, support_effect="bin", backend_capability="visibility"
        ),
        ISAOpcode.SORT_KEY: _effect(
            (FieldKind.DENSE,), FieldKind.DENSE, ordering_required=True, backend_capability="visibility"
        ),
        ISAOpcode.VIS_SELECT: _effect(
            (FieldKind.DENSE, FieldKind.DENSE),
            FieldKind.DENSE,
            support_effect="shrink",
            backend_capability="visibility",
        ),
        ISAOpcode.DISPLAY_CONVERT: _effect((FieldKind.LOG_POS,), FieldKind.DISPLAY, error_effect="display"),
    }
)


@dataclass(frozen=True, slots=True)
class ISAValue:
    kind: FieldKind | str
    domain: Domain | str = Domain.CUSTOM
    coherence: Coherence | str = Coherence.NOT_APPLICABLE

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, FieldKind) else FieldKind(str(self.kind).lower())
        domain = self.domain if isinstance(self.domain, Domain) else Domain(str(self.domain).upper())
        coherence = self.coherence if isinstance(self.coherence, Coherence) else Coherence(str(self.coherence).upper())
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "coherence", coherence)


@dataclass(frozen=True, slots=True)
class ISAInstruction:
    opcode: ISAOpcode | str
    output: str
    inputs: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    output_value: ISAValue | None = None

    def __post_init__(self) -> None:
        opcode = self.opcode if isinstance(self.opcode, ISAOpcode) else ISAOpcode(str(self.opcode).upper())
        if not str(self.output):
            raise ValueError("ISA output must be non-empty")
        object.__setattr__(self, "opcode", opcode)
        object.__setattr__(self, "output", str(self.output))
        object.__setattr__(self, "inputs", tuple(str(value) for value in self.inputs))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, slots=True)
class ISAProgram:
    instructions: tuple[ISAInstruction, ...]
    inputs: Mapping[str, ISAValue] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        instructions = tuple(self.instructions)
        inputs = MappingProxyType(dict(self.inputs))
        available: dict[str, ISAValue] = dict(inputs)
        for instruction in instructions:
            opcode = cast(ISAOpcode, instruction.opcode)
            if instruction.output in available:
                raise ValueError(f"duplicate ISA value {instruction.output!r}")
            missing = [name for name in instruction.inputs if name not in available]
            if missing:
                raise ValueError(f"{opcode.value} has missing/forward inputs: {', '.join(missing)}")
            effect = ISA_EFFECTS[opcode]
            variadic_reducer = (
                len(effect.input_kinds) == 1 and len(instruction.inputs) >= 1 and opcode.name.startswith("REDUCE_")
            )
            if len(instruction.inputs) != len(effect.input_kinds) and not variadic_reducer:
                raise ValueError(f"{opcode.value} input arity mismatch")
            for name, expected in zip(instruction.inputs, effect.input_kinds, strict=False):
                actual = cast(FieldKind, available[name].kind)
                if expected is not FieldKind.ANY and actual is not expected:
                    raise TypeError(f"{opcode.value} requires {expected.value}, received {actual.value} from {name!r}")
            result = instruction.output_value or ISAValue(effect.output_kind)
            available[instruction.output] = result
        outputs = tuple(self.outputs) or ((instructions[-1].output,) if instructions else ())
        if any(name not in available for name in outputs):
            raise ValueError("ISA outputs must exist")
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)


__all__ = [
    "ISA_EFFECTS",
    "EffectInfo",
    "ISAInstruction",
    "ISAOpcode",
    "ISAProgram",
    "ISAValue",
]
