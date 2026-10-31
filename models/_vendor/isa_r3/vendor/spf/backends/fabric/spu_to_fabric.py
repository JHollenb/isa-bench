"""Lower an Appendix-C SPU micro-op program onto ``pan-fabricd`` graph ops.

The question this answers: **how much of the v0.2 Spectral Processing Unit can
run on the KV260 today through the PAN fabric, with no new synthesis?**

The daemon registers exactly five SPF graph kernels
(``pan-isa/pan_fabricd/src/pan_fabricd.cpp:557-561``) — ``spf32.mul``,
``spf32.add``, ``spf32.div``, ``spf32.abs2``, ``spf32.fft64`` — plus an
integer ``mlp.gemm_k16`` (``:562``) that is shape-incoherent with the rest of
the stack. ``spf32.conj`` has an opcode constant (``:55``) but is never
registered, so the daemon answers ``UNSUPPORTED`` for it.

Those five are all *multiplicative or transform* ops on packed SPF32 words —
except ``spf32.add``, which is the one op that crosses the addition wall. The
lowerings below exploit the SPF word algebra to reach the log/phase micro-ops:

======================  ====================================  ==============
Appendix-C micro-op     fabric graph                          fidelity
======================  ====================================  ==============
``LADD.SAT  a,b``       ``spf32.mul(a, pack(b, 0))``          exact
``LSUB.SAT  a,b``       ``spf32.div(a, pack(b, 0))``          exact
``TADD.MOD  a,b``       ``spf32.mul(a, pack(0, b))``          exact
``TSUB.MOD  a,b``       ``spf32.div(a, pack(0, b))``          exact
``LOGADD.ROM a,b``      ``spf32.add`` at equal phase          <=1 L-code
``LOGSUB.ROM a,b``      ``spf32.add`` at anti-phase           <=1 L-code
``SPF.MUL``             ``spf32.mul``                         exact
``SPF.PAIR  a,b``       ``mul`` (phase flip) + 2x ``add``     <=1 L-code
``TRN.COMPOSE``         2x ``mul`` + 1x ``add`` per channel   <=1 L-code
======================  ====================================  ==============

``spf32.mul`` adds L-codes and phases; ``spf32.div`` subtracts them.  Packing
one operand as ``(L, 0)`` therefore makes ``mul``/``div`` a saturating
log-add/sub, and packing it as ``(0, theta)`` makes them a modular phase
add/sub.  ``spf32.add`` at equal phase is exactly ``logadd_pos``; at
anti-phase it is ``logsub_pos`` (see ``spf.tensor.v02._pair_local``, which
dispatches to those two ROMs for precisely those two phase relations).

Everything else in Appendix C — the jet forward-difference unit (§38.4), the
pivot reducer (§38.5), the tropical selects and the bound test — has **no**
registered fabric op, and is emitted fail-closed with a reason.

See ``obsidian/spf/2026-09-19-spu-on-the-pan-fabric-assessment.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType

import numpy as np

from spf.renderer.backends import SPUInstruction, SPUOpcode
from spf.tensor import v02 as arithmetic

from .panfg1 import DAEMON_REGISTERED_OPS, FabricGraph, FabricOp, encode_request

#: Per-dispatch cost of one PANFG1 graph node.
#:
#: ``pl_dispatch_seconds`` / ``pl_dispatch_joules`` are the cost model's own
#: ``spf32.mul`` entries (``pan-isa/pan_isa_x86_runtime_spf/fabric_graph.py:178-213``).
#: That model is named ``measured_kv260_v0`` but is *size-independent*
#: (``fabric_graph.py:221-222`` ignores the element count), so treat it as a
#: calibrated constant, not a per-run measurement.
#:
#: ``packet_roundtrip_seconds`` / ``packet_daemon_seconds`` ARE measured on the
#: board, for a one-op four-element graph (``pan-isa/STATUS.md:3045-3047``).
PL_DISPATCH_SECONDS = 3.5233469009399415e-4
PL_DISPATCH_JOULES = 2.679169471040368 / 20000
PACKET_ROUNDTRIP_SECONDS = 1.3132540007063653e-3
PACKET_DAEMON_SECONDS = 8.56248e-4

#: ``spf32.fft64`` is rejected unless the buffer is exactly 64 elements
#: (``pan_fabricd.cpp:956-957``); every op is capped at ``--max-n`` (``:213``).
FFT64_ELEMENTS = 64
MAX_ELEMENTS = 1 << 20

_PHASE_HALF = arithmetic.SPF32_PHASE_MOD // 2


def pack_word(L: int, theta: int) -> int:
    """Pack an SPF32 word from an L-code and a phase code."""
    return (int(np.uint16(np.int16(L))) << 16) | int(np.uint16(theta))


@dataclass(frozen=True, slots=True)
class Unsupported:
    """One micro-op the fabric cannot execute, with the reason it cannot."""

    index: int
    opcode: str
    cls: str
    reason: str


@dataclass
class LoweringResult:
    graph: FabricGraph
    constants: dict[str, int]
    unsupported: list[Unsupported] = field(default_factory=list)
    lowered: list[int] = field(default_factory=list)
    fidelity_notes: dict[str, str] = field(default_factory=dict)

    @property
    def executed_fraction(self) -> float:
        total = len(self.lowered) + len(self.unsupported)
        return len(self.lowered) / total if total else 0.0

    @property
    def dispatches(self) -> int:
        return len(self.graph.ops)

    def cost(self) -> dict[str, float]:
        n = self.dispatches
        return {
            "dispatches": float(n),
            "dispatches_per_lowered_microop": (n / len(self.lowered)) if self.lowered else 0.0,
            "modelled_pl_seconds": n * PL_DISPATCH_SECONDS,
            "modelled_pl_joules": n * PL_DISPATCH_JOULES,
            "measured_packet_roundtrip_seconds_if_one_op_per_packet": n * PACKET_ROUNDTRIP_SECONDS,
            "measured_packet_roundtrip_seconds_if_one_packet": PACKET_ROUNDTRIP_SECONDS,
        }

    def bind(
        self, payloads: Mapping[str, Sequence[int] | np.ndarray], *, elements: int = 1
    ) -> dict[str, np.ndarray]:
        """Merge caller payloads with the lowering's synthesized constant words.

        A lowering that packs an operand as ``(L, 0)``, ``(0, theta)`` or the
        anti-phase flip ``(0, 2^15)`` needs those words uploaded as ordinary
        graph inputs; they are broadcast to ``elements``.
        """
        bound = {k: np.asarray(v, dtype=np.uint32).reshape(-1) for k, v in payloads.items()}
        for name, word in self.constants.items():
            if name not in bound:
                bound[name] = np.full(elements, np.uint32(word), dtype=np.uint32)
        return bound

    def encode(self, payloads: Mapping[str, Sequence[int] | np.ndarray], **kw: object) -> bytes:
        return encode_request(self.graph, self.bind(payloads), **kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# class (b)/(c) reasons
# ---------------------------------------------------------------------------

#: Micro-ops with no registered fabric op. ``cls`` is the path that WOULD
#: realize them: ``b`` = a small HLS kernel through the existing Vitis flow
#: (source already exists in the SPF tree where noted); ``c`` = would need
#: learned/direct bitstream mapping. Nothing here is class (c)-only: every
#: entry has a class (b) route, so (c) is an alternative, never a requirement.
_UNSUPPORTED: Mapping[SPUOpcode, tuple[str, str]] = {
    SPUOpcode.SPF_CONJ: (
        "b",
        "spf32.conj has an opcode constant (pan_fabricd.cpp:55) but is never passed to "
        "add_graph_kernel (:557-562), so the daemon answers UNSUPPORTED; the spf32_conj HLS "
        "kernel exists (hw/hls/spf32_conj) and is in the fb7 overlays but not in spf32_core_pl. "
        "Fix = one daemon registration line + a conj CU in the loaded xclbin.",
    ),
    SPUOpcode.JET_SET: (
        "b",
        "Jet header init (design v0.2 §38.4). No fabric op. hw/hls/spf_ramp folds the whole "
        "polynomial phase ramp per tile (kernel version 2 = ABI v23 full Hessian) but is "
        "unsynthesized: no baked overlay carries a ramp CU.",
    ),
    SPUOpcode.JET_STEP: (
        "b",
        "Forward-difference recurrence value+=d1; d1+=d2 (§38.4). No fabric op. The batch "
        "equivalent is one spf_ramp dispatch per tile, not one per sample -- but spf_ramp is "
        "phase-only (Theta += ramp(q), L untouched) and the SPU jet carries magnitude too.",
    ),
    SPUOpcode.JET_NEXTROW: ("b", "Row-advance of the 2D forward-difference state (§38.4). Same as JET.STEP."),
    SPUOpcode.PIVOT_MAX: (
        "b",
        "Pivot reducer max-L scan (§38.5). No fabric op. hw/hls/spf_reduce exists as a kernel "
        "source but no shipped overlay resides it (src/spf/tensor/pl_lowering.py:66-70).",
    ),
    SPUOpcode.PIVOT_ADDPOS: ("b", "Aligned-mantissa integer accumulate (§38.5). See PIVOT.MAX."),
    SPUOpcode.PIVOT_ADDSIGN: ("b", "Signed pivot accumulate (§38.5). See PIVOT.MAX."),
    SPUOpcode.PIVOT_ADDXY: ("b", "Complex/atan2 pivot accumulate (§38.5). See PIVOT.MAX."),
    SPUOpcode.PIVOT_FINAL: ("b", "Leading-one/log finalizer + optional atan2 (§38.5). See PIVOT.MAX."),
    SPUOpcode.TROP_MIN: (
        "b",
        "Tropical min over L-codes. No fabric op: every registered SPF op is arithmetic, none "
        "is a compare/select. spf_reduce's TROPICAL domain is the natural home.",
    ),
    SPUOpcode.TROP_MAX: ("b", "Tropical max over L-codes. See TROP.MIN."),
    SPUOpcode.BOUND_TEST: (
        "b",
        "Predicate contribution <= budget (§38 control plane). No fabric op returns a mask; the "
        "daemon's only dtype is u32 (pan_fabricd.cpp:933) and there is no compare kernel.",
    ),
}


class SPUToFabricLowering:
    """Lower :class:`SPUInstruction` programs to a PANFG1 graph, fail-closed."""

    def __init__(self, *, elements: int = 1) -> None:
        if not 0 < elements <= MAX_ELEMENTS:
            raise ValueError(f"elements must be in 1..{MAX_ELEMENTS} (pan_fabricd.cpp:213,934)")
        self.elements = int(elements)

    # -- helpers ---------------------------------------------------------
    def _const(self, result: LoweringResult, name: str, word: int) -> str:
        result.constants[name] = int(word)
        return name

    # -- main ------------------------------------------------------------
    def lower(self, program: Sequence[SPUInstruction]) -> LoweringResult:
        n = self.elements
        result = LoweringResult(FabricGraph((), (), ()), {})
        pending: list[FabricOp] = []
        defined: set[str] = set()
        inputs: list[str] = []
        last_out: str | None = None

        def src(name: str) -> str:
            if name not in defined:
                defined.add(name)
                inputs.append(name)
            return name

        def emit(op: str, ins: tuple[str, ...], out: str) -> str:
            if op not in DAEMON_REGISTERED_OPS:
                raise AssertionError(f"{op!r} is not registered by pan-fabricd")
            pending.append(FabricOp(f"n{len(pending)}", op, ins, out, n))
            defined.add(out)
            return out

        for i, ins_ in enumerate(program):
            op = ins_.opcode
            dst = ins_.dst if isinstance(ins_.dst, str) else ins_.dst[0]

            if op in _UNSUPPORTED:
                cls, reason = _UNSUPPORTED[op]
                result.unsupported.append(Unsupported(i, op.value, cls, reason))
                continue

            if op is SPUOpcode.SPF_MUL:
                emit("spf32.mul", (src(ins_.sources[0]), src(ins_.sources[1])), dst)

            elif op in {SPUOpcode.LADD_SAT, SPUOpcode.LSUB_SAT}:
                # L-code add/sub: pack the second operand as (L, theta=0) so the
                # word multiply/divide IS the saturating log add/sub.
                fab = "spf32.mul" if op is SPUOpcode.LADD_SAT else "spf32.div"
                b = ins_.sources[1] if len(ins_.sources) > 1 else None
                if b is None:
                    b = self._const(result, f"_k{i}_L", pack_word(int(ins_.immediate), 0))
                emit(fab, (src(ins_.sources[0]), src(b)), dst)
                result.fidelity_notes[op.value] = "exact: word L-plane add/sub with saturation"

            elif op in {SPUOpcode.TADD_MOD, SPUOpcode.TSUB_MOD}:
                # Phase add/sub: pack the second operand as (L=0, theta).
                fab = "spf32.mul" if op is SPUOpcode.TADD_MOD else "spf32.div"
                b = ins_.sources[1] if len(ins_.sources) > 1 else None
                if b is None:
                    b = self._const(result, f"_k{i}_T", pack_word(0, int(ins_.immediate)))
                emit(fab, (src(ins_.sources[0]), src(b)), dst)
                result.fidelity_notes[op.value] = "exact: word Theta-plane add/sub mod 2^16"

            elif op is SPUOpcode.LOGADD_ROM:
                emit("spf32.add", (src(ins_.sources[0]), src(ins_.sources[1])), dst)
                result.fidelity_notes[op.value] = (
                    "requires equal operand phase; spf32_add CU is fp32 LUT+3NR, not the int16 ROM"
                )

            elif op is SPUOpcode.LOGSUB_ROM:
                flip = self._const(result, "_pi", pack_word(0, _PHASE_HALF))
                nb = emit("spf32.mul", (src(ins_.sources[1]), src(flip)), f"_{dst}__anti")
                emit("spf32.add", (src(ins_.sources[0]), nb), dst)
                result.fidelity_notes[op.value] = "anti-phase add; same fp32 caveat as LOGADD.ROM"

            elif op is SPUOpcode.SPF_PAIR:
                a, b = src(ins_.sources[0]), src(ins_.sources[1])
                top = ins_.dst[0] if isinstance(ins_.dst, tuple) else dst
                bot = ins_.dst[1] if isinstance(ins_.dst, tuple) and len(ins_.dst) > 1 else f"_{dst}__bot"
                flip = self._const(result, "_pi", pack_word(0, _PHASE_HALF))
                nb = emit("spf32.mul", (b, src(flip)), f"_{top}__nb")
                emit("spf32.add", (a, b), top)
                emit("spf32.add", (a, nb), bot)
                result.fidelity_notes[op.value] = "FB3 add/sub pair as 3 dispatches; fp32 add caveat"

            elif op is SPUOpcode.TRN_COMPOSE:
                # §38.6: T = T1*T2; C = C1 + T1*C2. Sources are (T1,C1,T2,C2).
                s = ins_.sources
                if len(s) != 4:
                    result.unsupported.append(
                        Unsupported(i, op.value, "b", "TRN.COMPOSE lowering needs (T1,C1,T2,C2) word sources")
                    )
                    continue
                t1, c1, t2, c2 = (src(x) for x in s)
                dt = ins_.dst if isinstance(ins_.dst, tuple) else (dst, f"_{dst}__C")
                emit("spf32.mul", (t1, t2), dt[0])
                att = emit("spf32.mul", (t1, c2), f"_{dt[0]}__att")
                emit("spf32.add", (c1, att), dt[1])
                result.fidelity_notes[op.value] = (
                    "value words only: 2x mul + 1x add. The Transport error ledger "
                    "(v02.transport_compose:5515-5545) is float bookkeeping and stays on the host."
                )
            else:  # pragma: no cover - enum is exhaustive above
                result.unsupported.append(Unsupported(i, op.value, "b", "no lowering rule"))
                continue

            result.lowered.append(i)
            last_out = dst

        outputs = tuple(dict.fromkeys([o for o in (last_out,) if o]))
        result.graph = FabricGraph(tuple(pending), tuple(inputs), outputs)
        return result


# ---------------------------------------------------------------------------
# host-side execution of the lowered graph (what the daemon would do on PL)
# ---------------------------------------------------------------------------

def simulate_graph(
    graph: FabricGraph,
    payloads: Mapping[str, Sequence[int] | np.ndarray],
    *,
    add_reference: str = "spec_rom",
) -> dict[str, np.ndarray]:
    """Execute a PANFG1 graph through ``spf.tensor.v02`` word arithmetic.

    This is the no-board path.  ``add_reference`` selects which ``spf32.add``
    the daemon is modelled as dispatching:

    * ``"spec_rom"`` -- ``spf_pair_addsub_hybrid(precision="exact")``, i.e. the
      LOGADD_PLUS/LOGADD_MINUS ROM the design mandates.
    * ``"cu_float"`` -- ``spf_add_complex_reference``, the algorithm the
      shipped ``spf32_add`` HLS CU implements (recovered at ``4eb3b03f^``:
      fp32 exp2/cos/sin/atan2/log2 with a Q0.15 seed LUT + 3 Newton iters),
      evaluated here in float64.

    The two differ, and the difference is the whole fidelity question: see the
    assessment note.  Nothing here touches hardware.
    """
    env: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.uint32).reshape(-1) for k, v in payloads.items()
    }
    for op in graph.ops:
        a = env[op.inputs[0]]
        if op.op == "spf32.mul":
            env[op.output] = np.asarray(arithmetic.spf_mul(a, env[op.inputs[1]]), dtype=np.uint32)
        elif op.op == "spf32.div":
            env[op.output] = np.asarray(arithmetic.spf_div(a, env[op.inputs[1]]), dtype=np.uint32)
        elif op.op == "spf32.abs2":
            # v02.spf_abs2 returns a typed LogPos, not a packed word; the CU
            # writes back a word with zero phase.
            env[op.output] = np.asarray(
                [pack_word(int(arithmetic.spf_abs2(int(x)).L), 0) for x in a], dtype=np.uint32
            )
        elif op.op == "spf32.add":
            b = env[op.inputs[1]]
            if add_reference == "cu_float":
                out = [arithmetic.spf_add_complex_reference(int(x), int(y)) for x, y in zip(a, b, strict=True)]
            else:
                out = []
                for x, y in zip(a, b, strict=True):
                    pair = arithmetic.spf_pair_addsub_hybrid(int(x), int(y), precision="exact")
                    out.append(int(pair[0] if isinstance(pair, (tuple, list)) else pair))
            env[op.output] = np.asarray(out, dtype=np.uint32)
        elif op.op == "spf32.fft64":
            from spf.fft import spf32_fft

            if a.size != FFT64_ELEMENTS:
                raise ValueError("spf32.fft64 requires exactly 64 elements (pan_fabricd.cpp:956-957)")
            env[op.output] = np.asarray(spf32_fft.fft(a), dtype=np.uint32)
        else:  # pragma: no cover
            raise ValueError(f"simulate_graph: unregistered fabric op {op.op!r}")
    return env


__all__ = [
    "PL_DISPATCH_JOULES",
    "PL_DISPATCH_SECONDS",
    "LoweringResult",
    "SPUToFabricLowering",
    "Unsupported",
    "pack_word",
    "simulate_graph",
]


# ---------------------------------------------------------------------------
# adapter for the SPU hardware ISA assembler (hw/spu/isa.py)
# ---------------------------------------------------------------------------

def load_spu_isa(hw_root: str | None = None) -> ModuleType | None:
    """Import ``hw/spu/isa.py`` (the SPU RTL agent's 64-bit ISA), or ``None``.

    Read-only: this never writes to ``hw/spu``.  Returns the module so callers
    can use ``MicroOp``, ``parse_asm``, ``ISA_LOWERING`` and ``lower_isa_op``.
    """
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(hw_root) if hw_root else Path(__file__).resolve().parents[4] / "hw"
    path = root / "spu" / "isa.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_spu_isa", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: hw/spu/isa.py uses @dataclass(slots=True), whose
    # field resolution looks the defining module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: Appendix-C micro-ops with a lowering onto a registered ``pan-fabricd`` op.
CLASS_A_MICROOPS = frozenset(
    {
        "LADD.SAT",
        "LSUB.SAT",
        "TADD.MOD",
        "TSUB.MOD",
        "LOGADD.ROM",
        "LOGSUB.ROM",
        "SPF.MUL",
        "SPF.PAIR",
        "TRN.COMPOSE",
    }
)


def classify_isa_ops(isa_module: ModuleType, n: int = 1) -> dict[str, dict[str, object]]:
    """Classify every §35 ISA op by expanding it through the SPU's own table.

    An ISA op is class (a) only when **every** micro-op it expands to is class
    (a); otherwise it is class (b) and the blocking micro-ops are named.
    """
    out: dict[str, dict[str, object]] = {}
    for name, low in isa_module.ISA_LOWERING.items():
        if low.unlowered_reason is not None:
            out[name] = {"cls": "b", "reason": low.unlowered_reason, "blockers": (), "n_microops": 0}
            continue
        ops = [m.mnemonic for m in low.expand(n)]
        blockers = sorted({m for m in ops if m not in CLASS_A_MICROOPS})
        out[name] = {
            "cls": "a" if not blockers else "b",
            "section": low.section,
            "blockers": tuple(blockers),
            "n_microops": len(ops),
        }
    return out
