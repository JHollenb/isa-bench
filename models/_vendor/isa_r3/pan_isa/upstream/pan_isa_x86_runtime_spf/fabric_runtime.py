"""PAN fabric registry and KV260 SPF kernel runtime.

This module is the first PS/PL control plane above the x86 transition router.
It treats existing KV260 xclbins as typed SPF operation providers rather than
pretending they implement the PAN x86 transition-packet ABI.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence


DEFAULT_FABRIC_WORKDIR = "/home/ubuntu/spfr"
DEFAULT_FABRIC_XCLBIN_DIRS = (
    "/home/ubuntu/spfr/xclbins",
    "/home/ubuntu/spfr/xclbins-pr197",
)


KERNEL_CATALOG: dict[str, dict[str, Any]] = {
    "spf32_add_kernel": {
        "op": "spf32.add",
        "arity": 2,
        "dtype": "spf32_word_u32",
        "call_abi": "binary_u32_n",
        "validation": "spf32_exact",
    },
    "spf32_mul_kernel": {
        "op": "spf32.mul",
        "arity": 2,
        "dtype": "spf32_word_u32",
        "call_abi": "binary_u32_n",
        "validation": "spf32_exact",
    },
    "spf32_div_kernel": {
        "op": "spf32.div",
        "arity": 2,
        "dtype": "spf32_word_u32",
        "call_abi": "binary_u32_n",
        "validation": "spf32_exact",
    },
    "spf32_abs2_kernel": {
        "op": "spf32.abs2",
        "arity": 1,
        "dtype": "spf32_word_u32",
        "call_abi": "unary_u32_n",
        "validation": "spf32_exact",
    },
    "spf32_conj_kernel": {
        "op": "spf32.conj",
        "arity": 1,
        "dtype": "spf32_word_u32",
        "call_abi": "unary_u32_n",
        "validation": "spf32_exact",
    },
    "spf32_neg_kernel": {
        "op": "spf32.neg",
        "arity": 1,
        "dtype": "spf32_word_u32",
        "call_abi": "unary_u32_n",
        "validation": "spf32_exact",
    },
    "spf32_fft64_kernel": {
        "op": "spf32.fft64",
        "arity": 1,
        "dtype": "spf32_word_u32",
        "call_abi": "fft64_u32",
        "fixed_elements": 64,
        "validation": "spf32_fft64",
        "requires_scalar_n": True,
    },
    "spf32_fft64_fb7_kernel": {
        "op": "spf32.fft64_fb7",
        "arity": 1,
        "dtype": "spf32_word_u32",
        "call_abi": "fft64_u32",
        "fixed_elements": 64,
        "validation": "spf32_fft64_fb7",
    },
    "spf32_fft64_fb7_fields_kernel": {
        "op": "spf32.fft64_fb7_fields",
        "arity": 1,
        "dtype": "spf32_word_u32",
        "call_abi": "fft64_u32",
        "fixed_elements": 64,
        "validation": "spf32_fft64_fb7_fields",
        "requires_scalar_n": True,
    },
    "spf32_fb7_bittrick_kernel": {
        "op": "spf32.fb7_bittrick",
        "arity": 1,
        "dtype": "spf32_word_u32",
        "call_abi": "unary_u32_n",
        "validation": "unsafe",
        "unsafe": True,
        "unsafe_reason": "kv260_cli_run_signal_11",
    },
    "spf32_fb7_bittrick256_kernel": {
        "op": "spf32.fb7_bittrick256",
        "arity": 1,
        "dtype": "spf32_word_u32",
        "call_abi": "unary_u32_n",
        "fixed_elements": 256,
        "validation": "unsafe",
        "unsafe": True,
        "unsafe_reason": "kv260_wedged_board_during_base_op_sweep",
    },
    "spf16_abs2_kernel": {
        "op": "spf16.abs2",
        "arity": 1,
        "dtype": "spf16_word_u16_packed",
        "call_abi": "unary_u32_n",
        "validation": "spf16_packed_exact",
    },
    "spf16_conj_kernel": {
        "op": "spf16.conj",
        "arity": 1,
        "dtype": "spf16_word_u16_packed",
        "call_abi": "unary_u32_n",
        "validation": "spf16_packed_exact",
    },
    "spf16_mul_kernel": {
        "op": "spf16.mul",
        "arity": 2,
        "dtype": "spf16_word_u16_packed",
        "call_abi": "binary_u32_n",
        "validation": "spf16_packed_exact",
    },
    "spf16_div_kernel": {
        "op": "spf16.div",
        "arity": 2,
        "dtype": "spf16_word_u16_packed",
        "call_abi": "binary_u32_n",
        "validation": "spf16_packed_exact",
    },
    "spf16_neg_kernel": {
        "op": "spf16.neg",
        "arity": 1,
        "dtype": "spf16_word_u16_packed",
        "call_abi": "unary_u32_n",
        "validation": "spf16_packed_exact",
    },
    "spf16_fft64_fb7_kernel": {
        "op": "spf16.fft64_fb7",
        "arity": 1,
        "dtype": "spf16_word_u16_packed",
        "call_abi": "fft64_u32",
        "fixed_elements": 64,
        "validation": "spf16_fft64_fb7",
    },
    "spf16_fft64_fb7_sdf_kernel": {
        "op": "spf16.fft64_fb7_sdf",
        "arity": 1,
        "dtype": "spf16_word_u16_packed",
        "call_abi": "fft64_u32",
        "fixed_elements": 64,
        "validation": "spf16_fft64_fb7_sdf",
    },
    "mlp_gemm_k16_kernel": {
        "op": "mlp.gemm_k16",
        "arity": 2,
        "dtype": "u32_word",
        "call_abi": "binary_u32_n",
        "validation": "integer_exact",
        "fixed_k": 16,
    },
}

UNSAFE_FABRIC_OPS = frozenset(
    str(row["op"])
    for row in KERNEL_CATALOG.values()
    if bool(row.get("unsafe"))
)


@dataclass(frozen=True)
class FabricKernelArg:
    name: str
    arg_id: int
    kind: str
    type_name: str
    port: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "id": self.arg_id,
            "kind": self.kind,
            "type": self.type_name,
            "port": self.port,
        }


@dataclass(frozen=True)
class FabricKernelSpec:
    op: str
    kernel_name: str
    xclbin_path: str
    app_name: str
    arity: int
    dtype: str
    call_abi: str
    args: tuple[FabricKernelArg, ...] = ()
    fixed_elements: Optional[int] = None
    validation: str = "raw_u32"
    unsafe: bool = False
    unsafe_reason: Optional[str] = None
    requires_scalar_n: bool = False
    source: str = "xclbin"

    def to_json(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "kernel_name": self.kernel_name,
            "xclbin_path": self.xclbin_path,
            "app_name": self.app_name,
            "arity": self.arity,
            "dtype": self.dtype,
            "call_abi": self.call_abi,
            "fixed_elements": self.fixed_elements,
            "validation": self.validation,
            "unsafe": self.unsafe,
            "unsafe_reason": self.unsafe_reason,
            "requires_scalar_n": self.requires_scalar_n,
            "args": [arg.to_json() for arg in self.args],
            "source": self.source,
        }


@dataclass(frozen=True)
class FabricRouteDecision:
    op: str
    element_count: int
    requested_backend: str
    selected_backend: str
    reason: str
    kernel: Optional[FabricKernelSpec] = None
    estimated_transfer_bytes: int = 0
    min_pl_elements: int = 1024
    active_app: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "element_count": self.element_count,
            "requested_backend": self.requested_backend,
            "selected_backend": self.selected_backend,
            "reason": self.reason,
            "estimated_transfer_bytes": self.estimated_transfer_bytes,
            "min_pl_elements": self.min_pl_elements,
            "active_app": self.active_app,
            "kernel": None if self.kernel is None else self.kernel.to_json(),
        }


@dataclass(frozen=True)
class FabricDispatchResult:
    op: str
    element_count: int
    kernel_name: str
    xclbin_path: str
    output_words: Any
    transfer_to_pl_bytes: int
    transfer_from_pl_bytes: int
    elapsed_seconds: float

    def to_json(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "element_count": self.element_count,
            "kernel_name": self.kernel_name,
            "xclbin_path": self.xclbin_path,
            "transfer_to_pl_bytes": self.transfer_to_pl_bytes,
            "transfer_from_pl_bytes": self.transfer_from_pl_bytes,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class FabricRegistry:
    kernels: list[FabricKernelSpec] = field(default_factory=list)
    workdir: str = DEFAULT_FABRIC_WORKDIR
    xclbin_dirs: tuple[str, ...] = DEFAULT_FABRIC_XCLBIN_DIRS
    errors: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def discover(
        cls,
        *,
        workdir: Optional[str] = None,
        xclbin_dirs: Optional[Sequence[str]] = None,
    ) -> "FabricRegistry":
        resolved_workdir = workdir or os.environ.get("PAN_FABRIC_WORKDIR") or DEFAULT_FABRIC_WORKDIR
        dirs = tuple(xclbin_dirs or _default_xclbin_dirs(resolved_workdir))
        registry = cls(workdir=resolved_workdir, xclbin_dirs=dirs)
        seen: set[tuple[str, str, str]] = set()
        for directory in dirs:
            root = Path(directory)
            if not root.is_dir():
                registry.errors.append({"path": str(root), "error": "xclbin_dir_missing"})
                continue
            for path in sorted(root.glob("*.xclbin")):
                try:
                    specs = specs_from_xclbin(path)
                except Exception as exc:
                    registry.errors.append({"path": str(path), "error": str(exc)})
                    continue
                for spec in specs:
                    key = (spec.op, spec.kernel_name, spec.xclbin_path)
                    if key not in seen:
                        seen.add(key)
                        registry.kernels.append(spec)
        registry.kernels.sort(key=lambda item: (item.op, _kernel_rank(item), item.app_name, item.kernel_name))
        return registry

    def specs_for_op(self, op: str, *, include_unsafe: bool = False) -> list[FabricKernelSpec]:
        return [
            spec
            for spec in self.kernels
            if spec.op == op and (include_unsafe or not spec.unsafe)
        ]

    def select_kernel(
        self,
        op: str,
        *,
        active_app: Optional[str] = None,
        include_unsafe: bool = False,
    ) -> Optional[FabricKernelSpec]:
        specs = self.specs_for_op(op, include_unsafe=include_unsafe)
        if not specs:
            return None
        return sorted(specs, key=lambda item: _selection_rank(item, active_app=active_app))[0]

    def unsafe_specs_for_op(self, op: str) -> list[FabricKernelSpec]:
        return [spec for spec in self.kernels if spec.op == op and spec.unsafe]

    def to_json(self) -> dict[str, Any]:
        ops = sorted({spec.op for spec in self.kernels})
        return {
            "schema": "pan_fabric_registry_v1",
            "workdir": self.workdir,
            "xclbin_dirs": list(self.xclbin_dirs),
            "kernel_count": len(self.kernels),
            "ops": ops,
            "kernels": [spec.to_json() for spec in self.kernels],
            "errors": list(self.errors),
        }


class FabricRouter:
    """Policy layer for deciding whether a typed SPF op should run on PS or PL."""

    def __init__(self, *, min_pl_elements: int = 1024, backend: str = "auto"):
        if backend not in {"auto", "ps", "pl"}:
            raise ValueError(f"unsupported fabric backend {backend!r}")
        self.min_pl_elements = int(min_pl_elements)
        self.backend = backend

    def choose(
        self,
        registry: FabricRegistry,
        *,
        op: str,
        element_count: int,
        active_app: Optional[str] = None,
    ) -> FabricRouteDecision:
        spec = registry.select_kernel(op, active_app=active_app)
        element_count = int(element_count)
        if self.backend == "ps":
            return self._decision(op, element_count, "ps", "forced_ps", None, active_app)
        if spec is None:
            if registry.unsafe_specs_for_op(op):
                return self._decision(op, element_count, "ps", "unsafe_pl_kernel", None, active_app)
            return self._decision(op, element_count, "ps", "no_pl_kernel", None, active_app)
        if spec.fixed_elements is not None and element_count != spec.fixed_elements:
            return self._decision(op, element_count, "ps", "fixed_size_mismatch", spec, active_app)
        if self.backend == "pl":
            return self._decision(op, element_count, "pl", "forced_pl", spec, active_app)
        if spec.fixed_elements is not None:
            return self._decision(op, element_count, "pl", "fixed_size_pl_primitive", spec, active_app)
        if element_count >= self.min_pl_elements:
            return self._decision(op, element_count, "pl", "large_typed_spf_op", spec, active_app)
        return self._decision(op, element_count, "ps", "below_pl_threshold", spec, active_app)

    def _decision(
        self,
        op: str,
        element_count: int,
        selected_backend: str,
        reason: str,
        spec: Optional[FabricKernelSpec],
        active_app: Optional[str],
    ) -> FabricRouteDecision:
        arity = 1 if spec is None else int(spec.arity)
        transfer_to = element_count * 4 * arity
        transfer_from = element_count * 4 if selected_backend == "pl" else 0
        return FabricRouteDecision(
            op=op,
            element_count=element_count,
            requested_backend=self.backend,
            selected_backend=selected_backend,
            reason=reason,
            kernel=spec,
            estimated_transfer_bytes=transfer_to + transfer_from,
            min_pl_elements=self.min_pl_elements,
            active_app=active_app,
        )


class KV260FabricSession:
    """Resident on-board XRT session for SPF word-array kernels."""

    def __init__(self, xclbin_path: str):
        try:
            import pyxrt as xrt  # type: ignore
        except ImportError:
            import xrt  # type: ignore

        self.xrt = xrt
        self.xclbin_path = str(xclbin_path)
        self.device = xrt.device(0)
        self.uuid = self.device.load_xclbin(self.xclbin_path)
        self._kernels: dict[str, Any] = {}
        self._bo_cache: dict[tuple[int, int], Any] = {}
        self._bo_flag = _xrt_cacheable_bo_flag(xrt)

    def kernel(self, name: str) -> Any:
        if name not in self._kernels:
            self._kernels[name] = self.xrt.kernel(self.device, self.uuid, name)
        return self._kernels[name]

    def dispatch(self, spec: FabricKernelSpec, *inputs: Any) -> FabricDispatchResult:
        np = _numpy()
        if len(inputs) != spec.arity:
            raise ValueError(f"{spec.op} expects {spec.arity} inputs, got {len(inputs)}")
        arrays = [np.ascontiguousarray(value, dtype=np.uint32).reshape(-1) for value in inputs]
        if not arrays:
            raise ValueError(f"{spec.op} needs at least one input array")
        n = int(arrays[0].size)
        if any(int(arr.size) != n for arr in arrays):
            raise ValueError(f"{spec.op} input lengths do not match")
        if spec.fixed_elements is not None and n != spec.fixed_elements:
            raise ValueError(f"{spec.op} expects {spec.fixed_elements} words, got {n}")
        kernel = self.kernel(spec.kernel_name)
        nbytes = n * 4
        started = time.time()
        if spec.call_abi == "binary_u32_n":
            bo_a = self._bo(kernel.group_id(0), nbytes)
            bo_b = self._bo(kernel.group_id(1), nbytes)
            bo_out = self._bo(kernel.group_id(2), nbytes)
            _xrt_write_array(bo_a, arrays[0])
            _xrt_write_array(bo_b, arrays[1])
            _xrt_sync_bo(bo_a, self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes)
            _xrt_sync_bo(bo_b, self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes)
            run = kernel(bo_a, bo_b, bo_out, n)
            run.wait()
            _xrt_sync_bo(bo_out, self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes)
            output = _xrt_read_array(bo_out, n)
        elif spec.call_abi == "unary_u32_n":
            bo_in = self._bo(kernel.group_id(0), nbytes)
            bo_out = self._bo(kernel.group_id(1), nbytes)
            _xrt_write_array(bo_in, arrays[0])
            _xrt_sync_bo(bo_in, self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes)
            run = kernel(bo_in, bo_out, n)
            run.wait()
            _xrt_sync_bo(bo_out, self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes)
            output = _xrt_read_array(bo_out, n)
        elif spec.call_abi == "fft64_u32":
            bo_in = self._bo(kernel.group_id(0), nbytes)
            bo_out = self._bo(kernel.group_id(1), nbytes)
            _xrt_write_array(bo_in, arrays[0])
            _xrt_sync_bo(bo_in, self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes)
            run = kernel(bo_in, bo_out, _scalar_n_arg(spec, n)) if spec.requires_scalar_n else kernel(bo_in, bo_out)
            run.wait()
            _xrt_sync_bo(bo_out, self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes)
            output = _xrt_read_array(bo_out, n)
        else:
            raise ValueError(f"unsupported fabric call ABI {spec.call_abi!r}")
        elapsed = time.time() - started
        return FabricDispatchResult(
            op=spec.op,
            element_count=n,
            kernel_name=spec.kernel_name,
            xclbin_path=self.xclbin_path,
            output_words=output,
            transfer_to_pl_bytes=nbytes * len(arrays),
            transfer_from_pl_bytes=nbytes,
            elapsed_seconds=elapsed,
        )

    def _bo(self, group_id: int, nbytes: int) -> Any:
        key = (int(group_id), int(nbytes))
        if key not in self._bo_cache:
            self._bo_cache[key] = self.xrt.bo(self.device, int(nbytes), self._bo_flag, int(group_id))
        return self._bo_cache[key]


def _scalar_n_arg(spec: FabricKernelSpec, elements: int) -> int:
    """Return the scalar count argument for kernels that process fixed-size blocks."""

    if spec.fixed_elements:
        if int(elements) % int(spec.fixed_elements) != 0:
            raise ValueError(f"{spec.op} expects a multiple of {spec.fixed_elements} words, got {elements}")
        return int(elements) // int(spec.fixed_elements)
    return int(elements)


def specs_from_xclbin(path: str | Path) -> list[FabricKernelSpec]:
    path = Path(path)
    text = _strings_for_path(path)
    metadata = _extract_xclbin_metadata(text)
    discovered: dict[str, tuple[FabricKernelArg, ...]] = {}
    if metadata:
        for kernel in _metadata_kernels(metadata):
            name = str(kernel.get("name") or "")
            if name:
                discovered[name] = _metadata_args(kernel)
    for instance in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*_kernel):[A-Za-z_][A-Za-z0-9_]*_\d+\b", text):
        discovered.setdefault(instance, ())
    specs: list[FabricKernelSpec] = []
    for kernel_name, args in discovered.items():
        catalog = KERNEL_CATALOG.get(kernel_name)
        if not catalog:
            continue
        specs.append(
            FabricKernelSpec(
                op=str(catalog["op"]),
                kernel_name=kernel_name,
                xclbin_path=str(path),
                app_name=path.stem,
                arity=int(catalog["arity"]),
                dtype=str(catalog["dtype"]),
                call_abi=str(catalog["call_abi"]),
                fixed_elements=None if catalog.get("fixed_elements") is None else int(catalog["fixed_elements"]),
                validation=str(catalog.get("validation", "raw_u32")),
                unsafe=bool(catalog.get("unsafe", False)),
                unsafe_reason=None if catalog.get("unsafe_reason") is None else str(catalog["unsafe_reason"]),
                requires_scalar_n=bool(catalog.get("requires_scalar_n", False)),
                args=args,
            )
        )
    return sorted(specs, key=lambda item: (item.op, item.kernel_name))


def fabric_xrt_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "pyxrt_imported": False,
        "device_visible": False,
        "device_error": None,
        "active_app": None,
        "xmutil_available": False,
    }
    try:
        try:
            import pyxrt as xrt  # type: ignore
        except ImportError:
            import xrt  # type: ignore

        report["pyxrt_imported"] = True
        try:
            device = xrt.device(0)
            del device
            report["device_visible"] = True
        except Exception as exc:
            report["device_error"] = str(exc)
    except Exception as exc:
        report["device_error"] = str(exc)
    active = active_xmutil_app()
    report["active_app"] = active
    report["xmutil_available"] = active is not None
    return report


def active_xmutil_app() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["sudo", "-n", "xmutil", "listapps"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if "0," not in line:
            continue
        parts = line.split()
        if parts:
            return parts[0]
    return None


def load_xmutil_app(app_name: str, *, unload_first: bool = True) -> dict[str, Any]:
    commands: list[list[str]] = []
    if unload_first:
        commands.append(["sudo", "-n", "xmutil", "unloadapp"])
    commands.append(["sudo", "-n", "xmutil", "loadapp", app_name])
    results = []
    ok = True
    for cmd in commands:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        results.append(
            {
                "cmd": cmd,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
        ok = ok and proc.returncode == 0
    return {"ok": ok, "app_name": app_name, "steps": results, "active_app": active_xmutil_app()}


def run_fabric_smoke(
    *,
    op: str = "spf32.mul",
    n: int = 4096,
    repeats: int = 1,
    seed: int = 1,
    workdir: Optional[str] = None,
    xclbin_dirs: Optional[Sequence[str]] = None,
    load_app: bool = False,
    libspf32_path: Optional[str] = None,
) -> dict[str, Any]:
    if op != "spf32.mul":
        raise ValueError("fabric-smoke currently supports op=spf32.mul")
    np = _numpy()
    registry = FabricRegistry.discover(workdir=workdir, xclbin_dirs=xclbin_dirs)
    spec = registry.select_kernel(op, active_app=active_xmutil_app())
    if spec is None:
        return {"ok": False, "op": op, "reason": "no_pl_kernel", "registry": registry.to_json()}
    load_report = None
    if load_app:
        load_report = load_xmutil_app(spec.app_name)
        if not load_report["ok"]:
            return {"ok": False, "op": op, "reason": "xmutil_load_failed", "load": load_report}
    lib = _load_libspf32(libspf32_path, workdir=registry.workdir)
    rng = np.random.default_rng(seed)
    a_src = _seeded_complex(rng, n)
    b_src = _seeded_complex(rng, n)
    a_enc = _spf32_encode(lib, a_src)
    b_enc = _spf32_encode(lib, b_src)
    ref = np.empty_like(a_enc)
    lib.spf32_mul_array(
        a_enc.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        b_enc.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ref.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_size_t(n),
    )
    session = KV260FabricSession(spec.xclbin_path)
    results = [session.dispatch(spec, a_enc, b_enc) for _ in range(max(1, int(repeats)))]
    out = results[-1].output_words
    ok = bool(np.array_equal(out, ref))
    mismatch_count = int(np.sum(out != ref)) if not ok else 0
    dispatch_times = [float(result.elapsed_seconds) for result in results]
    return {
        "ok": ok,
        "op": op,
        "n": n,
        "repeats": len(results),
        "seed": seed,
        "kernel": spec.to_json(),
        "load": load_report,
        "dispatch": results[-1].to_json(),
        "dispatch_elapsed_seconds": dispatch_times,
        "dispatch_elapsed_summary": {
            "min": min(dispatch_times),
            "max": max(dispatch_times),
            "mean": sum(dispatch_times) / len(dispatch_times),
        },
        "mismatch_count": mismatch_count,
        "first_outputs": [f"0x{int(word):08x}" for word in out[:4]],
    }


def _default_xclbin_dirs(workdir: str) -> tuple[str, ...]:
    env = os.environ.get("PAN_FABRIC_XCLBIN_DIRS")
    if env:
        return tuple(part for part in env.split(os.pathsep) if part)
    if workdir == DEFAULT_FABRIC_WORKDIR:
        return DEFAULT_FABRIC_XCLBIN_DIRS
    return (str(Path(workdir) / "xclbins"), str(Path(workdir) / "xclbins-pr197"))


def _kernel_rank(spec: FabricKernelSpec) -> int:
    if "core_pl" in spec.app_name or "hotpath" in spec.app_name or "tensor" in spec.app_name:
        return 0
    return 10


def _selection_rank(spec: FabricKernelSpec, *, active_app: Optional[str]) -> tuple[int, int, str]:
    active_rank = 0 if active_app and spec.app_name == active_app else 5
    return active_rank, _kernel_rank(spec), spec.app_name


def _strings_for_path(path: Path) -> str:
    try:
        return subprocess.check_output(["strings", str(path)], text=True, errors="ignore", timeout=20)
    except Exception:
        return path.read_bytes().decode("utf-8", errors="ignore")


def _extract_xclbin_metadata(text: str) -> Optional[dict[str, Any]]:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or "build_metadata" not in stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _metadata_kernels(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    kernels: list[dict[str, Any]] = []
    regions = (
        metadata.get("build_metadata", {})
        .get("xclbin", {})
        .get("user_regions", [])
    )
    for region in regions:
        for kernel in region.get("kernels", []) or []:
            if isinstance(kernel, dict):
                kernels.append(kernel)
    return kernels


def _metadata_args(kernel: dict[str, Any]) -> tuple[FabricKernelArg, ...]:
    args = []
    for item in kernel.get("arguments", []) or []:
        if not isinstance(item, dict):
            continue
        args.append(
            FabricKernelArg(
                name=str(item.get("name") or ""),
                arg_id=int(str(item.get("id") or "0"), 0),
                kind="pointer" if str(item.get("address_qualifier") or "") == "1" else "scalar",
                type_name=str(item.get("type") or ""),
                port=None if item.get("port") is None else str(item.get("port")),
            )
        )
    return tuple(args)


def _xrt_cacheable_bo_flag(xrt: Any) -> Any:
    bo_type = getattr(xrt, "bo")
    if hasattr(bo_type, "cacheable"):
        return bo_type.cacheable
    flags = getattr(bo_type, "flags", None)
    if flags is not None and hasattr(flags, "cacheable"):
        return flags.cacheable
    if flags is not None and hasattr(flags, "normal"):
        return flags.normal
    return 0


def _xrt_sync_bo(bo: Any, direction: Any, size: int) -> None:
    try:
        bo.sync(direction, int(size), 0)
    except TypeError:
        bo.sync(direction)


def _xrt_write_array(bo: Any, array: Any) -> None:
    view = memoryview(bo.map()).cast("B")
    payload = memoryview(array).cast("B")
    view[: len(payload)] = payload


def _xrt_read_array(bo: Any, n_words: int) -> Any:
    np = _numpy()
    raw = memoryview(bo.map()).cast("B")[: n_words * 4]
    return np.frombuffer(raw, dtype=np.uint32, count=n_words).copy()


def _xrt_export_bo(bo: Any) -> dict[str, Any]:
    """Best-effort BO export descriptor for pyxrt/xrt Python bindings.

    XRT Python wheels have exposed this under a few names across versions.
    Returning a structured failure keeps the runtime honest on boards where
    fd export is unavailable.
    """
    for method_name in ("export_buffer", "export_handle", "get_fd"):
        method = getattr(bo, method_name, None)
        if method is None:
            continue
        try:
            handle = method()
        except Exception as exc:
            return {"ok": False, "method": method_name, "error": str(exc)}
        try:
            handle_value = int(handle)
        except Exception:
            handle_value = None
        return {
            "ok": handle_value is not None and handle_value >= 0,
            "method": method_name,
            "handle": handle_value,
        }
    return {"ok": False, "reason": "xrt_bo_export_not_supported"}


def _xrt_import_bo(session: KV260FabricSession, descriptor: dict[str, Any], group_id: int, nbytes: int) -> Any:
    handle = descriptor.get("handle")
    if handle is None:
        raise RuntimeError("BO descriptor has no export handle")
    errors = []
    constructors = (
        (session.device, int(handle)),
        (session.device, int(handle), session._bo_flag, int(group_id)),
        (session.device, int(nbytes), session._bo_flag, int(group_id), int(handle)),
    )
    for args in constructors:
        try:
            return session.xrt.bo(*args)
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("xrt BO import failed: " + "; ".join(errors))


def _numpy() -> Any:
    import numpy as np

    return np


def _find_libspf32(explicit: Optional[str], *, workdir: str) -> str:
    if explicit:
        return explicit
    env = os.environ.get("SPF_LIBSPF32_PATH")
    if env:
        return env
    candidates = (
        Path(workdir) / "lib" / "libspf32.so",
        Path(workdir) / "native" / "libspf32.so",
        Path("/home/ubuntu/spfr/lib/libspf32.so"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("libspf32.so not found; set SPF_LIBSPF32_PATH")


def _load_libspf32(path: Optional[str], *, workdir: str) -> Any:
    lib = ctypes.CDLL(_find_libspf32(path, workdir=workdir))
    lib.spf32_encode_array.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.spf32_encode_array.restype = None
    lib.spf32_mul_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.spf32_mul_array.restype = None
    return lib


def _seeded_complex(rng: Any, n: int) -> Any:
    np = _numpy()
    reals = rng.standard_normal(int(n), dtype=np.float64)
    imags = rng.standard_normal(int(n), dtype=np.float64)
    return reals + 1j * imags


def _spf32_encode(lib: Any, z: Any) -> Any:
    np = _numpy()
    arr = np.ascontiguousarray(z, dtype=np.complex128)
    pairs = arr.view(np.float64).reshape(-1)
    out = np.empty(arr.size, dtype=np.uint32)
    lib.spf32_encode_array(
        pairs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_size_t(arr.size),
    )
    return np.ascontiguousarray(out)
