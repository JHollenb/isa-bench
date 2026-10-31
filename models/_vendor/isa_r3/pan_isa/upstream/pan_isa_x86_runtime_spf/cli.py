"""Command line interface for the standalone SPF PAN x86 runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from pan_hardware_resolver.ledger import write_ledger_jsonl

from .action_ledger import fabric_action_ledger_row
from .fabric_daemon import FabricDaemonGraphSession, fabric_daemon_bench, write_json as write_daemon_json
from .fabric_executor import (
    FabricGraphExecutor,
    _PSOpEngine,
    load_fabric_inputs,
    train_fabric_policy_from_traces,
    write_jsonl as write_fabric_jsonl,
)
from .fabric_foundry import (
    FabricFoundryCatalog,
    catalog_summary,
    generate_candidates,
    ingest_measurements,
    mine_regions,
    record_control_decision,
    scheduler_hint_for_graph,
)
from .fabric_graph import FabricGraph, FabricGraphPlanner, FabricPolicyHead
from .fabric_runtime import (
    FabricRegistry,
    FabricRouter,
    active_xmutil_app,
    fabric_xrt_report,
    run_fabric_smoke,
)
from .gdbstub import GdbRemoteServer
from .model import load_transition_model
from .neural_runtime import train_runtime_neural_model
from .nn_runtime import NNExecutionResult, NNRuntimeSession, run_nn_model, train_nn_policy_from_traces
from .pl_router import backend_availability_report
from .program_compare import compare_program
from .runtime import SPFPrimaryPanRuntime
from .spf_backend import dependency_report


def _program_argv(program: str, rest: Optional[Sequence[str]], argv0: Optional[str] = None) -> list[str]:
    return [argv0 or program, *list(rest or [])]


def _write_or_print(payload: dict[str, Any], path: Optional[str]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _write_jsonl(rows: list[dict[str, Any]], path: Optional[str]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json_if_path(payload: dict[str, Any], path: Optional[str]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _foundry_catalog(root: str | Path, args: argparse.Namespace) -> FabricFoundryCatalog:
    return FabricFoundryCatalog(root, action_ledger_path=getattr(args, "foundry_action_ledger_jsonl", None))


def _load_model_from_args(args: argparse.Namespace) -> Any:
    return load_transition_model(
        getattr(args, "model", None),
        learned_only=bool(getattr(args, "learned_only", False)),
        profile=getattr(args, "profile", "long64-ring3"),
        promotion_ledger_path=getattr(args, "promotion_ledger", None),
    )


def _abi_ids(args: argparse.Namespace) -> dict[str, Optional[int]]:
    uid = args.uid
    euid = args.euid if args.euid is not None else uid
    gid = args.gid
    egid = args.egid if args.egid is not None else gid
    return {"uid": uid, "euid": euid, "gid": gid, "egid": egid}


def _xclbin_dirs_from_args(args: argparse.Namespace) -> Optional[list[str]]:
    values = list(getattr(args, "xclbin_dir", None) or [])
    catalog_root = getattr(args, "foundry_catalog", None)
    if catalog_root:
        values.extend(FabricFoundryCatalog(catalog_root).artifact_xclbin_dirs())
    return None if not values else list(dict.fromkeys(values))


def _backend_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": getattr(args, "backend", "auto"),
        "pl_xclbin": getattr(args, "pl_xclbin", None),
        "pl_kernel_name": getattr(args, "pl_kernel_name", "pan_simd_v1"),
        "pl_min_instructions": getattr(args, "pl_min_instructions", 8),
        "pl_min_groups": getattr(args, "pl_min_groups", 2),
        "fabric_mode": getattr(args, "fabric", "backend"),
        "fabric_daemon_socket": getattr(args, "fabric_daemon", None),
    }


def cmd_run(args: argparse.Namespace) -> int:
    try:
        model = _load_model_from_args(args)
        runtime = SPFPrimaryPanRuntime.from_elf(
            Path(args.program),
            argv=_program_argv(args.program, args.argv, args.argv0),
            **_abi_ids(args),
            model=model,
            profile=args.profile,
            trace_proposals=bool(args.trace_proposals),
            trace_action_ledger=bool(args.write_action_ledger_jsonl),
            **_backend_kwargs(args),
        )
    except ValueError as exc:
        _write_or_print({"ok": False, "program": args.program, "error": str(exc)}, args.write)
        return 1
    if args.gdb:
        GdbRemoteServer.from_endpoint(runtime, args.gdb).serve_once()
    else:
        runtime.run(max_steps=args.max_steps)
    payload = runtime.summary()
    _write_jsonl(runtime.proposal_trace, args.trace_proposals)
    _write_jsonl(runtime.route_trace, args.route_trace)
    _write_jsonl(runtime.fabric_trace, args.fabric_trace)
    _write_jsonl(runtime.fabric_training_records, args.fabric_training_jsonl)
    if args.write_action_ledger_jsonl:
        write_ledger_jsonl(runtime.action_ledger_rows, args.write_action_ledger_jsonl)
    _write_json_if_path(runtime.router.stats, args.backend_stats)
    _write_or_print(payload, args.write)
    return 0 if payload.get("ok") else 1


def cmd_gdb(args: argparse.Namespace) -> int:
    model = _load_model_from_args(args)
    runtime = SPFPrimaryPanRuntime.from_elf(
        Path(args.program),
        argv=_program_argv(args.program, args.argv, args.argv0),
        **_abi_ids(args),
        model=model,
        profile=args.profile,
        trace_proposals=bool(args.trace_proposals),
        **_backend_kwargs(args),
    )
    GdbRemoteServer.from_endpoint(runtime, args.listen).serve_once()
    _write_jsonl(runtime.proposal_trace, args.trace_proposals)
    _write_jsonl(runtime.route_trace, args.route_trace)
    _write_jsonl(runtime.fabric_trace, args.fabric_trace)
    _write_jsonl(runtime.fabric_training_records, args.fabric_training_jsonl)
    _write_json_if_path(runtime.router.stats, args.backend_stats)
    _write_or_print(runtime.summary(), args.write)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    model = _load_model_from_args(args)
    payload = compare_program(
        Path(args.program),
        argv=_program_argv(args.program, args.argv, args.argv0),
        max_steps=args.max_steps,
        **_abi_ids(args),
        model=model,
        profile=args.profile,
        backend=args.backend,
        pl_xclbin=args.pl_xclbin,
        pl_kernel_name=args.pl_kernel_name,
        pl_min_instructions=args.pl_min_instructions,
        pl_min_groups=args.pl_min_groups,
    )
    _write_or_print(payload, args.write)
    return 0 if payload.get("ok") or payload.get("accepted_blocker") else 1


def cmd_validate(args: argparse.Namespace) -> int:
    model = _load_model_from_args(args)
    payload = compare_program(
        Path(args.program),
        argv=_program_argv(args.program, args.argv, args.argv0),
        max_steps=args.max_steps,
        **_abi_ids(args),
        model=model,
        profile=args.profile,
        backend=args.backend,
        pl_xclbin=args.pl_xclbin,
        pl_kernel_name=args.pl_kernel_name,
        pl_min_instructions=args.pl_min_instructions,
        pl_min_groups=args.pl_min_groups,
    )
    payload["suite"] = args.suite
    payload["validation_runtime"] = "pan-isa-x86-runtime-spf"
    _write_or_print(payload, args.write)
    return 0 if payload.get("ok") or payload.get("accepted_blocker") else 1


def cmd_inspect_model(args: argparse.Namespace) -> int:
    payload = dependency_report()
    payload["runtime"] = "pan-isa-x86-runtime-spf"
    payload["requested_model_manifest"] = args.model
    payload["backend_availability"] = backend_availability_report(args.pl_xclbin, kernel_name=args.pl_kernel_name)
    if args.model or args.learned_only:
        try:
            model = _load_model_from_args(args)
            payload["model_manifest"] = model.manifest
        except ValueError as exc:
            payload["ok"] = False
            payload["error"] = str(exc)
    _write_or_print(payload, args.write)
    return 0 if payload.get("ok") else 1


def cmd_train(args: argparse.Namespace) -> int:
    trace_manifest = None
    if args.trace_manifest:
        trace_manifest = json.loads(Path(args.trace_manifest).read_text(encoding="utf-8"))
    model_out = Path(args.model_out)
    payload = train_runtime_neural_model(
        model_out,
        checkpoint_out=Path(args.checkpoint_out) if args.checkpoint_out else None,
        samples_per_form=args.samples_per_form,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        min_confidence=args.min_confidence,
        trace_manifest=trace_manifest,
    )
    payload["model_manifest"] = dependency_report()
    _write_or_print(payload, args.write)
    return 0


def cmd_inspect_fabric(args: argparse.Namespace) -> int:
    registry = FabricRegistry.discover(
        workdir=args.workdir,
        xclbin_dirs=_xclbin_dirs_from_args(args),
    )
    payload = registry.to_json()
    payload["runtime"] = "pan-fabric-runtime-v0"
    payload["xrt"] = fabric_xrt_report() if args.probe_xrt else None
    _write_or_print(payload, args.write)
    return 0


def cmd_fabric_route(args: argparse.Namespace) -> int:
    registry = FabricRegistry.discover(
        workdir=args.workdir,
        xclbin_dirs=_xclbin_dirs_from_args(args),
    )
    active_app = args.active_app
    if active_app == "auto":
        active_app = active_xmutil_app()
    router = FabricRouter(backend=args.fabric_backend, min_pl_elements=args.min_pl_elements)
    decision = router.choose(
        registry,
        op=args.op,
        element_count=args.elements,
        active_app=active_app,
    )
    payload = {
        "runtime": "pan-fabric-runtime-v0",
        "decision": decision.to_json(),
        "registry_summary": {
            "kernel_count": len(registry.kernels),
            "ops": sorted({spec.op for spec in registry.kernels}),
            "errors": registry.errors,
        },
    }
    _write_or_print(payload, args.write)
    return 0 if decision.selected_backend == "pl" or args.fabric_backend != "pl" else 1


def cmd_fabric_smoke(args: argparse.Namespace) -> int:
    try:
        payload = run_fabric_smoke(
            op=args.op,
            n=args.n,
            repeats=args.repeats,
            seed=args.seed,
            workdir=args.workdir,
            xclbin_dirs=_xclbin_dirs_from_args(args),
            load_app=bool(args.load_app),
            libspf32_path=args.libspf32,
        )
    except Exception as exc:
        payload = {
            "ok": False,
            "runtime": "pan-fabric-runtime-v0",
            "op": args.op,
            "error": str(exc),
        }
    payload["runtime"] = "pan-fabric-runtime-v0"
    _write_or_print(payload, args.write)
    return 0 if payload.get("ok") else 1


def cmd_fabric_plan(args: argparse.Namespace) -> int:
    try:
        graph, policy, registry, active_app, plan = _fabric_plan_from_args(args)
        payload = plan.to_json()
        payload["runtime"] = "pan-fabric-runtime-v0"
        payload["active_app"] = active_app
        payload["registry_summary"] = {
            "kernel_count": len(registry.kernels),
            "ops": sorted({spec.op for spec in registry.kernels}),
            "errors": registry.errors,
        }
        _attach_foundry_hint(payload, graph, args)
    except Exception as exc:
        payload = {
            "ok": False,
            "runtime": "pan-fabric-runtime-v0",
            "error": str(exc),
        }
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0


def cmd_fabric_exec(args: argparse.Namespace) -> int:
    try:
        graph, policy, registry, active_app, plan = _fabric_plan_from_args(args)
        scheduler_hint = _foundry_hint_for_graph(graph, args)
        executor = FabricGraphExecutor(
            plan,
            execution_backend=args.execute_backend,
            seed=args.seed,
            workdir=args.workdir,
            libspf32_path=args.libspf32,
            libspf16_path=args.libspf16,
            load_app=bool(args.load_app),
            initial_inputs=load_fabric_inputs(args.inputs),
            daemon_socket=args.fabric_daemon,
            daemon_buffer_storage=args.daemon_buffer_storage,
            materialize_outputs=not bool(args.no_output_digest),
            enable_bo_import=bool(args.enable_bo_import),
            allow_xrt_app_switch=bool(args.allow_xrt_app_switch),
            scheduler_hint=scheduler_hint,
        )
        result = executor.execute(repeats=args.repeats)
        write_fabric_jsonl(result.trace, args.trace_jsonl)
        write_fabric_jsonl(result.training_records, args.training_jsonl)
        payload = result.to_json(include_trace=bool(args.include_trace))
        payload["runtime"] = "pan-fabric-runtime-v0"
        payload["active_app"] = active_app
        payload["policy"] = policy.to_json()
        payload["registry_summary"] = {
            "kernel_count": len(registry.kernels),
            "ops": sorted({spec.op for spec in registry.kernels}),
            "errors": registry.errors,
        }
        if args.execute_backend == "fabric-session":
            _attach_session_foundry(payload, graph, args)
        else:
            _attach_foundry_hint(payload, graph, args)
        _attach_control_record(payload, graph, args)
        if args.write_action_ledger_jsonl:
            write_ledger_jsonl(
                [
                    fabric_action_ledger_row(
                        {
                            "active_app": active_app,
                            "execution_backend": args.execute_backend,
                            "fabric_backend": args.fabric_backend,
                        },
                        {
                            "kind": "execute_fabric_graph",
                            "graph": graph.to_json(),
                            "graph_outputs": list(graph.outputs),
                        },
                        {
                            "target": args.foundry_target,
                            "execute_backend": args.execute_backend,
                        },
                        plan,
                        measurement={
                            "environment": {
                                "runtime": "pan-fabric-runtime-v0",
                                "active_app": active_app,
                                "execution_backend": args.execute_backend,
                            },
                            "metrics": result.summary(),
                        },
                        lineage={"command": "fabric-exec"},
                    )
                ],
                args.write_action_ledger_jsonl,
            )
    except Exception as exc:
        payload = {
            "ok": False,
            "runtime": "pan-fabric-runtime-v0",
            "error": str(exc),
        }
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0 if payload.get("ok") else 1


def cmd_fabric_control(args: argparse.Namespace) -> int:
    try:
        catalog = _foundry_catalog(args.catalog, args)
        payload = catalog_summary(catalog)
        payload["ok"] = True
        payload["runtime"] = "pan-fabric-runtime-v0"
        payload["schema"] = "pan_fabric_control_plane_status_v1"
        if args.graph or args.linear_ops:
            graph = _fabric_graph_from_source_args(args)
            payload["graph"] = graph.to_json()
            payload["scheduler_hint"] = scheduler_hint_for_graph(graph, catalog=catalog, target=args.target)
    except Exception as exc:
        payload = {
            "ok": False,
            "runtime": "pan-fabric-runtime-v0",
            "schema": "pan_fabric_control_plane_status_v1",
            "error": str(exc),
        }
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0


def cmd_fabric_train_policy(args: argparse.Namespace) -> int:
    try:
        policy = train_fabric_policy_from_traces(args.trace_jsonl, l2=args.l2)
        payload = {
            "ok": True,
            "runtime": "pan-fabric-runtime-v0",
            "policy": policy.to_json(),
        }
        if args.policy_out:
            out = Path(args.policy_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(policy.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        payload = {
            "ok": False,
            "runtime": "pan-fabric-runtime-v0",
            "error": str(exc),
        }
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0


def cmd_fabric_promote_policy(args: argparse.Namespace) -> int:
    try:
        policy = FabricPolicyHead.from_path(args.policy)
        metrics = json.loads(Path(args.heldout_metrics).read_text(encoding="utf-8"))
        promoted = policy.with_promotion_metrics(
            metrics,
            min_samples=args.min_samples,
            max_latency_ratio=args.max_latency_ratio,
            max_transfer_ratio=args.max_transfer_ratio,
            max_energy_ratio=args.max_energy_ratio,
            min_compile_priority_precision=args.min_compile_priority_precision,
        )
        payload = {
            "ok": promoted.can_override_rule_scheduler(),
            "runtime": "pan-fabric-runtime-v0",
            "schema": "pan_fabric_policy_promotion_v1",
            "promotion_evaluation": promoted.heldout_metrics.get("promotion_evaluation"),
            "policy": promoted.to_json(),
        }
        if args.policy_out:
            out = Path(args.policy_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(promoted.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        payload = {
            "ok": False,
            "runtime": "pan-fabric-runtime-v0",
            "schema": "pan_fabric_policy_promotion_v1",
            "error": str(exc),
        }
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0 if payload.get("ok") else 1


def cmd_nn_run(args: argparse.Namespace) -> int:
    try:
        inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8")) if args.inputs else None
        policy = FabricPolicyHead.from_path(args.policy) if args.policy else None
        backend = args.fabric if args.fabric is not None else args.backend
        feedback_map = _parse_nn_feedback_outputs(args.feedback_output)
        repeats = int(args.repeats)
        if repeats < 1:
            raise ValueError("--repeats must be at least 1")
        promotion_warmup_runs = int(args.promotion_warmup_runs)
        if promotion_warmup_runs < 0:
            raise ValueError("--promotion-warmup-runs must be non-negative")
        if bool(args.resident_inputs) and repeats > 1 and promotion_warmup_runs >= repeats:
            raise ValueError("--promotion-warmup-runs must be smaller than --repeats for resident-input promotion windows")
        run_kwargs = {
            "backend": backend,
            "inputs": inputs,
            "seed": args.seed,
            "no_output_digest": bool(args.no_output_digest),
            "validate_output": bool(args.validate_output),
            "validation_rtol": args.validation_rtol,
            "validation_atol": args.validation_atol,
            "validation_sample_count": args.validation_sample_count,
            "foundry_auto_queue": bool(args.foundry_auto_queue),
            "foundry_catalog": args.foundry_catalog,
            "foundry_action_ledger_path": args.foundry_action_ledger_jsonl,
            "foundry_target": args.target,
            "fabric_daemon": args.fabric_daemon,
            "workdir": args.workdir,
            "xclbin_dirs": args.xclbin_dir,
            "active_app": args.active_app,
            "fabric_backend": args.fabric_backend,
            "daemon_buffer_storage": args.daemon_buffer_storage,
            "policy": policy,
            "policy_materialization": bool(args.policy_materialization),
        }
        if repeats > 1 or args.session_id or args.resident_inputs or feedback_map:
            session = NNRuntimeSession.from_model(
                args.model,
                resident_weights=bool(args.resident_weights),
                resident_inputs=bool(args.resident_inputs),
                seed=args.seed,
                session_id=args.session_id,
            )
            try:
                results: list[NNExecutionResult] = []
                all_trace: list[dict[str, Any]] = []
                all_training: list[dict[str, Any]] = []
                for run_index in range(repeats):
                    result = session.run(**run_kwargs, feedback_map=feedback_map)
                    results.append(result)
                    all_trace.extend(_tag_nn_rows(result.trace, run_index=run_index, session_id=session.value_store.session_id))
                    all_training.extend(
                        _tag_nn_rows(result.training_records, run_index=run_index, session_id=session.value_store.session_id)
                    )
                    if not result.ok:
                        break
                last = results[-1]
                payload = last.to_json(include_trace=False)
                payload["ok"] = all(result.ok for result in results)
                steady_state = bool((args.resident_inputs or feedback_map) and len(results) > 1)
                payload["summary"] = _aggregate_nn_summary(
                    results,
                    steady_state=steady_state,
                    promotion_warmup_runs=promotion_warmup_runs,
                )
                payload["value_store"] = session.value_store.to_json()
                payload["measurement_scope"] = _nn_measurement_scope(
                    steady_state=steady_state,
                    requested_repeats=repeats,
                    completed_runs=len(results),
                    resident_inputs=bool(args.resident_inputs),
                    state_feedback=bool(feedback_map),
                    promotion_warmup_runs=promotion_warmup_runs,
                )
                payload["session"] = {
                    "schema": "pan_nn_runtime_session_v1",
                    "session_id": session.value_store.session_id,
                    "requested_repeats": repeats,
                    "completed_runs": len(results),
                    "run_count": session.run_count,
                    "resident_weights": bool(args.resident_weights),
                    "resident_inputs": bool(args.resident_inputs),
                    "state_feedback": bool(feedback_map),
                    "feedback_output_map": dict(sorted(feedback_map.items())),
                    "promotion_warmup_runs": promotion_warmup_runs,
                    "retained_value_ids": {
                        name: value.value_id
                        for name, value in sorted(session.retained_values.items())
                    },
                    "retained_input_value_ids": {
                        name: value.value_id
                        for name, value in sorted(session.retained_input_values.items())
                    },
                    "retained_feedback_value_ids": {
                        name: value.value_id
                        for name, value in sorted(session.retained_feedback_values.items())
                    },
                    "retained_values": {
                        name: value.to_json()
                        for name, value in sorted(session.retained_values.items())
                    },
                    "retained_input_values": {
                        name: value.to_json()
                        for name, value in sorted(session.retained_input_values.items())
                    },
                    "retained_feedback_values": {
                        name: value.to_json()
                        for name, value in sorted(session.retained_feedback_values.items())
                    },
                    "runs": [
                        {
                            "run_index": index,
                            "ok": result.ok,
                            "elapsed_seconds": result.elapsed_seconds,
                            "summary": result.summary(),
                            "validation": result.validation,
                            "error": result.error,
                        }
                        for index, result in enumerate(results)
                    ],
                }
                if bool(args.include_trace):
                    payload["trace"] = all_trace
                    payload["training_records"] = all_training
                result = last
                trace_rows = all_trace
                training_rows = all_training
            finally:
                session.close()
        else:
            result = run_nn_model(
                args.model,
                resident_weights=bool(args.resident_weights),
                **run_kwargs,
            )
            payload = result.to_json(include_trace=bool(args.include_trace))
            payload["measurement_scope"] = _nn_measurement_scope(
                steady_state=False,
                requested_repeats=1,
                completed_runs=1,
                resident_inputs=False,
                state_feedback=False,
                promotion_warmup_runs=0,
            )
            trace_rows = list(result.trace)
            training_rows = list(result.training_records)
        payload["runtime"] = "pan-nn-runtime-v0"
        payload["fabric_daemon"] = args.fabric_daemon
        if policy is not None:
            payload["policy"] = {
                "policy_kind": policy.policy_kind,
                "policy_role": policy.policy_role,
                "promotion_status": policy.promotion_status,
                "can_override_rule_scheduler": policy.can_override_rule_scheduler(),
            }
        if args.trace_jsonl:
            _write_jsonl(trace_rows, args.trace_jsonl)
        if args.training_jsonl:
            _write_jsonl(training_rows, args.training_jsonl)
    except Exception as exc:
        payload = {
            "ok": False,
            "runtime": "pan-nn-runtime-v0",
            "schema": "pan_nn_execution_v1",
            "error": str(exc),
        }
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0 if payload.get("ok") else 1


def _tag_nn_rows(rows: Sequence[dict[str, Any]], *, run_index: int, session_id: str) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        item = dict(row)
        item["session_id"] = session_id
        item["session_run_index"] = int(run_index)
        tagged.append(item)
    return tagged


def _parse_nn_feedback_outputs(items: Optional[Sequence[str]]) -> dict[str, str]:
    feedback: dict[str, str] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError("--feedback-output entries must use INPUT=OUTPUT")
        input_name, output_name = (part.strip() for part in item.split("=", 1))
        if not input_name or not output_name:
            raise ValueError("--feedback-output entries must use non-empty INPUT=OUTPUT names")
        if input_name in feedback and feedback[input_name] != output_name:
            raise ValueError(f"duplicate feedback input {input_name!r}")
        feedback[input_name] = output_name
    return dict(sorted(feedback.items()))


def _aggregate_nn_summary(
    results: Sequence[NNExecutionResult],
    *,
    steady_state: bool = False,
    promotion_warmup_runs: int = 1,
) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "completed_runs": len(results),
        "failed_runs": sum(1 for result in results if not result.ok),
        "elapsed_seconds": sum(float(result.elapsed_seconds) for result in results),
    }
    for result in results:
        for key, value in result.summary().items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    if steady_state:
        warmup_runs = max(0, int(promotion_warmup_runs))
        steady_results = list(results[warmup_runs:])
        steady_elapsed = sum(float(result.elapsed_seconds) for result in steady_results)
        steady_run_count = len(steady_results)
        steady_mean_elapsed = steady_elapsed / steady_run_count if steady_run_count else 0.0
        steady_summary: dict[str, Any] = {
            "steady_state_completed_runs": steady_run_count,
            "steady_state_warmup_runs": warmup_runs,
            "steady_state_elapsed_seconds": steady_elapsed,
            "steady_state_mean_elapsed_seconds": steady_mean_elapsed,
        }
        for result in steady_results:
            for key, value in result.summary().items():
                if isinstance(value, (int, float)):
                    steady_key = f"steady_state_{key}"
                    steady_summary[steady_key] = steady_summary.get(steady_key, 0) + value
        steady_transfer = (
            int(steady_summary.get("steady_state_timed_transfer_bytes") or 0)
            + int(steady_summary.get("steady_state_setup_transfer_bytes") or 0)
            + int(steady_summary.get("steady_state_forced_materialization_bytes") or 0)
        )
        totals.update(steady_summary)
        totals["promotion_elapsed_seconds"] = steady_mean_elapsed
        totals["promotion_transfer_bytes"] = steady_transfer
        totals["promotion_materialization_bytes"] = int(
            steady_summary.get("steady_state_forced_materialization_bytes") or 0
        )
    return totals


def _nn_measurement_scope(
    *,
    steady_state: bool,
    requested_repeats: int,
    completed_runs: int,
    resident_inputs: bool,
    promotion_warmup_runs: int,
    state_feedback: bool = False,
) -> dict[str, Any]:
    warmup_runs = max(0, int(promotion_warmup_runs)) if steady_state else 0
    if steady_state:
        if state_feedback:
            scope = "hot_cache_state_feedback" if warmup_runs > 1 else "steady_state_state_feedback"
        else:
            scope = "hot_cache_resident_inputs" if warmup_runs > 1 else "steady_state_resident_inputs"
    else:
        scope = "full_execution"
    return {
        "schema": "pan_nn_measurement_scope_v1",
        "scope": scope,
        "resident_inputs": bool(resident_inputs),
        "state_feedback": bool(state_feedback),
        "requested_repeats": int(requested_repeats),
        "completed_runs": int(completed_runs),
        "warmup_runs_excluded": warmup_runs,
        "promotion_window_start_run_index": warmup_runs,
        "promotion_window_completed_runs": max(0, int(completed_runs) - warmup_runs) if steady_state else int(completed_runs),
    }


def cmd_nn_train_policy(args: argparse.Namespace) -> int:
    try:
        policy = train_nn_policy_from_traces(args.trace_jsonl, l2=args.l2)
        payload = {
            "ok": True,
            "runtime": "pan-nn-runtime-v0",
            "schema": "pan_nn_policy_training_v1",
            "policy": policy.to_json(),
        }
        if args.policy_out:
            out = Path(args.policy_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(policy.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        payload = {
            "ok": False,
            "runtime": "pan-nn-runtime-v0",
            "schema": "pan_nn_policy_training_v1",
            "error": str(exc),
        }
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0


def cmd_fabric_values(args: argparse.Namespace) -> int:
    try:
        payload: dict[str, Any] = {
            "ok": True,
            "runtime": "pan-fabric-runtime-v0",
            "schema": "pan_fabric_values_inspect_v1",
        }
        if args.execution_json:
            execution = json.loads(Path(args.execution_json).read_text(encoding="utf-8"))
            payload["execution_json"] = args.execution_json
            payload["execution_ok"] = execution.get("ok")
            payload["summary"] = execution.get("summary")
            payload["value_store"] = execution.get("value_store") or {}
            payload["fabric_values"] = execution.get("fabric_values") or {}
        if args.fabric_daemon:
            with FabricDaemonGraphSession(args.fabric_daemon, timeout=args.timeout) as session:
                if args.release_handle is not None:
                    result = session.release_value(int(args.release_handle))
                elif args.retain_handle is not None:
                    result = session.retain_value(int(args.retain_handle))
                elif args.materialize_handle is not None:
                    result = session.materialize_value(int(args.materialize_handle), elements=int(args.elements or 0))
                elif args.export_handle is not None:
                    result = session.export_value(int(args.export_handle))
                elif args.describe_handle is not None:
                    result = session.describe_value(int(args.describe_handle))
                else:
                    result = session.query_values()
            payload["daemon"] = result.to_json()
        if not args.execution_json and not args.fabric_daemon:
            raise ValueError("fabric-values requires --execution-json and/or --fabric-daemon")
    except Exception as exc:
        payload = {"ok": False, "runtime": "pan-fabric-runtime-v0", "schema": "pan_fabric_values_inspect_v1", "error": str(exc)}
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0 if payload.get("ok") else 1


def cmd_fabric_daemon_bench(args: argparse.Namespace) -> int:
    try:
        result = fabric_daemon_bench(
            args.socket,
            n_words=args.n,
            reps=args.reps,
            warmup=args.warmup,
            seed_a=args.seed_a,
            seed_b=args.seed_b,
            timeout=args.timeout,
        )
        payload = result.to_json()
        payload["runtime"] = "pan-fabric-runtime-v0"
    except Exception as exc:
        payload = {
            "ok": False,
            "runtime": "pan-fabric-runtime-v0",
            "error": str(exc),
        }
        _write_or_print(payload, args.write)
        return 1
    write_daemon_json(payload, args.write)
    return 0 if payload.get("ok") else 1


def cmd_fabric_daemon_chain_bench(args: argparse.Namespace) -> int:
    try:
        np = _import_numpy()
        graph = FabricGraph.linear(
            ["spf32.mul"],
            elements=args.n,
            input_residency="pl",
            output_residency="pl",
        )
        registry = FabricRegistry.discover(
            workdir=args.workdir,
            xclbin_dirs=_xclbin_dirs_from_args(args),
        )
        active_app = args.active_app
        if active_app == "auto":
            active_app = active_xmutil_app()
        plan = FabricGraphPlanner(
            registry,
            backend="pl",
            active_app=active_app,
        ).plan(graph)
        _validate_daemon_chain_plan(plan)

        rng = np.random.default_rng(int(args.seed))
        a = rng.integers(0, 2**32, size=int(args.n), dtype=np.uint32)
        b = rng.integers(0, 2**32, size=int(args.n), dtype=np.uint32)
        ones = np.ones(int(args.n), dtype=np.uint32)

        setup_results = []
        warmup_results = []
        timed_results = []
        final_result = None
        with FabricDaemonGraphSession(args.socket, timeout=args.timeout) as session:
            setup_started = time.perf_counter()
            ones_result = _daemon_mul_graph(session, plan, ones, ones, materialize_outputs=False)
            ones_handle = _single_daemon_output(ones_result)
            a_result = _daemon_mul_graph(session, plan, a, ones_handle, materialize_outputs=False)
            a_handle = _single_daemon_output(a_result)
            b_result = _daemon_mul_graph(session, plan, b, ones_handle, materialize_outputs=False)
            b_handle = _single_daemon_output(b_result)
            setup_elapsed = time.perf_counter() - setup_started
            setup_results.extend([ones_result, a_result, b_result])

            h = a_handle
            for _ in range(int(args.warmup)):
                result = _daemon_mul_graph(session, plan, h, b_handle, materialize_outputs=False)
                warmup_results.append(result)
                h = _single_daemon_output(result)

            timed_started = time.perf_counter()
            for _ in range(int(args.reps)):
                result = _daemon_mul_graph(session, plan, h, b_handle, materialize_outputs=False)
                timed_results.append(result)
                h = _single_daemon_output(result)
            timed_elapsed = time.perf_counter() - timed_started

            final_sha = None
            expected_sha = None
            sha_matches_expected = None
            if args.materialize_final or args.validate_final:
                final_result = _daemon_mul_graph(session, plan, h, ones_handle, materialize_outputs=True)
                final_array = _single_materialized_output(final_result)
                final_sha = _sha256_u32(final_array)
                if args.validate_final:
                    ps = _PSOpEngine(workdir=args.workdir, libspf32_path=args.libspf32)
                    expected_ones = ps.compute("spf32.mul", [ones, ones], int(args.n))
                    expected = ps.compute("spf32.mul", [a, expected_ones], int(args.n))
                    expected_b = ps.compute("spf32.mul", [b, expected_ones], int(args.n))
                    for _ in range(int(args.warmup) + int(args.reps)):
                        expected = ps.compute("spf32.mul", [expected, expected_b], int(args.n))
                    expected = ps.compute("spf32.mul", [expected, expected_ones], int(args.n))
                    expected_sha = _sha256_u32(expected)
                    sha_matches_expected = final_sha == expected_sha

        timed_daemon_elapsed = sum(result.daemon_elapsed_seconds for result in timed_results)
        timed_roundtrip_elapsed = sum(result.roundtrip_seconds for result in timed_results)
        timed_input_bytes = sum(result.input_bytes for result in timed_results)
        timed_output_bytes = sum(result.output_bytes for result in timed_results)
        setup_transfer_bytes = sum(result.input_bytes + result.output_bytes for result in setup_results)
        warmup_transfer_bytes = sum(result.input_bytes + result.output_bytes for result in warmup_results)
        final_transfer_bytes = (final_result.input_bytes + final_result.output_bytes) if final_result else 0
        words = int(args.n) * int(args.reps)
        payload = {
            "ok": True,
            "runtime": "pan-fabric-runtime-v0",
            "schema": "pan_fabric_daemon_chain_bench_v1",
            "socket_path": args.socket,
            "active_app": active_app,
            "target": getattr(args, "foundry_target", "kv260"),
            "ops": ["spf32.mul"],
            "elements": int(args.n),
            "reps": int(args.reps),
            "warmup": int(args.warmup),
            "seed": int(args.seed),
            "setup_graph_packets": len(setup_results),
            "timed_graph_packets": len(timed_results),
            "setup_elapsed_seconds": setup_elapsed,
            "timed_elapsed_seconds": timed_elapsed,
            "timed_roundtrip_seconds": timed_roundtrip_elapsed,
            "timed_daemon_elapsed_seconds": timed_daemon_elapsed,
            "timed_mwords_per_s": (words / timed_elapsed / 1e6) if timed_elapsed else 0.0,
            "timed_daemon_mwords_per_s": (words / timed_daemon_elapsed / 1e6) if timed_daemon_elapsed else 0.0,
            "timed_input_bytes": timed_input_bytes,
            "timed_output_bytes": timed_output_bytes,
            "timed_transfer_bytes": timed_input_bytes + timed_output_bytes,
            "setup_transfer_bytes": setup_transfer_bytes,
            "warmup_transfer_bytes": warmup_transfer_bytes,
            "final_transfer_bytes": final_transfer_bytes,
            "final_output": _single_output_json(final_result) if final_result else None,
            "final_sha256": final_sha,
            "expected_sha256": expected_sha,
            "sha_matches_expected": sha_matches_expected,
            "final_handle": h,
        }
        _attach_chain_foundry(payload, graph, args)
    except Exception as exc:
        payload = {
            "ok": False,
            "runtime": "pan-fabric-runtime-v0",
            "schema": "pan_fabric_daemon_chain_bench_v1",
            "error": str(exc),
        }
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0 if payload.get("ok") else 1


def _import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy is required for fabric daemon chain benchmarking") from exc
    return np


def _validate_daemon_chain_plan(plan: Any) -> None:
    run_pl = [step for step in plan.steps if step.action == "run_pl"]
    if len(run_pl) != 1 or len(plan.graph.ops) != 1:
        raise ValueError("daemon chain benchmark expects exactly one PL graph op per packet")
    step = run_pl[0]
    if step.op != "spf32.mul" or step.kernel is None or step.kernel.call_abi != "binary_u32_n":
        raise ValueError("daemon chain benchmark currently supports only spf32.mul binary_u32_n")


def _daemon_mul_graph(
    session: FabricDaemonGraphSession,
    plan: Any,
    left: Any,
    right: Any,
    *,
    materialize_outputs: bool,
) -> Any:
    result = session.execute_graph(
        plan,
        {"input0": left, "input1": right},
        output_names=plan.graph.outputs,
        materialize_outputs=materialize_outputs,
    )
    if not result.ok:
        raise RuntimeError(f"fabric daemon graph packet failed: {result.error or result.status}")
    return result


def _single_daemon_output(result: Any) -> dict[str, Any]:
    value = next(iter(result.outputs.values()))
    if not (isinstance(value, dict) and value.get("schema") == "pan_daemon_bo_v1"):
        raise RuntimeError("expected daemon-resident BO output")
    return value


def _single_materialized_output(result: Any) -> Any:
    value = next(iter(result.outputs.values()))
    if isinstance(value, dict) and value.get("schema") == "pan_daemon_bo_v1":
        raise RuntimeError("expected materialized daemon output")
    return value


def _single_output_json(result: Any) -> Optional[dict[str, Any]]:
    if result is None:
        return None
    value = next(iter(result.to_json()["outputs"].values()))
    return value if isinstance(value, dict) else None


def _sha256_u32(array: Any) -> str:
    np = _import_numpy()
    arr = np.ascontiguousarray(array, dtype=np.uint32).reshape(-1)
    return hashlib.sha256(memoryview(arr).cast("B")).hexdigest()


def cmd_fabric_foundry_mine(args: argparse.Namespace) -> int:
    paths = list(args.execution_json or []) + list(args.trace_jsonl or [])
    try:
        regions = mine_regions(paths, max_ops=args.max_ops, min_count=args.min_count)
        catalog = _foundry_catalog(args.catalog, args)
        for region in regions:
            catalog.put_region(region)
        payload = {
            "ok": True,
            "runtime": "pan-fabric-runtime-v0",
            "catalog": str(catalog.root),
            "regions": [region.to_json() for region in regions],
            "region_count": len(regions),
        }
    except Exception as exc:
        payload = {"ok": False, "runtime": "pan-fabric-runtime-v0", "error": str(exc)}
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0


def cmd_fabric_foundry_candidates(args: argparse.Namespace) -> int:
    try:
        catalog = _foundry_catalog(args.catalog, args)
        regions = catalog.load_regions()
        candidates = generate_candidates(regions, target=args.target)
        queued = []
        for candidate in candidates:
            catalog.put_candidate(candidate)
            catalog.put_core_manifest(candidate, artifact_status="queued" if args.queue else "candidate")
            if args.queue and candidate.priority >= args.min_priority:
                queued.append(catalog.append_queue(candidate).to_json())
        payload = {
            "ok": True,
            "runtime": "pan-fabric-runtime-v0",
            "catalog": str(catalog.root),
            "candidate_count": len(candidates),
            "queued_count": len(queued),
            "candidates": [candidate.to_json() for candidate in candidates],
            "queued": queued,
        }
    except Exception as exc:
        payload = {"ok": False, "runtime": "pan-fabric-runtime-v0", "error": str(exc)}
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0


def cmd_fabric_foundry_ingest(args: argparse.Namespace) -> int:
    try:
        catalog = _foundry_catalog(args.catalog, args)
        measurements = ingest_measurements(
            args.measurement,
            catalog=catalog,
            candidate_id=args.candidate_id,
            artifact_path=args.artifact,
        )
        payload = {
            "ok": True,
            "runtime": "pan-fabric-runtime-v0",
            "catalog": str(catalog.root),
            "measurements": measurements,
        }
    except Exception as exc:
        payload = {"ok": False, "runtime": "pan-fabric-runtime-v0", "error": str(exc)}
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0


def cmd_fabric_foundry_schedule(args: argparse.Namespace) -> int:
    try:
        graph = _fabric_graph_from_source_args(args)
        catalog = _foundry_catalog(args.catalog, args)
        hint = scheduler_hint_for_graph(graph, catalog=catalog, target=args.target)
        queued = None
        if args.auto_queue and hint.get("compile_better_for_next_time") and hint.get("candidate"):
            candidate = generate_candidates(
                [mine_region_from_graph(graph)],
                target=args.target,
            )[0]
            catalog.put_region(mine_region_from_graph(graph))
            catalog.put_candidate(candidate)
            catalog.put_core_manifest(candidate, artifact_status="queued")
            queued = catalog.append_queue(candidate).to_json()
            hint = scheduler_hint_for_graph(graph, catalog=catalog, target=args.target)
        payload = {
            "ok": True,
            "runtime": "pan-fabric-runtime-v0",
            "catalog": str(catalog.root),
            "hint": hint,
            "queued": queued,
        }
    except Exception as exc:
        payload = {"ok": False, "runtime": "pan-fabric-runtime-v0", "error": str(exc)}
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0


def cmd_fabric_foundry_status(args: argparse.Namespace) -> int:
    try:
        payload = catalog_summary(_foundry_catalog(args.catalog, args))
        payload["ok"] = True
        payload["runtime"] = "pan-fabric-runtime-v0"
    except Exception as exc:
        payload = {"ok": False, "runtime": "pan-fabric-runtime-v0", "error": str(exc)}
        _write_or_print(payload, args.write)
        return 1
    _write_or_print(payload, args.write)
    return 0


def _attach_control_record(payload: dict[str, Any], graph: FabricGraph, args: argparse.Namespace) -> None:
    catalog_root = getattr(args, "foundry_catalog", None)
    if not catalog_root:
        return
    catalog = _foundry_catalog(catalog_root, args)
    target = getattr(args, "foundry_target", "kv260")
    payload["control_plane"] = record_control_decision(
        catalog,
        graph,
        payload=payload,
        hint=payload.get("foundry_hint"),
        target=target,
        source_path=f"fabric-exec:{getattr(args, 'execute_backend', 'auto')}",
    )


def _fabric_plan_from_args(
    args: argparse.Namespace,
) -> tuple[FabricGraph, FabricPolicyHead, FabricRegistry, Optional[str], Any]:
    graph = _fabric_graph_from_source_args(args)
    policy = FabricPolicyHead.from_path(args.policy) if args.policy else FabricPolicyHead.default()
    registry = FabricRegistry.discover(
        workdir=args.workdir,
        xclbin_dirs=_xclbin_dirs_from_args(args),
    )
    active_app = args.active_app
    if active_app == "auto":
        active_app = active_xmutil_app()
    planner = FabricGraphPlanner(
        registry,
        policy=policy,
        backend=args.fabric_backend,
        active_app=active_app,
    )
    return graph, policy, registry, active_app, planner.plan(graph)


def _fabric_graph_from_source_args(args: argparse.Namespace) -> FabricGraph:
    return (
        FabricGraph.from_path(args.graph)
        if args.graph
        else FabricGraph.linear(
            [part.strip() for part in str(args.linear_ops or "").split(",") if part.strip()],
            elements=args.elements,
            input_residency=getattr(args, "input_residency", "ps"),
            output_residency=getattr(args, "output_residency", "ps"),
        )
    )


def mine_region_from_graph(graph: FabricGraph) -> Any:
    from .fabric_foundry import FabricRegion

    ops = tuple(op.op for op in graph.ops)
    elements = int(graph.ops[0].elements) if graph.ops else 0
    seed = json.dumps({"ops": ops, "elements": elements}, sort_keys=True)
    import hashlib

    return FabricRegion(
        region_id="region_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16],
        ops=ops,
        elements=elements,
        graph=graph,
    )


def _foundry_hint_for_graph(graph: FabricGraph, args: argparse.Namespace) -> dict[str, Any]:
    catalog_root = getattr(args, "foundry_catalog", None)
    if not catalog_root:
        return {}
    catalog = _foundry_catalog(catalog_root, args)
    return scheduler_hint_for_graph(graph, catalog=catalog, target=getattr(args, "foundry_target", "kv260"))


def _attach_foundry_hint(payload: dict[str, Any], graph: FabricGraph, args: argparse.Namespace) -> None:
    catalog_root = getattr(args, "foundry_catalog", None)
    if not catalog_root:
        return
    catalog = _foundry_catalog(catalog_root, args)
    hint = scheduler_hint_for_graph(graph, catalog=catalog, target=getattr(args, "foundry_target", "kv260"))
    queued = None
    if getattr(args, "foundry_auto_queue", False) and hint.get("compile_better_for_next_time") and hint.get("candidate"):
        region = mine_region_from_graph(graph)
        candidate = generate_candidates([region], target=getattr(args, "foundry_target", "kv260"))[0]
        catalog.put_region(region)
        catalog.put_candidate(candidate)
        catalog.put_core_manifest(candidate, artifact_status="queued")
        queued = catalog.append_queue(candidate).to_json()
        hint = scheduler_hint_for_graph(graph, catalog=catalog, target=getattr(args, "foundry_target", "kv260"))
    payload["foundry_hint"] = hint
    payload["foundry_queued"] = queued


def _attach_session_foundry(payload: dict[str, Any], graph: FabricGraph, args: argparse.Namespace) -> None:
    catalog_root = getattr(args, "foundry_catalog", None)
    if not catalog_root:
        return
    catalog = _foundry_catalog(catalog_root, args)
    target = getattr(args, "foundry_target", "kv260")
    region = mine_region_from_graph(graph)
    from .fabric_foundry import FabricRegion

    summary = dict(payload.get("summary") or {})
    region = FabricRegion(
        region_id=region.region_id,
        ops=region.ops,
        elements=region.elements,
        occurrence_count=int(summary.get("timed_graph_packets") or payload.get("repeats") or 1),
        observed_seconds=float(summary.get("timed_elapsed_seconds") or 0.0),
        observed_transfer_bytes=int(summary.get("timed_transfer_bytes") or 0),
        source_paths=("fabric-exec:fabric-session",),
        graph=graph,
    )
    candidates = generate_candidates([region], target=target)
    candidate = candidates[0] if candidates else None
    queued = None
    if candidate is not None:
        catalog.put_region(region)
        catalog.put_candidate(candidate)
        artifact_status = "queued" if getattr(args, "foundry_auto_queue", False) and candidate.priority > 0.0 else "candidate"
        catalog.put_core_manifest(candidate, artifact_status=artifact_status)
        if artifact_status == "queued":
            queued = catalog.append_queue(candidate, reason="fabric_session_measurement_compile_better").to_json()
    else:
        catalog.put_region(region)
    output_name = next(iter(payload.get("outputs") or {}), None)
    output_sha = None
    if output_name:
        output_sha = ((payload.get("outputs") or {}).get(output_name) or {}).get("sha256")
    timed_seconds = float(summary.get("timed_elapsed_seconds") or 0.0)
    daemon_seconds = float(summary.get("timed_daemon_elapsed_seconds") or 0.0)
    elements = int(region.elements)
    timed_packets = int(summary.get("timed_graph_packets") or 0)
    words = elements * timed_packets
    measurement = {
        "schema": "pan_fabric_measurement_v1",
        "candidate_id": None if candidate is None else candidate.candidate_id,
        "source_path": "fabric-exec:fabric-session",
        "ok": bool(payload.get("ok")),
        "execution_backend": "daemon_session_resident",
        "target": target,
        "ops": list(region.ops),
        "elements": elements,
        "elapsed_seconds": timed_seconds,
        "observed_run_daemon_seconds": daemon_seconds,
        "observed_run_pl_seconds": 0.0,
        "observed_transfer_bytes": int(summary.get("timed_transfer_bytes") or 0),
        "output_sha256": output_sha,
        "timed_mwords_per_s": (words / timed_seconds / 1e6) if timed_seconds else 0.0,
        "timed_daemon_mwords_per_s": (words / daemon_seconds / 1e6) if daemon_seconds else 0.0,
        "resident": True,
        "timed_graph_packets": timed_packets,
        "ingested_at": time.time(),
    }
    catalog.append_measurement(measurement)
    hint = scheduler_hint_for_graph(graph, catalog=catalog, target=target)
    payload["foundry_hint"] = hint
    payload["foundry_queued"] = queued
    payload["foundry"] = {
        "catalog": str(catalog.root),
        "region": region.to_json(),
        "candidate": None if candidate is None else candidate.to_json(),
        "queued": queued,
        "measurement": measurement,
        "hint": hint,
    }


def _attach_chain_foundry(payload: dict[str, Any], graph: FabricGraph, args: argparse.Namespace) -> None:
    catalog_root = getattr(args, "foundry_catalog", None)
    if not catalog_root:
        return
    catalog = _foundry_catalog(catalog_root, args)
    target = getattr(args, "foundry_target", "kv260")
    region = mine_region_from_graph(graph)
    from .fabric_foundry import FabricRegion

    region = FabricRegion(
        region_id=region.region_id,
        ops=region.ops,
        elements=region.elements,
        occurrence_count=int(payload.get("reps") or 1),
        observed_seconds=float(payload.get("timed_elapsed_seconds") or 0.0),
        observed_transfer_bytes=int(payload.get("timed_transfer_bytes") or 0),
        source_paths=("fabric-daemon-chain-bench",),
        graph=graph,
    )
    candidates = generate_candidates([region], target=target)
    candidate = candidates[0] if candidates else None
    queued = None
    if candidate is not None:
        catalog.put_region(region)
        catalog.put_candidate(candidate)
        artifact_status = "queued" if getattr(args, "foundry_auto_queue", False) and candidate.priority > 0.0 else "candidate"
        catalog.put_core_manifest(candidate, artifact_status=artifact_status)
        if artifact_status == "queued":
            queued = catalog.append_queue(candidate, reason="resident_daemon_measurement_compile_better").to_json()
    else:
        catalog.put_region(region)
    measurement = {
        "schema": "pan_fabric_measurement_v1",
        "candidate_id": None if candidate is None else candidate.candidate_id,
        "source_path": "fabric-daemon-chain-bench",
        "ok": bool(payload.get("ok")),
        "execution_backend": "daemon_session_resident",
        "target": target,
        "ops": list(region.ops),
        "elements": int(region.elements),
        "elapsed_seconds": float(payload.get("timed_elapsed_seconds") or 0.0),
        "observed_run_daemon_seconds": float(payload.get("timed_daemon_elapsed_seconds") or 0.0),
        "observed_run_pl_seconds": 0.0,
        "observed_transfer_bytes": int(payload.get("timed_transfer_bytes") or 0),
        "output_sha256": payload.get("final_sha256"),
        "timed_mwords_per_s": float(payload.get("timed_mwords_per_s") or 0.0),
        "timed_daemon_mwords_per_s": float(payload.get("timed_daemon_mwords_per_s") or 0.0),
        "resident": True,
        "timed_graph_packets": int(payload.get("timed_graph_packets") or 0),
        "ingested_at": time.time(),
    }
    catalog.append_measurement(measurement)
    hint = scheduler_hint_for_graph(graph, catalog=catalog, target=target)
    payload["foundry"] = {
        "catalog": str(catalog.root),
        "region": region.to_json(),
        "candidate": None if candidate is None else candidate.to_json(),
        "queued": queued,
        "measurement": measurement,
        "hint": hint,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", help="Learned checkpoint or manifest to load.")
        p.add_argument("--profile", default="long64-ring3", help="Runtime profile identity for proposal validation.")
        p.add_argument("--learned-only", action="store_true", help="Require a learned model and disable bootstrap fallback.")
        p.add_argument("--promotion-ledger", help="Promotion ledger JSON used to constrain learned runtime forms.")

    def add_backend_args(p: argparse.ArgumentParser, *, traces: bool = False) -> None:
        p.add_argument("--fabric", choices=("auto", "off", "force-ps", "force-pl", "backend"), default="backend")
        p.add_argument("--backend", choices=("auto", "ps", "pl-sim", "pl-xrt"), default="auto")
        p.add_argument(
            "--fabric-daemon",
            nargs="?",
            const="/tmp/pan-fabricd.sock",
            default=None,
            help="Unix socket for resident pan-fabricd graph execution and fabric telemetry.",
        )
        p.add_argument("--pl-xclbin", help="XRT xclbin implementing the pan_simd_v1 packet ABI.")
        p.add_argument("--pl-kernel-name", default="pan_simd_v1", help="XRT kernel name inside --pl-xclbin.")
        p.add_argument("--pl-min-instructions", type=int, default=8, help="Minimum SIMD instructions for PL routing.")
        p.add_argument("--pl-min-groups", type=int, default=2, help="Repeated SIMD groups required for short PL blocks.")
        if traces:
            p.add_argument("--route-trace", help="Write PS/PL route decisions as JSONL.")
            p.add_argument("--backend-stats", help="Write PS/PL backend counters and transfer stats as JSON.")
            p.add_argument("--fabric-trace", help="Write integrated PAN fabric runtime observations as JSONL.")
            p.add_argument("--fabric-training-jsonl", help="Write integrated PAN fabric placement training records as JSONL.")

    run = subparsers.add_parser("run", help="Run a static x86-64 ELF under the SPF PAN runtime.")
    run.add_argument("--program", required=True)
    run.add_argument("--argv0", help="Override argv[0] for reproducible stack layouts across hosts.")
    run.add_argument("--argv", nargs="*", default=[])
    run.add_argument("--uid", type=int, help="Override AT_UID for reproducible auxv.")
    run.add_argument("--euid", type=int, help="Override AT_EUID; defaults to --uid when omitted.")
    run.add_argument("--gid", type=int, help="Override AT_GID for reproducible auxv.")
    run.add_argument("--egid", type=int, help="Override AT_EGID; defaults to --gid when omitted.")
    run.add_argument("--max-steps", type=int, default=100000)
    run.add_argument("--gdb", help="Serve one GDB remote session, e.g. :1234.")
    add_model_args(run)
    add_backend_args(run, traces=True)
    run.add_argument("--trace-proposals", help="Write committed/attempted transition proposals as JSONL.")
    run.add_argument("--write-action-ledger-jsonl", help="Append emitted ISA transition action ledger rows to this JSONL file.")
    run.add_argument("--write")
    run.set_defaults(func=cmd_run)

    compare = subparsers.add_parser("compare", help="Compare SPF PAN execution against qemu-x86_64 when available.")
    compare.add_argument("--program", required=True)
    compare.add_argument("--argv0", help="Override argv[0] for reproducible stack layouts across hosts.")
    compare.add_argument("--argv", nargs="*", default=[])
    compare.add_argument("--uid", type=int, help="Override AT_UID for reproducible auxv.")
    compare.add_argument("--euid", type=int, help="Override AT_EUID; defaults to --uid when omitted.")
    compare.add_argument("--gid", type=int, help="Override AT_GID for reproducible auxv.")
    compare.add_argument("--egid", type=int, help="Override AT_EGID; defaults to --gid when omitted.")
    compare.add_argument("--max-steps", type=int, default=100000)
    compare.add_argument("--engine", choices=("qemu",), default="qemu")
    add_model_args(compare)
    add_backend_args(compare)
    compare.add_argument("--write")
    compare.set_defaults(func=cmd_compare)

    gdb = subparsers.add_parser("gdb", help="Serve one GDB remote session for the SPF runtime.")
    gdb.add_argument("--program", required=True)
    gdb.add_argument("--argv0", help="Override argv[0] for reproducible stack layouts across hosts.")
    gdb.add_argument("--argv", nargs="*", default=[])
    gdb.add_argument("--uid", type=int, help="Override AT_UID for reproducible auxv.")
    gdb.add_argument("--euid", type=int, help="Override AT_EUID; defaults to --uid when omitted.")
    gdb.add_argument("--gid", type=int, help="Override AT_GID for reproducible auxv.")
    gdb.add_argument("--egid", type=int, help="Override AT_EGID; defaults to --gid when omitted.")
    gdb.add_argument("--listen", default="127.0.0.1:1234")
    add_model_args(gdb)
    add_backend_args(gdb, traces=True)
    gdb.add_argument("--trace-proposals", help="Write committed/attempted transition proposals as JSONL.")
    gdb.add_argument("--write")
    gdb.set_defaults(func=cmd_gdb)

    train = subparsers.add_parser("train", help="Train a neural runtime transition model.")
    train.add_argument("--trace-manifest")
    train.add_argument("--model-out", required=True, help="JSON manifest path containing exported runtime weights.")
    train.add_argument("--checkpoint-out", help="Optional Torch checkpoint path; defaults next to --model-out.")
    train.add_argument("--samples-per-form", type=int, default=16)
    train.add_argument("--train-steps", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=2e-3)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--hidden-dim", type=int, default=96)
    train.add_argument("--embedding-dim", type=int, default=8)
    train.add_argument("--min-confidence", type=float, default=0.50)
    train.add_argument("--write", help="Write the training report without embedded weights; stdout when omitted.")
    train.set_defaults(func=cmd_train)

    validate = subparsers.add_parser("validate", help="Run a comparison-backed runtime validation suite.")
    validate.add_argument("--program", required=True)
    validate.add_argument("--argv0", help="Override argv[0] for reproducible stack layouts across hosts.")
    validate.add_argument("--argv", nargs="*", default=[])
    validate.add_argument("--uid", type=int, help="Override AT_UID for reproducible auxv.")
    validate.add_argument("--euid", type=int, help="Override AT_EUID; defaults to --uid when omitted.")
    validate.add_argument("--gid", type=int, help="Override AT_GID for reproducible auxv.")
    validate.add_argument("--egid", type=int, help="Override AT_EGID; defaults to --gid when omitted.")
    validate.add_argument("--suite", default="smoke")
    validate.add_argument("--max-steps", type=int, default=100000)
    add_model_args(validate)
    add_backend_args(validate)
    validate.add_argument("--write")
    validate.set_defaults(func=cmd_validate)

    inspect = subparsers.add_parser("inspect-model", help="Inspect Bob SPF dependency and runtime model manifest.")
    inspect.add_argument("--model")
    inspect.add_argument("--profile", default="long64-ring3")
    inspect.add_argument("--learned-only", action="store_true")
    inspect.add_argument("--promotion-ledger")
    inspect.add_argument("--pl-xclbin")
    inspect.add_argument("--pl-kernel-name", default="pan_simd_v1")
    inspect.add_argument("--write")
    inspect.set_defaults(func=cmd_inspect_model)

    fabric = subparsers.add_parser("inspect-fabric", help="Inspect existing KV260/SPF PL kernels and fabric availability.")
    fabric.add_argument("--workdir", help="SPF workdir on a KV260-style target; defaults to PAN_FABRIC_WORKDIR or /home/ubuntu/spfr.")
    fabric.add_argument("--xclbin-dir", action="append", help="Directory containing xclbins; may be supplied more than once.")
    fabric.add_argument("--probe-xrt", action="store_true", help="Also probe pyxrt device visibility and active xmutil app.")
    fabric.add_argument("--write")
    fabric.set_defaults(func=cmd_inspect_fabric)

    route = subparsers.add_parser("fabric-route", help="Explain PS/PL routing for one typed SPF fabric operation.")
    route.add_argument("--op", required=True, help="Typed SPF op, e.g. spf32.mul or spf32.fft64.")
    route.add_argument("--elements", type=int, required=True, help="Number of SPF words/elements in the operation.")
    route.add_argument("--fabric-backend", choices=("auto", "ps", "pl"), default="auto")
    route.add_argument("--min-pl-elements", type=int, default=1024)
    route.add_argument("--active-app", default="auto", help="Active xmutil app name, 'auto' to probe, or empty for none.")
    route.add_argument("--workdir", help="SPF workdir on a KV260-style target.")
    route.add_argument("--xclbin-dir", action="append", help="Directory containing xclbins; may be supplied more than once.")
    route.add_argument("--write")
    route.set_defaults(func=cmd_fabric_route)

    smoke = subparsers.add_parser("fabric-smoke", help="Run an on-board SPF PL smoke through the PAN fabric session.")
    smoke.add_argument("--op", default="spf32.mul", help="Smoke operation; currently supports spf32.mul.")
    smoke.add_argument("--n", type=int, default=4096, help="Number of SPF32 words to exercise.")
    smoke.add_argument("--repeats", type=int, default=1, help="Dispatch count through one resident XRT session.")
    smoke.add_argument("--seed", type=int, default=1)
    smoke.add_argument("--workdir", help="SPF workdir on the KV260 target.")
    smoke.add_argument("--xclbin-dir", action="append", help="Directory containing xclbins; may be supplied more than once.")
    smoke.add_argument("--libspf32", help="Path to libspf32.so for golden comparison.")
    smoke.add_argument("--load-app", action="store_true", help="Unload the current xmutil app and load the selected kernel app before running.")
    smoke.add_argument("--write")
    smoke.set_defaults(func=cmd_fabric_smoke)

    plan = subparsers.add_parser("fabric-plan", help="Plan PS/PL residency for a typed SPF operation graph.")
    source = plan.add_mutually_exclusive_group(required=True)
    source.add_argument("--graph", help="JSON graph file using schema pan_fabric_graph_v1.")
    source.add_argument("--linear-ops", help="Comma-separated typed SPF ops for a simple linear graph.")
    plan.add_argument("--elements", type=int, default=4096, help="Elements per op for --linear-ops.")
    plan.add_argument("--input-residency", choices=("ps", "pl", "both"), default="ps")
    plan.add_argument("--output-residency", choices=("ps", "pl", "both"), default="ps")
    plan.add_argument("--fabric-backend", choices=("auto", "ps", "pl"), default="auto")
    plan.add_argument("--active-app", default="auto", help="Active xmutil app name, 'auto' to probe, or empty for none.")
    plan.add_argument("--policy", help="Optional JSON linear fabric policy head.")
    plan.add_argument("--workdir", help="SPF workdir on a KV260-style target.")
    plan.add_argument("--xclbin-dir", action="append", help="Directory containing xclbins; may be supplied more than once.")
    plan.add_argument("--foundry-catalog", help="Optional PAN fabric foundry catalog for scheduler hints.")
    plan.add_argument("--foundry-target", default="kv260")
    plan.add_argument("--foundry-auto-queue", action="store_true", help="Queue a foundry build candidate when the hint recommends it.")
    plan.add_argument("--foundry-action-ledger-jsonl", help="Append emitted foundry lifecycle action ledger rows to this JSONL file.")
    plan.add_argument("--write")
    plan.set_defaults(func=cmd_fabric_plan)

    exec_parser = subparsers.add_parser("fabric-exec", help="Plan and execute a typed SPF graph with resident PS/PL buffers.")
    exec_source = exec_parser.add_mutually_exclusive_group(required=True)
    exec_source.add_argument("--graph", help="JSON graph file using schema pan_fabric_graph_v1.")
    exec_source.add_argument("--linear-ops", help="Comma-separated typed SPF ops for a simple linear graph.")
    exec_parser.add_argument("--elements", type=int, default=4096, help="Elements per op for --linear-ops.")
    exec_parser.add_argument("--input-residency", choices=("ps", "pl", "both"), default="ps")
    exec_parser.add_argument("--output-residency", choices=("ps", "pl", "both"), default="ps")
    exec_parser.add_argument("--fabric-backend", choices=("auto", "ps", "pl"), default="auto")
    exec_parser.add_argument("--execute-backend", choices=("auto", "sim", "xrt", "daemon", "fabric-session"), default="auto")
    exec_parser.add_argument("--active-app", default="auto", help="Active xmutil app name, 'auto' to probe, or empty for none.")
    exec_parser.add_argument("--policy", help="Optional JSON linear fabric policy head.")
    exec_parser.add_argument("--inputs", help="Optional JSON mapping buffer names to uint32 word arrays.")
    exec_parser.add_argument("--seed", type=int, default=1)
    exec_parser.add_argument("--repeats", type=int, default=1)
    exec_parser.add_argument("--workdir", help="SPF workdir on a KV260-style target.")
    exec_parser.add_argument("--xclbin-dir", action="append", help="Directory containing xclbins; may be supplied more than once.")
    exec_parser.add_argument("--libspf32", help="Path to libspf32.so for PS-side SPF execution.")
    exec_parser.add_argument("--libspf16", help="Path to libspf16.so for PS-side packed SPF16 validation.")
    exec_parser.add_argument("--load-app", action="store_true", help="Load the selected xclbin app before XRT PL regions.")
    exec_parser.add_argument("--enable-bo-import", action="store_true", help="Try XRT BO fd import/export across compatible PL regions.")
    exec_parser.add_argument("--allow-xrt-app-switch", action="store_true", help="Experimentally allow multiple xclbin apps in one Python XRT process.")
    exec_parser.add_argument("--no-output-digest", action="store_true", help="Do not materialize final PL outputs only to compute hashes.")
    exec_parser.add_argument("--fabric-daemon", default="/tmp/pan-fabricd.sock", help="Unix socket for resident pan-fabricd graph execution.")
    exec_parser.add_argument(
        "--daemon-buffer-storage",
        choices=("inline", "fd"),
        default="inline",
        help="Graph packet buffer transport for pan-fabricd.",
    )
    exec_parser.add_argument("--trace-jsonl", help="Write observed execution step trace as JSONL.")
    exec_parser.add_argument("--training-jsonl", help="Write policy-training records as JSONL.")
    exec_parser.add_argument("--write-action-ledger-jsonl", help="Append emitted fabric action ledger rows to this JSONL file.")
    exec_parser.add_argument("--include-trace", action="store_true", help="Embed the full step trace in the JSON response.")
    exec_parser.add_argument("--foundry-catalog", help="Optional PAN fabric foundry catalog for scheduler hints.")
    exec_parser.add_argument("--foundry-target", default="kv260")
    exec_parser.add_argument("--foundry-auto-queue", action="store_true", help="Queue a foundry build candidate when the hint recommends it.")
    exec_parser.add_argument("--foundry-action-ledger-jsonl", help="Append emitted foundry lifecycle action ledger rows to this JSONL file.")
    exec_parser.add_argument("--write")
    exec_parser.set_defaults(func=cmd_fabric_exec)

    train_policy = subparsers.add_parser("fabric-train-policy", help="Fit a linear neural fabric policy head from fabric-exec traces.")
    train_policy.add_argument("--trace-jsonl", nargs="+", required=True, help="One or more fabric-exec --training-jsonl files.")
    train_policy.add_argument("--policy-out", help="Write the trained policy JSON.")
    train_policy.add_argument("--l2", type=float, default=1.0e-6)
    train_policy.add_argument("--write")
    train_policy.set_defaults(func=cmd_fabric_train_policy)

    promote_policy = subparsers.add_parser("fabric-promote-policy", help="Promote a neural fabric policy only when held-out gates pass.")
    promote_policy.add_argument("--policy", required=True, help="Candidate policy JSON produced by fabric-train-policy.")
    promote_policy.add_argument("--heldout-metrics", required=True, help="Held-out comparison metrics JSON.")
    promote_policy.add_argument("--policy-out", help="Write the promoted or still-candidate policy JSON.")
    promote_policy.add_argument("--min-samples", type=int, default=1)
    promote_policy.add_argument("--max-latency-ratio", type=float, default=1.0)
    promote_policy.add_argument("--max-transfer-ratio", type=float, default=1.0)
    promote_policy.add_argument("--max-energy-ratio", type=float, default=1.0)
    promote_policy.add_argument("--min-compile-priority-precision", type=float, default=0.80)
    promote_policy.add_argument("--write")
    promote_policy.set_defaults(func=cmd_fabric_promote_policy)

    nn_run = subparsers.add_parser("nn-run", help="Run a typed PAN neural-network graph through PS or PAN Fabric planning.")
    nn_run.add_argument("--model", required=True, help="ONNX model or pan_nn_graph_v1 JSON model.")
    nn_run.add_argument("--backend", choices=("auto", "ps", "sim", "fabric-session"), default="auto")
    nn_run.add_argument("--fabric", choices=("auto", "ps", "sim", "fabric-session"), help="Alias for --backend; matches the PAN-LLM north-star CLI.")
    nn_run.add_argument("--target", default="kv260")
    nn_run.add_argument("--inputs", help="Optional JSON mapping input tensor names to nested arrays.")
    nn_run.add_argument("--seed", type=int, default=1)
    nn_run.add_argument("--repeats", type=int, default=1, help="Run repeated inferences through one retained NN runtime session.")
    nn_run.add_argument("--session-id", help="Stable session id for retained NN tensor values.")
    nn_run.add_argument("--resident-weights", action="store_true", help="Treat initializer tensors as resident fabric values.")
    nn_run.add_argument("--resident-inputs", action="store_true", help="Retain identical input tensors across repeated NN session runs.")
    nn_run.add_argument(
        "--feedback-output",
        action="append",
        help="Retain a produced output as the next repeated run's input, formatted INPUT=OUTPUT.",
    )
    nn_run.add_argument(
        "--promotion-warmup-runs",
        type=int,
        default=1,
        help="Exclude this many leading repeated resident-input runs from promotion timing and transfer metrics.",
    )
    nn_run.add_argument("--no-output-digest", action="store_true", help="Do not materialize final outputs only to compute digests.")
    nn_run.add_argument("--validate-output", action="store_true", help="Compare outputs against the NumPy PS oracle.")
    nn_run.add_argument(
        "--validation-rtol",
        type=float,
        help="Relative tolerance for NN output validation; overrides model metadata and runtime default.",
    )
    nn_run.add_argument(
        "--validation-atol",
        type=float,
        help="Absolute tolerance for NN output validation; overrides model metadata and runtime default.",
    )
    nn_run.add_argument("--validation-sample-count", type=int, help="Validate only this many sampled elements per output; omit for full validation.")
    nn_run.add_argument("--fabric-daemon", default="/tmp/pan-fabricd.sock")
    nn_run.add_argument("--workdir", default="/home/ubuntu/spfr")
    nn_run.add_argument("--xclbin-dir", action="append", help="Directory containing PAN-managed fabric xclbins.")
    nn_run.add_argument("--active-app", default="auto", help="Currently loaded xmutil app, or auto.")
    nn_run.add_argument("--fabric-backend", choices=("auto", "ps", "pl"), default="pl")
    nn_run.add_argument("--daemon-buffer-storage", choices=("inline", "fd"), default="inline")
    nn_run.add_argument("--policy", help="Optional gated PAN NN policy JSON produced by nn-train-policy.")
    nn_run.add_argument(
        "--policy-materialization",
        action="store_true",
        help="Allow a promoted NN policy materialization head to keep final outputs resident or request materialization.",
    )
    nn_run.add_argument("--trace-jsonl", help="Write NN region execution trace rows as JSONL.")
    nn_run.add_argument("--training-jsonl", help="Write NN policy-training rows as JSONL.")
    nn_run.add_argument("--include-trace", action="store_true", help="Embed trace and training rows in the JSON response.")
    nn_run.add_argument("--foundry-catalog", help="Reserved for NN foundry candidate ingestion.")
    nn_run.add_argument("--foundry-auto-queue", action="store_true", help="Mark repeated eligible NN regions as compile-better candidates.")
    nn_run.add_argument("--foundry-action-ledger-jsonl", help="Append emitted NN foundry lifecycle action ledger rows to this JSONL file.")
    nn_run.add_argument("--write")
    nn_run.set_defaults(func=cmd_nn_run)

    nn_train_policy = subparsers.add_parser("nn-train-policy", help="Fit a gated neural PAN NN fabric policy from nn-run training rows.")
    nn_train_policy.add_argument("--trace-jsonl", nargs="+", required=True, help="One or more nn-run --training-jsonl files.")
    nn_train_policy.add_argument("--policy-out", help="Write the trained NN policy JSON.")
    nn_train_policy.add_argument("--l2", type=float, default=1.0e-6)
    nn_train_policy.add_argument("--write")
    nn_train_policy.set_defaults(func=cmd_nn_train_policy)

    values = subparsers.add_parser("fabric-values", help="Inspect PAN fabric values from execution JSON or pan-fabricd.")
    values.add_argument("--execution-json", help="fabric-exec JSON output containing value_store/fabric_values.")
    values.add_argument("--fabric-daemon", help="Unix socket for resident pan-fabricd value management.")
    value_action = values.add_mutually_exclusive_group()
    value_action.add_argument("--describe-handle", type=int, help="Describe one daemon-resident BO handle.")
    value_action.add_argument("--materialize-handle", type=int, help="Materialize one daemon-resident BO handle.")
    value_action.add_argument("--retain-handle", type=int, help="Retain one daemon-resident BO handle.")
    value_action.add_argument("--release-handle", type=int, help="Release one daemon-resident BO handle.")
    value_action.add_argument("--export-handle", type=int, help="Attempt BO export for one daemon-resident handle.")
    values.add_argument("--elements", type=int, default=0, help="Expected element count for --materialize-handle.")
    values.add_argument("--timeout", type=float, default=120.0)
    values.add_argument("--write")
    values.set_defaults(func=cmd_fabric_values)

    daemon_bench = subparsers.add_parser("fabric-daemon-bench", help="Benchmark a resident pan-fabricd Unix socket.")
    daemon_bench.add_argument("--socket", default="/tmp/pan-fabricd.sock")
    daemon_bench.add_argument("--n", type=int, default=4096)
    daemon_bench.add_argument("--reps", type=int, default=2000)
    daemon_bench.add_argument("--warmup", type=int, default=20)
    daemon_bench.add_argument("--seed-a", type=int, default=3310)
    daemon_bench.add_argument("--seed-b", type=int, default=4310)
    daemon_bench.add_argument("--timeout", type=float, default=120.0)
    daemon_bench.add_argument("--write")
    daemon_bench.set_defaults(func=cmd_fabric_daemon_bench)

    daemon_chain = subparsers.add_parser(
        "fabric-daemon-chain-bench",
        help="Benchmark repeated graph packets over one persistent pan-fabricd connection.",
    )
    daemon_chain.add_argument("--socket", default="/tmp/pan-fabricd.sock")
    daemon_chain.add_argument("--n", type=int, default=4096)
    daemon_chain.add_argument("--reps", type=int, default=100)
    daemon_chain.add_argument("--warmup", type=int, default=5)
    daemon_chain.add_argument("--seed", type=int, default=1)
    daemon_chain.add_argument("--timeout", type=float, default=120.0)
    daemon_chain.add_argument("--workdir", default="/home/ubuntu/spfr")
    daemon_chain.add_argument("--xclbin-dir", action="append", help="Directory containing xclbins; may be supplied more than once.")
    daemon_chain.add_argument("--libspf32", help="Path to libspf32.so for PS-side validation.")
    daemon_chain.add_argument("--active-app", default="spf32_core_pl")
    daemon_chain.add_argument("--materialize-final", action="store_true", help="Materialize the final resident BO after timed packets.")
    daemon_chain.add_argument("--validate-final", action="store_true", help="Materialize and compare the final output against NumPy.")
    daemon_chain.add_argument("--foundry-catalog", help="Optional PAN fabric foundry catalog for resident measurement ingestion.")
    daemon_chain.add_argument("--foundry-target", default="kv260")
    daemon_chain.add_argument("--foundry-auto-queue", action="store_true", help="Queue a better fused/core candidate from the resident measurement.")
    daemon_chain.add_argument("--foundry-action-ledger-jsonl", help="Append emitted foundry lifecycle action ledger rows to this JSONL file.")
    daemon_chain.add_argument("--write")
    daemon_chain.set_defaults(func=cmd_fabric_daemon_chain_bench)

    foundry_mine = subparsers.add_parser("fabric-foundry-mine", help="Mine repeated typed SPF graph regions from execution traces.")
    foundry_mine.add_argument("--execution-json", action="append", default=[], help="fabric-exec JSON output to mine; may be repeated.")
    foundry_mine.add_argument("--trace-jsonl", action="append", default=[], help="fabric-exec trace/training JSONL to mine; may be repeated.")
    foundry_mine.add_argument("--catalog", default=".pan_fabric_foundry")
    foundry_mine.add_argument("--max-ops", type=int, default=6)
    foundry_mine.add_argument("--min-count", type=int, default=1)
    foundry_mine.add_argument("--foundry-action-ledger-jsonl", help="Append emitted foundry lifecycle action ledger rows to this JSONL file.")
    foundry_mine.add_argument("--write")
    foundry_mine.set_defaults(func=cmd_fabric_foundry_mine)

    foundry_candidates = subparsers.add_parser("fabric-foundry-candidates", help="Generate hardware candidates and optional build queue entries.")
    foundry_candidates.add_argument("--catalog", default=".pan_fabric_foundry")
    foundry_candidates.add_argument("--target", default="kv260")
    foundry_candidates.add_argument("--queue", action="store_true")
    foundry_candidates.add_argument("--min-priority", type=float, default=0.0)
    foundry_candidates.add_argument("--foundry-action-ledger-jsonl", help="Append emitted foundry lifecycle action ledger rows to this JSONL file.")
    foundry_candidates.add_argument("--write")
    foundry_candidates.set_defaults(func=cmd_fabric_foundry_candidates)

    foundry_ingest = subparsers.add_parser("fabric-foundry-ingest", help="Ingest measured runtime output into the fabric foundry catalog.")
    foundry_ingest.add_argument("--catalog", default=".pan_fabric_foundry")
    foundry_ingest.add_argument("--measurement", action="append", required=True, help="Measurement JSON to ingest; may be repeated.")
    foundry_ingest.add_argument("--candidate-id")
    foundry_ingest.add_argument("--artifact", help="Optional built artifact path associated with this measurement.")
    foundry_ingest.add_argument("--foundry-action-ledger-jsonl", help="Append emitted foundry lifecycle action ledger rows to this JSONL file.")
    foundry_ingest.add_argument("--write")
    foundry_ingest.set_defaults(func=cmd_fabric_foundry_ingest)

    foundry_schedule = subparsers.add_parser("fabric-foundry-schedule", help="Ask the foundry whether to use existing fabric or compile better hardware.")
    schedule_source = foundry_schedule.add_mutually_exclusive_group(required=True)
    schedule_source.add_argument("--graph", help="JSON graph file using schema pan_fabric_graph_v1.")
    schedule_source.add_argument("--linear-ops", help="Comma-separated typed SPF ops for a simple linear graph.")
    foundry_schedule.add_argument("--elements", type=int, default=4096)
    foundry_schedule.add_argument("--input-residency", choices=("ps", "pl", "both"), default="ps")
    foundry_schedule.add_argument("--output-residency", choices=("ps", "pl", "both"), default="ps")
    foundry_schedule.add_argument("--catalog", default=".pan_fabric_foundry")
    foundry_schedule.add_argument("--target", default="kv260")
    foundry_schedule.add_argument("--auto-queue", action="store_true")
    foundry_schedule.add_argument("--foundry-action-ledger-jsonl", help="Append emitted foundry lifecycle action ledger rows to this JSONL file.")
    foundry_schedule.add_argument("--write")
    foundry_schedule.set_defaults(func=cmd_fabric_foundry_schedule)

    foundry_status = subparsers.add_parser("fabric-foundry-status", help="Summarize the PAN fabric foundry catalog.")
    foundry_status.add_argument("--catalog", default=".pan_fabric_foundry")
    foundry_status.add_argument("--foundry-action-ledger-jsonl", help="Read foundry lifecycle action ledger row counts from this JSONL file.")
    foundry_status.add_argument("--write")
    foundry_status.set_defaults(func=cmd_fabric_foundry_status)

    control = subparsers.add_parser("fabric-control", help="Inspect the PAN living fabric control-plane ledger and scheduler state.")
    control_source = control.add_mutually_exclusive_group()
    control_source.add_argument("--graph", help="JSON graph file using schema pan_fabric_graph_v1.")
    control_source.add_argument("--linear-ops", help="Comma-separated typed SPF ops for a simple linear graph.")
    control.add_argument("--elements", type=int, default=4096)
    control.add_argument("--input-residency", choices=("ps", "pl", "both"), default="ps")
    control.add_argument("--output-residency", choices=("ps", "pl", "both"), default="ps")
    control.add_argument("--catalog", default=".pan_fabric_foundry")
    control.add_argument("--target", default="kv260")
    control.add_argument("--foundry-action-ledger-jsonl", help="Read foundry lifecycle action ledger row counts from this JSONL file.")
    control.add_argument("--write")
    control.set_defaults(func=cmd_fabric_control)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
