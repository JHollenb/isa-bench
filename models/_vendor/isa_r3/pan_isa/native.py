"""Load frozen upstream implementations, including the original nested runtime head."""
from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from isa_lab.dependencies import namespace, source

ROOT = Path(__file__).resolve().parent / "upstream"
STAGE_MODULES = ("stage0_x86_pan", "stage1_memory_pan", "stage2_control_flow_pan",
                 "stage3_complex_scalar_pan", "stage4_simd_x87_pan")


def mount():
    for name in ("pan_x86_stages", "pan_x86_isa", "pan_isa_x86_runtime_spf", "pan_hardware_resolver"):
        namespace(name, ROOT / name)
    # The original runtime needs package-level SPF types; allow its real initializer.
    spf_parent = str(source("spf").parent)
    if spf_parent not in sys.path:
        sys.path.insert(0, spf_parent)
    os.environ["PAN_BOB_SPF_SRC"] = spf_parent


def stages():
    mount()
    return tuple(importlib.import_module(f"pan_x86_stages.{name}") for name in STAGE_MODULES)


def runtime_module():
    mount()
    return importlib.import_module("pan_isa_x86_runtime_spf.neural_runtime")


def original_runtime_head():
    """Execute the exact original class AST, without calling its classical trainer."""
    module = runtime_module()
    path = ROOT / "pan_isa_x86_runtime_spf/neural_runtime.py"
    tree = ast.parse(path.read_text())
    trainer = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
                   and x.name == "train_runtime_neural_model")
    head = next(x for x in trainer.body if isinstance(x, ast.ClassDef)
                and x.name == "TorchRuntimeNeuralHead")
    scope = {"torch": torch, "nn": nn, "F": F, "Any": object,
             "embedding_dim": module.DEFAULT_EMBEDDING_DIM,
             "hidden_dim": module.DEFAULT_HIDDEN_DIM,
             "RAW_BYTE_COUNT": module.RAW_BYTE_COUNT,
             "NUMERIC_FEATURE_NAMES": module.NUMERIC_FEATURE_NAMES,
             "RUNTIME_NEURAL_LABELS": module.RUNTIME_NEURAL_LABELS}
    exec(compile(ast.Module(body=[head], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102 - verified local source
    return scope["TorchRuntimeNeuralHead"]()


def native_heads():
    s = stages()
    return nn.ModuleDict({"0": s[0].FactorizedStage0PAN(8), "1": s[1].FactorizedStage1PAN(),
                          "2": s[2].FactorizedStage2PAN(), "3": s[3].FactorizedStage3PAN(),
                          "4": s[4].FactorizedStage4PAN(), "runtime": original_runtime_head()})
