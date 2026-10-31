"""Explicit source mounts for existing workspace engines, with source fingerprints.

Minimal namespace packages avoid unrelated heavyweight package initialization.
Every executed primitive remains the original source implementation. Launch
sealing copies these exact files; no arithmetic or state controller is vendored
as a second handwritten implementation in this repository.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
SOURCES = {
    "spf": WORKSPACE / "spectral-phase-format-research/src/spf",
    "spectral_native": WORKSPACE / "spf-wt-phaselock/domains/ml/spectral-native/spectral_native",
    "controller": WORKSPACE / "saturn/src/saturn/ar/training_checkpoint_controller.py",
}


def source(name):
    sealed = REPO / "vendor" / ("controller.py" if name == "controller" else name)
    path = sealed if sealed.exists() else SOURCES[name]
    if not path.exists():
        raise RuntimeError(f"required {name} source is unavailable: {path}")
    return path


def namespace(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if str(path) not in list(getattr(existing, "__path__", [])):
            raise RuntimeError(f"{name} already loaded from a different source")
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def spf_modules():
    root = source("spf")
    namespace("spf", root)
    namespace("spf.scalar", root / "scalar")
    namespace("spf.tensor", root / "tensor")
    return tuple(importlib.import_module(f"spf.tensor.{name}")
                 for name in ("core", "phase_circuits", "select_head"))


def field_modules():
    namespace("spectral_native", source("spectral_native"))
    return (importlib.import_module("spectral_native.field_fabric"),
            importlib.import_module("spectral_native.role_chart_field"))


def saturn_controller():
    name = "_isa_saturn_controller"
    if name not in sys.modules:
        path = source("controller")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def fingerprints():
    paths = [p for root in (REPO / "isa_lab", REPO / "nn", REPO / "fieldlm", REPO / "spf_transformer")
             for p in root.rglob("*.py")]
    for name in SOURCES:
        root = source(name)
        paths.extend(root.rglob("*.py") if root.is_dir() else [root])
    return {str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p.relative_to(WORKSPACE)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
