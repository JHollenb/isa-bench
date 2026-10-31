from __future__ import annotations

from .contracts import PROFILE, TRACK

MODEL_NAMES = ("nn", "fieldlm", "spf_transformer")


def create(name, seed=17, device="cpu"):
    if name == "nn":
        from nn.model import NNSelector
        return NNSelector(seed, device)
    if name == "fieldlm":
        from fieldlm.model import FieldLMSelector
        return FieldLMSelector(seed, device)
    if name == "spf_transformer":
        from spf_transformer.model import SPFSelector
        return SPFSelector(seed, device)
    raise ValueError(f"unknown model: {name}")


def restore(artifact, device="cpu"):
    from .data import digest
    from .training import contract_fingerprint
    if artifact.get("sha256") != digest({k: v for k, v in artifact.items() if k != "sha256"}):
        raise ValueError("model artifact hash mismatch")
    if artifact.get("schema") != "isa-model-v1" or artifact.get("profile") != PROFILE or artifact.get("track") != TRACK:
        raise ValueError("unknown artifact schema/profile/track")
    if artifact.get("contract_sha256") != contract_fingerprint():
        raise ValueError("serving contract differs from fitted artifact")
    model = create(artifact["model"], artifact["seed"], device)
    if artifact["payload"]["implementation"] != model.implementation:
        raise ValueError("artifact implementation differs from serving model")
    model.restore_export(artifact["payload"])
    return model
