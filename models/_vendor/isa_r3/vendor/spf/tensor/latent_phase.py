"""Continuous-input latent phase-projection surface (experimental PAN handoff).

SPF already certifies known character surfaces over **discrete** indices
(:mod:`spf.tensor.character`, :mod:`spf.tensor.surface`). A live transformer's
hidden states have no such index — the phase basis must be *recovered* from the
continuous vectors first. This module builds that candidate basis: a learned
phase projection ``φ = x·W + b`` (optionally ``+ pos·ω``), lifted to a SPF-native
``[cos φ, sin φ]`` feature bank, with a low-rank ridge decoder for a target
residual.

It is deliberately marked experimental. A discrete character surface is exact
and certifiable; a recovered continuous basis is a falsifiable *candidate* that
must survive held-out behaviour before promotion. So the fitter emits an
``spf_handoff`` block naming the SPF primitives it would lower to
(``latent_phase_projection``, ``phase_feature_bank``, ``low_rank_residual_decode``,
``margin_risk_gate``) and an honest ``status: local_only`` until that check
passes. The kind ``BaseModelKind.LATENT_PHASE_PROJECTION`` is reserved for the
promoted form; this module does not register a v3 base model (the operator is
input-dependent, not a static stored wavefield).

The two SPF-native pieces are real today: ``cos φ`` / ``sin φ`` is exactly
``decode_complex`` of a phase-only word (phase add is O(1) in log-polar), and the
position term ``pos·ω`` is a SPF v2 phase ramp (``ramp_R`` / ``ramp_b``).

Pure ``numpy`` — the PAN original is torch; this is the SPF reference port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "LatentPhaseProjection",
    "add_margin_risk_gate",
    "fit_latent_spectral_residual_bridge",
    "latent_spectral_feature_bank",
]


def latent_spectral_feature_bank(
    inputs: np.ndarray,
    phase_weight: np.ndarray,
    phase_bias: np.ndarray,
    *,
    position_weight: np.ndarray | None = None,
) -> np.ndarray:
    """``[cos φ, sin φ]`` features for live hidden vectors, ``φ = x·W + b (+ pos·ω)``.

    ``inputs`` is ``(..., width)``; the returned bank is ``(..., 2·rank)`` (the
    cos and sin halves concatenated). When ``position_weight`` is given, a
    per-position phase ramp ``pos·ω`` is added along the second-to-last axis — the
    SPF v2 ramp, in continuous form.
    """
    x = np.asarray(inputs, dtype=np.float64)
    flat = x.reshape(-1, int(x.shape[-1]))
    phase = flat @ np.asarray(phase_weight, dtype=np.float64) + np.asarray(
        phase_bias, dtype=np.float64
    ).reshape(1, -1)
    if position_weight is not None:
        if x.ndim < 2:
            raise ValueError("position term needs at least a (position, width) input")
        seq_len = int(x.shape[-2])
        rank = phase.shape[-1]
        # seq_len always divides flat.shape[0] (it is one of the leading dims), so
        # the ramp tiles cleanly; the real failure mode is a position_weight that
        # does not carry one ω per phase channel — catch it with a clear message.
        pw = np.asarray(position_weight, dtype=np.float64).reshape(-1)
        if pw.size != rank:
            raise ValueError(
                f"position_weight length {pw.size} != phase rank {rank}; "
                "the per-position ramp needs one ω per phase channel"
            )
        pos = np.arange(seq_len, dtype=np.float64).reshape(seq_len, 1)
        pos_phase = pos * pw.reshape(1, rank)
        # Broadcast the (seq_len, rank) ramp across all leading batch rows.
        rows_per_seq = flat.shape[0] // seq_len
        phase = phase + np.tile(pos_phase, (rows_per_seq, 1))
    bank = np.concatenate([np.cos(phase), np.sin(phase)], axis=-1)
    return bank.reshape(*x.shape[:-1], -1)


def fit_latent_spectral_residual_bridge(
    inputs: np.ndarray,
    target_residual: np.ndarray,
    *,
    phase_rank: int = 8,
    ridge: float = 1e-4,
    include_position: bool = False,
) -> dict[str, Any]:
    """Fit a first-pass latent spectral residual bridge (basis-discovery candidate).

    The phase basis is seeded by principal hidden-state directions (PCA on the
    centred input), then a ridge solve fits the residual decoder from the
    ``[cos, sin]`` phase features. **Not an SPF claim** — a candidate to falsify
    on held-out rollout; the returned program carries the ``spf_handoff`` status.
    """
    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(target_residual, dtype=np.float64)
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("latent spectral bridge fit expects at least 2D inputs and targets")
    if x.shape[:-1] != y.shape[:-1]:
        raise ValueError(f"input/target batch shape mismatch: {x.shape} vs {y.shape}")
    input_width = int(x.shape[-1])
    output_width = int(y.shape[-1])
    flat_x = x.reshape(-1, input_width)
    flat_y = y.reshape(-1, output_width)
    centered = flat_x - flat_x.mean(axis=0, keepdims=True)
    rank = int(min(max(1, phase_rank), input_width, flat_x.shape[0]))
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        phase_weight = np.ascontiguousarray(vh[:rank].T)
    except np.linalg.LinAlgError:
        phase_weight = np.eye(input_width, rank, dtype=np.float64)
    phase_bias = np.zeros(rank, dtype=np.float64)
    position_weight = np.zeros(rank, dtype=np.float64) if include_position else None
    features = latent_spectral_feature_bank(
        x, phase_weight, phase_bias, position_weight=position_weight
    ).reshape(-1, 2 * rank)
    design = np.concatenate([features, np.ones((features.shape[0], 1), dtype=np.float64)], axis=-1)
    gram = design.T @ design
    reg = float(ridge) * np.eye(gram.shape[0], dtype=np.float64)
    coeff = np.linalg.solve(gram + reg, design.T @ flat_y)
    decoder = coeff[:-1]
    bias = coeff[-1]
    pred = design @ coeff
    rel_l2 = float(np.linalg.norm(pred - flat_y) / max(float(np.linalg.norm(flat_y)), 1e-30))
    program: dict[str, Any] = {
        "operator": "LatentSpectralResidualBridge",
        "input_width": input_width,
        "output_width": output_width,
        "phase_rank": rank,
        "feature_count": int(2 * rank),
        "phase_weight": phase_weight.tolist(),
        "phase_bias": phase_bias.tolist(),
        "decoder": decoder.tolist(),
        "bias": bias.tolist(),
        "fit": {
            "train_relative_l2_error": rel_l2,
            "ridge": float(ridge),
            "basis_seed": "hidden_state_pca",
        },
        "spf_handoff": {
            "status": "local_only",
            "reason": (
                "SPF tensor certifies known discrete character/product-group surfaces; this "
                "operator first recovers a continuous hidden-state phase basis to be falsified."
            ),
            "candidate_primitives": [
                "latent_phase_projection",
                "phase_feature_bank",
                "low_rank_residual_decode",
                "margin_risk_gate",
            ],
            "promoted_base_model_kind": "LATENT_PHASE_PROJECTION",
        },
    }
    if position_weight is not None:
        program["position_weight"] = position_weight.tolist()
    return program


def add_margin_risk_gate(
    program: dict[str, Any],
    *,
    gate_weight: np.ndarray,
    gate_bias: float = 0.0,
    gate_temperature: float = 1.0,
) -> dict[str, Any]:
    """Attach a hidden-state margin-risk gate to a latent spectral bridge program."""
    if program.get("operator") != "LatentSpectralResidualBridge":
        raise ValueError("expected LatentSpectralResidualBridge")
    updated = dict(program)
    updated["gate_weight"] = np.asarray(gate_weight, dtype=np.float64).reshape(-1).tolist()
    updated["gate_bias"] = [float(gate_bias)]
    updated["gate_temperature"] = float(gate_temperature)
    return updated


@dataclass(frozen=True)
class LatentPhaseProjection:
    """A fitted latent phase projection: ``φ = x·W + b``, decode ``[cos φ, sin φ]·D + c``.

    Build from a :func:`fit_latent_spectral_residual_bridge` program via
    :meth:`from_program`, or directly. ``apply`` runs the recovered operator on
    fresh hidden states — the held-out check that decides promotion to SPF.
    """

    phase_weight: np.ndarray
    phase_bias: np.ndarray
    decoder: np.ndarray
    bias: np.ndarray
    position_weight: np.ndarray | None = None
    handoff: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_program(cls, program: dict[str, Any]) -> LatentPhaseProjection:
        if program.get("operator") != "LatentSpectralResidualBridge":
            raise ValueError("expected LatentSpectralResidualBridge program")
        pos = program.get("position_weight")
        return cls(
            phase_weight=np.asarray(program["phase_weight"], dtype=np.float64),
            phase_bias=np.asarray(program["phase_bias"], dtype=np.float64),
            decoder=np.asarray(program["decoder"], dtype=np.float64),
            bias=np.asarray(program["bias"], dtype=np.float64),
            position_weight=None if pos is None else np.asarray(pos, dtype=np.float64),
            handoff=dict(program.get("spf_handoff", {})),
        )

    def features(self, inputs: np.ndarray) -> np.ndarray:
        return latent_spectral_feature_bank(
            inputs, self.phase_weight, self.phase_bias, position_weight=self.position_weight
        )

    def apply(self, inputs: np.ndarray) -> np.ndarray:
        """Predicted residual ``[cos φ, sin φ]·decoder + bias`` for ``inputs``."""
        x = np.asarray(inputs, dtype=np.float64)
        feats = self.features(x).reshape(-1, self.decoder.shape[0])
        out = feats @ self.decoder + self.bias.reshape(1, -1)
        return out.reshape(*x.shape[:-1], self.decoder.shape[1])
