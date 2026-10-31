"""Owned snapshots for repeated addressed SPFBlockV3 consumers."""

from __future__ import annotations

import copy
import operator
from dataclasses import dataclass, replace

import numpy as np

from spf.scalar import spf32
from spf.scalar._constants import SPF32

from . import core, v3
from .v2 import TensorBlock
from .v3 import SPFBlockV3


def _freeze_arrays(value):
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, dict):
        for item in value.values():
            _freeze_arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _freeze_arrays(item)
    elif hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            _freeze_arrays(getattr(value, name))


@dataclass(frozen=True)
class PreparedBlockConsumer:
    """Own the block and its mutable arrays; sampling delegates to the carrier.

    Preparation copies payloads. It is useful when reuse amortizes this cost;
    it does not create a second tensor representation or skip bounds checks.
    """

    _block: SPFBlockV3

    @classmethod
    def prepare(cls, block: SPFBlockV3):
        if not isinstance(block, SPFBlockV3):
            raise TypeError("prepared consumers require SPFBlockV3")
        owned = copy.deepcopy(block)
        owned.__post_init__()
        _freeze_arrays(owned)
        return cls(owned)

    def sample_words(self, linear, *, phase=True):
        return self._block.sample_words(linear, phase=phase)


def _integer_code(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("integer codes required, not bools")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError("integer codes required") from None


def fuse_shared_updates(block: TensorBlock, gains, phases) -> TensorBlock:
    """Fuse finite SPF32 gains into the existing header after a full-word certificate.

    Reject initial clipping and any saturated intermediate prefix. Absolute
    overrides execute the original multiplication chain, including sentinels.
    This preparation scans the block; reuse amortizes it. Returned residual
    arrays are shared with the source, so use PreparedBlockConsumer for ownership.
    """
    if not isinstance(block, TensorBlock):
        raise TypeError("shared updates require TensorBlock")
    gains = tuple(_integer_code(g) for g in gains)
    phases = tuple(_integer_code(t) for t in phases)
    header = block.header
    if len(gains) != len(phases):
        raise ValueError("one phase is required per gain")
    if any(not SPF32.l_code_min_finite <= int(g) <= SPF32.l_code_max_finite for g in gains):
        raise ValueError("finite gain codes required")
    # Check the original law BEFORE its packed finalizer clips it. A later
    # header change must not resurrect information lost by the initial clip.
    relabel = v3.tensorblock_to_v3(block)
    active, baseL, _ = v3.get_base_model(relabel.base.profile_id).reconstruct_planes(
        relabel.base.params, relabel.dims, relabel.origin
    )
    dL, _ = v3.get_residual_codec(relabel.residual.codec_id).decode(relabel.residual.payload, len(active))
    unclipped = baseL + dL
    ordinary = ~np.isin(active, list(block.outliers))
    if np.any((unclipped[ordinary] < SPF32.l_code_min_finite) | (unclipped[ordinary] > SPF32.l_code_max_finite)):
        raise ValueError("shared update would undo initial clipping or a sentinel")
    raw = block.to_words().reshape(-1)
    L = core.unpack_L(raw).astype(np.int64)
    finite = (SPF32.l_code_zero != L) & (raw != SPF32.word_nan)
    prefixes = []
    accumulated = 0
    for value in gains:
        accumulated += value
        prefixes.append(accumulated)
    if any(
        np.any((L[finite] + g < SPF32.l_code_min_finite) | (L[finite] + g > SPF32.l_code_max_finite)) for g in prefixes
    ):
        raise ValueError("shared update would hide intermediate saturation")
    # Active resolved sentinels were rejected above. Inactive ZERO entries
    # remain ZERO under multiplication and need no header certificate.
    gain, phase = int(sum(gains)), int(sum(phases)) & 65535
    overrides = dict(block.outliers)
    for index, word in overrides.items():
        value = np.uint32(word)
        for g, t in zip(gains, phases, strict=True):
            value = spf32.mul(value, core.pack_words(g, t))
        overrides[index] = int(value)
    return replace(
        block,
        header=replace(
            header, global_L_bias=header.global_L_bias + gain, global_T_bias=(header.global_T_bias + phase) & 65535
        ),
        outliers=overrides,
    )
