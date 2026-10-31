"""Repeated-run bench harness for sOFDM cross-link and link-load paths.

Replaces one-shot timing with statistics (min / p50 / p95 / mean / std /
iterations / warmup) and exposes four explicit benchmark modes so the
claim ledger can cite diagnostic-mode and hot-path-mode timings without
mixing them:

- :func:`run_cross_matrix_diagnostic` — full EVM/reference metrics
- :func:`run_cross_matrix_hot_path`   — diagnostics off
- :func:`run_link_load_diagnostic`    — link load with baseline + EVM
- :func:`run_link_load_hot_path`      — link load, no baseline, no EVM

All wrappers preserve the BER=0 cross-link contract.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from spf.dsp.ofdm_link import (
    CrossLinkCell,
    CrossLinkIdentity,
    CrossLinkMatrix,
    DACResult,
    SOFDMLoadConfig,
    SOFDMTXFrame,
    _time_ns,
    apply_known_channel,
    build_sofdm_tx_frame,
    run_basic_ofdm_baseline,
    run_cross_link_matrix,
    simulate_iq_dac,
)
from spf.dsp.ofdm_receiver import (
    ADCResult,
    SOFDMReceiverResult,
    receive_sofdm_frame,
    simulate_iq_adc,
)
from spf.dsp.ofdm_rx_plan import ChannelPolicy, OFDMRXPlan

T = TypeVar("T")
StageName: TypeAlias = str


@dataclass(frozen=True, slots=True)
class StageStats:
    """Aggregate timing stats for one stage across repeated runs."""

    min_ns: int
    p50_ns: int
    p95_ns: int
    mean_ns: float
    std_ns: float
    iterations: int
    warmup: int
    p99_ns: int | None = None

    def to_dict(self) -> dict[str, float | int]:
        return {
            "min_ns": self.min_ns,
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "p99_ns": self.p99_ns if self.p99_ns is not None else self.p95_ns,
            "mean_ns": self.mean_ns,
            "std_ns": self.std_ns,
            "iterations": self.iterations,
            "warmup": self.warmup,
        }


def _stats(samples: list[int], *, iterations: int, warmup: int) -> StageStats:
    arr = np.asarray(samples, dtype=np.int64)
    return StageStats(
        min_ns=int(arr.min()),
        p50_ns=int(np.percentile(arr, 50)),
        p95_ns=int(np.percentile(arr, 95)),
        p99_ns=int(np.percentile(arr, 99)),
        mean_ns=float(arr.mean()),
        std_ns=float(arr.std(ddof=0)),
        iterations=iterations,
        warmup=warmup,
    )


def _time_repeated(
    fn: Callable[[], T], *, iterations: int, warmup: int
) -> tuple[StageStats, T]:
    """Run ``fn`` warmup+iterations times; return aggregate stats and last result."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    for _ in range(warmup):
        fn()
    samples: list[int] = []
    last: T | None = None
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        last = fn()
        samples.append(time.perf_counter_ns() - t0)
    assert last is not None  # iterations >= 1
    return _stats(samples, iterations=iterations, warmup=warmup), last


@dataclass(frozen=True, slots=True)
class CrossLinkCellStats:
    """One cell of the repeated-run cross-link matrix."""

    tx_kind: str
    rx_kind: str
    bit_errors: int
    bits_total: int
    ber: float
    evm: float | None
    evm_db: float | None
    snr_db_estimate: float | None
    tx_ns: StageStats
    channel_ns: StageStats
    dac_ns: StageStats
    adc_ns: StageStats
    rx_ns: StageStats
    total_ns: StageStats

    def to_dict(self) -> dict[str, object]:
        return {
            "tx_kind": self.tx_kind,
            "rx_kind": self.rx_kind,
            "bit_errors": self.bit_errors,
            "bits_total": self.bits_total,
            "ber": self.ber,
            "evm": self.evm,
            "evm_db": self.evm_db,
            "snr_db_estimate": self.snr_db_estimate,
            "tx_ns": self.tx_ns.to_dict(),
            "channel_ns": self.channel_ns.to_dict(),
            "dac_ns": self.dac_ns.to_dict(),
            "adc_ns": self.adc_ns.to_dict(),
            "rx_ns": self.rx_ns.to_dict(),
            "total_ns": self.total_ns.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CrossLinkMatrixStats:
    """Repeated-run cross-link matrix stats wrapper."""

    config: SOFDMLoadConfig
    iterations: int
    warmup: int
    diagnostics: bool
    cells: tuple[CrossLinkCellStats, ...]
    identity: CrossLinkIdentity

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.config.scenario,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "diagnostics": self.diagnostics,
            "cells": [c.to_dict() for c in self.cells],
            "identity": {
                "fp_tx_rx_bits_match": self.identity.fp_tx_rx_bits_match,
                "fp_tx_rx_bit_diff": self.identity.fp_tx_rx_bit_diff,
                "sofdm_tx_rx_bits_match": self.identity.sofdm_tx_rx_bits_match,
                "sofdm_tx_rx_bit_diff": self.identity.sofdm_tx_rx_bit_diff,
            },
        }


def _aggregate_cross_cell(
    cell: CrossLinkCell,
    *,
    per_stage_samples: dict[StageName, list[int]],
    iterations: int,
    warmup: int,
) -> CrossLinkCellStats:
    return CrossLinkCellStats(
        tx_kind=cell.tx_kind,
        rx_kind=cell.rx_kind,
        bit_errors=cell.bit_errors,
        bits_total=cell.bits_total,
        ber=cell.ber,
        evm=cell.evm,
        evm_db=cell.evm_db,
        snr_db_estimate=cell.snr_db_estimate,
        tx_ns=_stats(per_stage_samples["tx_ns"], iterations=iterations, warmup=warmup),
        channel_ns=_stats(
            per_stage_samples["channel_ns"], iterations=iterations, warmup=warmup
        ),
        dac_ns=_stats(per_stage_samples["dac_ns"], iterations=iterations, warmup=warmup),
        adc_ns=_stats(per_stage_samples["adc_ns"], iterations=iterations, warmup=warmup),
        rx_ns=_stats(per_stage_samples["rx_ns"], iterations=iterations, warmup=warmup),
        total_ns=_stats(
            per_stage_samples["total_ns"], iterations=iterations, warmup=warmup
        ),
    )


def _run_cross_matrix_repeated(
    config: SOFDMLoadConfig,
    *,
    iterations: int,
    warmup: int,
    diagnostics: bool,
) -> CrossLinkMatrixStats:
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    cell_keys = (("fp", "fp"), ("fp", "sofdm"), ("sofdm", "fp"), ("sofdm", "sofdm"))
    stage_names = ("tx_ns", "channel_ns", "dac_ns", "adc_ns", "rx_ns", "total_ns")
    samples: dict[tuple[str, str], dict[str, list[int]]] = {
        key: {stage: [] for stage in stage_names} for key in cell_keys
    }
    for _ in range(warmup):
        run_cross_link_matrix(config, diagnostics=diagnostics)
    last: CrossLinkMatrix | None = None
    for _ in range(iterations):
        m = run_cross_link_matrix(config, diagnostics=diagnostics)
        last = m
        for cell in m.cells():
            key = (cell.tx_kind, cell.rx_kind)
            samples[key]["tx_ns"].append(cell.tx_ns)
            samples[key]["channel_ns"].append(cell.channel_ns)
            samples[key]["dac_ns"].append(cell.dac_ns)
            samples[key]["adc_ns"].append(cell.adc_ns)
            samples[key]["rx_ns"].append(cell.rx_ns)
            samples[key]["total_ns"].append(cell.total_ns)
    assert last is not None
    cells = tuple(
        _aggregate_cross_cell(
            cell,
            per_stage_samples=samples[(cell.tx_kind, cell.rx_kind)],
            iterations=iterations,
            warmup=warmup,
        )
        for cell in last.cells()
    )
    return CrossLinkMatrixStats(
        config=config,
        iterations=iterations,
        warmup=warmup,
        diagnostics=diagnostics,
        cells=cells,
        identity=last.identity,
    )


def run_cross_matrix_diagnostic(
    config: SOFDMLoadConfig, *, iterations: int = 10, warmup: int = 2
) -> CrossLinkMatrixStats:
    """Cross-link 2x2 with EVM/reference decode enabled, repeated."""
    return _run_cross_matrix_repeated(
        config, iterations=iterations, warmup=warmup, diagnostics=True
    )


def run_cross_matrix_hot_path(
    config: SOFDMLoadConfig, *, iterations: int = 50, warmup: int = 5
) -> CrossLinkMatrixStats:
    """Cross-link 2x2 with diagnostics off (hot-path operational only)."""
    return _run_cross_matrix_repeated(
        config, iterations=iterations, warmup=warmup, diagnostics=False
    )


@dataclass(frozen=True, slots=True)
class SOFDMLinkStats:
    """Repeated-run link-load stats wrapper."""

    config: SOFDMLoadConfig
    iterations: int
    warmup: int
    diagnostics: bool
    include_baseline: bool
    bit_errors: int
    bits_total: int
    ber: float
    evm: float | None
    tx_ns: StageStats
    channel_ns: StageStats
    dac_ns: StageStats
    adc_ns: StageStats
    rx_ns: StageStats
    total_ns: StageStats
    baseline_total_ns: StageStats | None

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.config.scenario,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "diagnostics": self.diagnostics,
            "include_baseline": self.include_baseline,
            "bit_errors": self.bit_errors,
            "bits_total": self.bits_total,
            "ber": self.ber,
            "evm": self.evm,
            "tx_ns": self.tx_ns.to_dict(),
            "channel_ns": self.channel_ns.to_dict(),
            "dac_ns": self.dac_ns.to_dict(),
            "adc_ns": self.adc_ns.to_dict(),
            "rx_ns": self.rx_ns.to_dict(),
            "total_ns": self.total_ns.to_dict(),
            "baseline_total_ns": (
                None if self.baseline_total_ns is None else self.baseline_total_ns.to_dict()
            ),
        }


def _run_link_load_once(
    config: SOFDMLoadConfig,
    *,
    diagnostics: bool,
    include_baseline: bool,
) -> tuple[
    dict[str, int],
    SOFDMTXFrame,
    DACResult,
    ADCResult,
    SOFDMReceiverResult,
    int | None,
]:
    """Run the link-load stages once; return stage timings + artifacts.

    ``diagnostics`` gates the reference-symbol decode in TX and the
    equalize/EVM in RX. ``include_baseline`` chooses whether the fp
    comparator arm runs (it is ~18 % of cumtime per profiling and is not
    needed for the hot-path claim).
    """
    tx_ns, tx = _time_ns(
        lambda: build_sofdm_tx_frame(
            config.tx, collect_reference_symbols=diagnostics
        )
    )
    channel_ns, channel_grid = _time_ns(
        lambda: apply_known_channel(tx.grid, config.channel_h, active_subcarriers=config.tx.active_subcarriers)
    )
    dac_ns, dac = _time_ns(
        lambda: simulate_iq_dac(channel_grid, ncp=config.tx.ncp, config=config.dac, validate_grid=False)
    )
    adc_ns, adc = _time_ns(lambda: simulate_iq_adc(dac.iq, config.adc))
    rx_cfg = OFDMRXPlan(
        n_fft=config.tx.n_fft,
        ncp=config.tx.ncp,
        modulation=config.tx.modulation,
        active_subcarriers=config.tx.active_subcarriers,
        channel_policy=ChannelPolicy.KNOWN,
        channel_h=config.channel_h,
    )
    rx_ns, rx = _time_ns(
        lambda: receive_sofdm_frame(
            adc.samples,
            rx_cfg,
            tx_bits=tx.bits,
            reference_symbols=tx.active_symbols if diagnostics else None,
            collect_equalized_symbols=diagnostics,
            collect_evm=diagnostics,
        )
    )
    baseline_ns: int | None = None
    if include_baseline:
        baseline_ns, _ = _time_ns(lambda: run_basic_ofdm_baseline(config, tx.bits))
    total_ns = tx_ns + channel_ns + dac_ns + adc_ns + rx_ns
    stages = {
        "tx_ns": tx_ns,
        "channel_ns": channel_ns,
        "dac_ns": dac_ns,
        "adc_ns": adc_ns,
        "rx_ns": rx_ns,
        "total_ns": total_ns,
    }
    return stages, tx, dac, adc, rx, baseline_ns


def _run_link_load_repeated(
    config: SOFDMLoadConfig,
    *,
    iterations: int,
    warmup: int,
    diagnostics: bool,
    include_baseline: bool,
) -> SOFDMLinkStats:
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    stage_names = ("tx_ns", "channel_ns", "dac_ns", "adc_ns", "rx_ns", "total_ns")
    samples: dict[str, list[int]] = {name: [] for name in stage_names}
    baseline_samples: list[int] = []
    last_rx: SOFDMReceiverResult | None = None
    last_tx_bits: npt.NDArray[np.uint8] | None = None
    for _ in range(warmup):
        _run_link_load_once(
            config, diagnostics=diagnostics, include_baseline=include_baseline
        )
    for _ in range(iterations):
        stages, tx, _dac, _adc, rx, baseline_ns = _run_link_load_once(
            config, diagnostics=diagnostics, include_baseline=include_baseline
        )
        for name in stage_names:
            samples[name].append(stages[name])
        if baseline_ns is not None:
            baseline_samples.append(baseline_ns)
        last_rx = rx
        last_tx_bits = tx.bits
    assert last_rx is not None and last_tx_bits is not None
    bit_errors = int(np.count_nonzero(last_tx_bits != last_rx.rx_bits))
    bits_total = int(last_tx_bits.size)
    ber = float(bit_errors / bits_total) if bits_total else 0.0
    return SOFDMLinkStats(
        config=config,
        iterations=iterations,
        warmup=warmup,
        diagnostics=diagnostics,
        include_baseline=include_baseline,
        bit_errors=bit_errors,
        bits_total=bits_total,
        ber=ber,
        evm=last_rx.evm,
        tx_ns=_stats(samples["tx_ns"], iterations=iterations, warmup=warmup),
        channel_ns=_stats(samples["channel_ns"], iterations=iterations, warmup=warmup),
        dac_ns=_stats(samples["dac_ns"], iterations=iterations, warmup=warmup),
        adc_ns=_stats(samples["adc_ns"], iterations=iterations, warmup=warmup),
        rx_ns=_stats(samples["rx_ns"], iterations=iterations, warmup=warmup),
        total_ns=_stats(samples["total_ns"], iterations=iterations, warmup=warmup),
        baseline_total_ns=(
            _stats(baseline_samples, iterations=iterations, warmup=warmup)
            if baseline_samples
            else None
        ),
    )


def run_link_load_diagnostic(
    config: SOFDMLoadConfig, *, iterations: int = 10, warmup: int = 2
) -> SOFDMLinkStats:
    """Link load with EVM + reference decode + fp baseline arm, repeated."""
    return _run_link_load_repeated(
        config,
        iterations=iterations,
        warmup=warmup,
        diagnostics=True,
        include_baseline=True,
    )


def run_link_load_hot_path(
    config: SOFDMLoadConfig, *, iterations: int = 50, warmup: int = 5
) -> SOFDMLinkStats:
    """Link load with diagnostics off and fp baseline arm omitted."""
    return _run_link_load_repeated(
        config,
        iterations=iterations,
        warmup=warmup,
        diagnostics=False,
        include_baseline=False,
    )


__all__ = [
    "CrossLinkCellStats",
    "CrossLinkMatrixStats",
    "SOFDMLinkStats",
    "StageStats",
    "run_cross_matrix_diagnostic",
    "run_cross_matrix_hot_path",
    "run_link_load_diagnostic",
    "run_link_load_hot_path",
]
