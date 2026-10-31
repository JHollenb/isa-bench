"""Measured process CPU, wall time and sampled resident memory, with explicit scopes."""
from __future__ import annotations

import platform
import resource
import threading
import time

import psutil


def lifetime_peak_mb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024 if platform.system() == "Darwin" else 1024)


class Usage:
    def __init__(self, sample_seconds=0.01):
        self.sample_seconds = sample_seconds
        self.process = psutil.Process()
        self.done = threading.Event()
        self.peak = 0

    def _sample(self):
        while not self.done.wait(self.sample_seconds):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.rss_start = self.process.memory_info().rss
        self.peak = self.rss_start
        self.wall_start, self.cpu_start = time.perf_counter(), time.process_time()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        wall, cpu = time.perf_counter() - self.wall_start, time.process_time() - self.cpu_start
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.done.set()
        self.thread.join()
        self.result = {"wall_s": wall, "cpu_s": cpu,
                       "cpu_core_equivalent_percent": 100 * cpu / wall if wall else 0,
                       "rss_start_mb": self.rss_start / 2**20,
                       "rss_end_mb": self.process.memory_info().rss / 2**20,
                       "sampled_peak_rss_mb": self.peak / 2**20,
                       "sampled_peak_delta_mb": (self.peak - self.rss_start) / 2**20,
                       "process_lifetime_peak_rss_mb": lifetime_peak_mb(),
                       "rss_sampling_interval_s": self.sample_seconds}


def stamp():
    return time.perf_counter(), time.process_time()


def elapsed(start):
    return {"wall_s": time.perf_counter() - start[0], "cpu_s": time.process_time() - start[1]}
