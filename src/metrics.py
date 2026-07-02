"""Measurement helpers for Assignment 05 — Phase 2 measurement core.

Pure, dependency-light utilities used by the (future) model runners:
- timing helpers (wall-clock context manager + a Timer),
- a peak-RAM sampler backed by psutil (process RSS),
- whitespace token-count approximation,
- tokens/sec calculation,
- a benchmark result-row builder that matches config.CSV_COLUMNS.

No model logic and no heavy deps live here so runners can import it cheaply.
Every metric produced here is a *real measurement* of whatever code it wraps;
it never fabricates benchmark numbers. (< 150 lines)
"""

import time
from contextlib import contextmanager

from . import config

try:
    import psutil
except ImportError:  # keep metrics importable even without psutil
    psutil = None


# --- Timing ------------------------------------------------------------------
class Timer:
    """Minimal start/stop wall-clock timer. Use .start() then .stop()."""

    def __init__(self):
        self._t0 = None
        self.elapsed_s = 0.0

    def start(self):
        self._t0 = time.perf_counter()
        return self

    def stop(self):
        if self._t0 is None:
            raise RuntimeError("Timer.stop() called before start()")
        self.elapsed_s = time.perf_counter() - self._t0
        return self.elapsed_s


@contextmanager
def timed():
    """Context manager yielding a Timer; elapsed_s is filled on exit.

    Usage:
        with timed() as t:
            do_work()
        print(t.elapsed_s)
    """
    t = Timer().start()
    try:
        yield t
    finally:
        t.stop()


# --- RAM sampling ------------------------------------------------------------
class RamSampler:
    """Track peak process RSS (resident memory) in MB across manual samples.

    Call .sample() repeatedly during a workload (e.g. inside a decode loop);
    .peak_mb holds the largest reading seen. Returns 0.0 if psutil is missing
    so callers degrade gracefully instead of crashing.
    """

    def __init__(self):
        self._proc = psutil.Process() if psutil is not None else None
        self.peak_mb = 0.0

    def sample(self):
        if self._proc is None:
            return 0.0
        rss_mb = self._proc.memory_info().rss / (1024 ** 2)
        if rss_mb > self.peak_mb:
            self.peak_mb = rss_mb
        return rss_mb


def current_ram_mb():
    """One-shot current process RSS in MB (0.0 if psutil unavailable)."""
    if psutil is None:
        return 0.0
    return round(psutil.Process().memory_info().rss / (1024 ** 2), 2)


# --- Token metrics -----------------------------------------------------------
def count_tokens(text):
    """Approximate token count by whitespace splitting.

    Deliberately simple: real tokenizers differ, but for throughput sanity
    checks a whitespace count is a transparent, model-agnostic proxy.
    """
    if not text:
        return 0
    return len(text.split())


def tokens_per_second(num_tokens, elapsed_s):
    """tokens/sec = tokens / seconds; 0.0 if elapsed is non-positive."""
    if not elapsed_s or elapsed_s <= 0:
        return 0.0
    return round(num_tokens / elapsed_s, 3)


# --- Result row builder ------------------------------------------------------
def build_result_row(config_name, model, *, params="", precision="",
                     load_s=0.0, ttft_s=0.0, tokens_per_s=0.0,
                     peak_ram_mb=0.0, vram="N/A", result="success",
                     result_type="real", error_reason=""):
    """Assemble one benchmark row keyed exactly to config.CSV_COLUMNS.

    Kept in metrics so runners build rows the same way. `result_type` records
    provenance (real | mock | controlled_analysis | environment_check) so an
    estimate is never mistaken for a measurement. `timestamp` is left for
    results_io to stamp at write time.
    """
    row = {
        "config": config_name,
        "model": model,
        "params": params,
        "precision": precision,
        "load_s": round(float(load_s), 3),
        "ttft_s": round(float(ttft_s), 3),
        "tokens_per_s": round(float(tokens_per_s), 3),
        "peak_ram_mb": round(float(peak_ram_mb), 2),
        "vram": vram,
        "result": result,
        "result_type": result_type,
        "error_reason": error_reason,
    }
    # Guard: only emit known columns (minus timestamp, added at write time).
    return {k: row[k] for k in config.CSV_COLUMNS if k in row}
