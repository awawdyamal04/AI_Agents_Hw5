"""Shared runner base for Assignment 05 — Phase 3A.

Holds the run-status constants, the row-provenance constants, and a small
schema helper so every runner (environment check, controlled analysis, and any
future *real* runner) writes rows keyed to the same config.CSV_COLUMNS.

This module runs NO model inference and never fabricates a measurement: rows it
helps build are always tagged with an honest ``result_type``. (< 150 lines)
"""

from .. import config
from .. import metrics
from .. import results_io

# --- Run status values (stored in the CSV `result` column) ------------------
STATUS_SUCCESS = "success"          # ran / would run as expected
STATUS_FAILED = "failed"            # errored or expected to fail (e.g. OOM)
STATUS_SKIPPED = "skipped"          # deliberately not attempted
STATUS_UNAVAILABLE = "unavailable"  # prerequisite missing (e.g. no Ollama)

# --- Row provenance (stored in the CSV `result_type` column) ----------------
# Controlled / environment rows must NEVER be labelled "real".
TYPE_REAL = "real"
TYPE_CONTROLLED = "controlled_analysis"
TYPE_ENV = "environment_check"


def make_row(config_name, model, *, result, result_type, params="",
             precision="", load_s=0.0, ttft_s=0.0, tokens_per_s=0.0,
             peak_ram_mb=0.0, vram="N/A", error_reason=""):
    """Build one CSV row via metrics, tagged with its provenance result_type.

    Thin wrapper so the runners in this package all produce rows the same way
    and cannot forget the result_type tag.
    """
    return metrics.build_result_row(
        config_name, model, params=params, precision=precision,
        load_s=load_s, ttft_s=ttft_s, tokens_per_s=tokens_per_s,
        peak_ram_mb=peak_ram_mb, vram=vram, result=result,
        result_type=result_type, error_reason=error_reason)


def append_row(row, csv_path=None):
    """Persist a runner row to the benchmark CSV. Returns the path written."""
    return results_io.append_csv_row(row, csv_path or config.BENCHMARK_CSV)
