"""Central constants for Assignment 05.

No implementation logic here — only reproducible settings, paths, the known
static hardware profile, and the results schema. Keeping these in one place
means every module reads the same source of truth. (< 150 lines)
"""

from pathlib import Path

# --- Project paths -----------------------------------------------------------
# config.py lives in src/, so the project root is one directory up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Output artifacts (created on demand by results_io).
HARDWARE_JSON = RESULTS_DIR / "hardware_profile.json"
BENCHMARK_CSV = RESULTS_DIR / "benchmark_results.csv"
DRY_RUN_LOG = RESULTS_DIR / "dry_run.log"

# --- Known static hardware profile ------------------------------------------
# These specs are documented from the physical machine and cannot be reliably
# read at runtime on this OS/Python without heavy deps, so they are recorded
# here as ground truth. Live CPU/RAM values are added at runtime by hardware.py.
STATIC_HARDWARE = {
    "os": "Windows 11 Pro",
    "laptop": "ASUS VivoBook 15 X540UBR",
    "cpu": "Intel Core i7-8550U",
    "ram_gb": 8,
    "igpu": "Intel UHD Graphics 620",
    "dgpu": "NVIDIA GeForce MX110",
    "dgpu_vram_gb_approx": 2,
    "disk": "SanDisk 256GB",
}

# --- Benchmark result schema -------------------------------------------------
# One row per run. Phase 1 only emits mock (dry-run) rows; real runners fill
# the same columns later so the CSV format never changes.
CSV_COLUMNS = [
    "timestamp",
    "config",        # baseline | quant | airllm | dry-run
    "model",
    "params",
    "precision",
    "load_s",
    "ttft_s",
    "tokens_per_s",
    "peak_ram_mb",
    "vram",
    "result",        # success | slow | failure | mock
    "error_reason",
]

# --- Benchmark run settings (used by later phases) --------------------------
PROMPT = "Explain what a KV cache is in one short paragraph."
MAX_NEW_TOKENS = 64
RUNS_PER_CONFIG = 1

# --- Dry-run mock values -----------------------------------------------------
# Clearly-fake numbers so a dry-run row is never mistaken for a real result.
DRY_RUN_ROW = {
    "config": "dry-run",
    "model": "mock-model (no inference)",
    "params": "0",
    "precision": "N/A",
    "load_s": 0.0,
    "ttft_s": 0.0,
    "tokens_per_s": 0.0,
    "peak_ram_mb": 0.0,
    "vram": "N/A",
    "result": "mock",
    "error_reason": "dry-run: no model was loaded or executed",
}
