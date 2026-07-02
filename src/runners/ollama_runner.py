"""Ollama backend runner for Assignment 05.

- ``run()`` — Phase 3B availability probe (``shutil.which`` only; env-check row,
  no inference).
- ``run_real()`` — Phase 4.5 REAL tiny benchmark. If Ollama and the already
  pulled ``smollm2:135m`` are present, runs ONE real local inference via the CLI,
  measures wall-time / output tokens / tokens-per-sec / best-effort peak RAM, and
  appends a ``result_type="real"`` row. Never installs and never re-pulls; if
  Ollama/model is missing it records an honest ``environment_check`` row.

Tiny model on purpose: 8 GB RAM / ~2 GB VRAM laptop. (< 150 lines)
"""

import re
import shutil
import subprocess
import time

from . import base_runner
from .. import config
from .. import metrics
from .. import results_io

try:
    import psutil
except ImportError:  # best-effort RAM sampling only
    psutil = None

APPROACH = "ollama_quantized_backend"          # Phase 3B check row
REAL_APPROACH = "ollama_smollm2_135m_real"     # Phase 4.5 real row
REAL_MODEL = "smollm2:135m"
REAL_PROMPT = ("Explain in one sentence what quantization means in local LLM "
               "inference.")
REAL_NOTE = "Real local Ollama inference on tiny quantized/on-device model"
REAL_LOG = config.RESULTS_DIR / "ollama_real_run.log"
TIMEOUT_S = 120
# Strip `ollama run` spinner ANSI/braille noise -> readable log + real token count.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|[⠀-⣿]")

def check_ollama():
    """Return the ollama executable path, or None if the CLI is absent."""
    return shutil.which("ollama")

def _model_available(ollama_path):
    """True if smollm2:135m is already pulled (checked via `ollama list`)."""
    try:
        out = subprocess.run([ollama_path, "list"], capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return REAL_MODEL in (out.stdout or "")

def _peak_rss_mb(proc):
    """Best-effort peak RSS (MB) of the ollama process tree; 0.0 w/o psutil."""
    if psutil is None:
        return 0.0
    try:
        p = psutil.Process(proc.pid)
        tree = [p] + p.children(recursive=True)
        return sum(x.memory_info().rss for x in tree if x.is_running()) / 2**20
    except (psutil.Error, OSError):
        return 0.0

def _run_inference(ollama_path):
    """Run one real inference; return (returncode, output, elapsed_s, peak_mb)."""
    cmd = [ollama_path, "run", REAL_MODEL, REAL_PROMPT]
    timer = metrics.Timer().start()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    peak_mb, deadline = 0.0, time.time() + TIMEOUT_S
    while proc.poll() is None:
        peak_mb = max(peak_mb, _peak_rss_mb(proc))
        if time.time() > deadline:
            proc.kill()
            break
        time.sleep(0.1)
    output = proc.stdout.read() if proc.stdout else ""
    elapsed = timer.stop()
    return proc.returncode, output, elapsed, round(peak_mb, 2)

def _record_unavailable(reason):
    """Append an honest environment_check row when the real run cannot happen."""
    row = base_runner.make_row(
        REAL_APPROACH, f"{REAL_MODEL} (not run)",
        result=base_runner.STATUS_UNAVAILABLE,
        result_type=base_runner.TYPE_ENV, error_reason=reason)
    results_io.append_log(f"ollama-real NOT run: {reason}", REAL_LOG)
    path = base_runner.append_row(row)
    print(f"status     : unavailable — {reason}")
    print(f"Appended environment_check row -> {path}")
    return row

def run_real():
    """Phase 4.5: run ONE real tiny Ollama inference (smollm2:135m)."""
    print("=== Ollama REAL tiny benchmark (Phase 4.5 — smollm2:135m) ===")
    ollama_path = check_ollama()
    if ollama_path is None:
        return _record_unavailable("Ollama CLI not installed")
    if not _model_available(ollama_path):
        return _record_unavailable(
            f"{REAL_MODEL} not pulled — refusing to download (no re-pull)")

    rc, output, elapsed, peak_mb = _run_inference(ollama_path)
    clean = " ".join(_ANSI.sub("", output).split())
    results_io.append_log(
        f"cmd: ollama run {REAL_MODEL} <prompt>\nprompt: {REAL_PROMPT}\n"
        f"returncode: {rc}\nelapsed_s: {elapsed:.3f}\npeak_ram_mb: {peak_mb}\n"
        f"--- model output (ansi-stripped) ---\n{clean}\n", REAL_LOG)
    n_tokens = metrics.count_tokens(clean)
    tps = metrics.tokens_per_second(n_tokens, elapsed)
    status = base_runner.STATUS_SUCCESS if rc == 0 else base_runner.STATUS_FAILED
    row = base_runner.make_row(
        REAL_APPROACH, REAL_MODEL, result=status,
        result_type=base_runner.TYPE_REAL, params="135M",
        precision="on-device", load_s=elapsed, ttft_s=0.0,
        tokens_per_s=tps, peak_ram_mb=peak_mb, vram="~2 GB",
        error_reason=REAL_NOTE)
    path = base_runner.append_row(row)
    print(f"ollama CLI : {ollama_path}  (model already pulled; NOT re-downloaded)")
    print(f"returncode : {rc} -> {status}   total_s: {elapsed:.3f}")
    print(f"output_tokens~={n_tokens}  tok/s~={tps}  peak_ram_mb={peak_mb}")
    print(f"log -> {REAL_LOG}")
    print(f"Appended {REAL_APPROACH} REAL row -> {path}")
    return row

def run():
    """Phase 3B probe: is the ollama CLI installed? (records env-check row)."""
    ollama_path = check_ollama()
    print("=== Ollama Backend Check (Phase 3B — no inference) ===")
    if ollama_path is None:
        model, result, reason = ("ollama-cli (absent)",
                                 base_runner.STATUS_UNAVAILABLE,
                                 "Ollama CLI not installed")
    else:
        model, result = f"ollama-cli:{ollama_path}", base_runner.STATUS_SKIPPED
        reason = (f"ollama_cli=found:{ollama_path}. Availability probe only; the "
                  "REAL tiny run is the separate 'ollama-real' command.")
    row = base_runner.make_row(APPROACH, model, result=result,
                               result_type=base_runner.TYPE_ENV,
                               error_reason=reason)
    print(f"ollama CLI : {'FOUND at ' + ollama_path if ollama_path else 'NOT installed'}")
    path = base_runner.append_row(row)
    print(f"Appended {APPROACH} row -> {path}")
    return row

if __name__ == "__main__":
    run_real()
