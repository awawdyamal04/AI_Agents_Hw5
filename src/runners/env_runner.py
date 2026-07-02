"""Environment-check runner for Assignment 05 — Phase 3A (no model inference).

Records a single ``environment_check`` row answering: can this machine run real
local LLM benchmarks right now? It checks three things without loading anything:

- is the ``ollama`` CLI installed (``shutil.which``),
- which Python version is active,
- how much system RAM ``psutil`` reports.

On the current environment Ollama is NOT installed, so the row honestly
documents that limitation and marks real quantized inference as unavailable.
(< 150 lines)
"""

import platform
import shutil

from .. import config
from . import base_runner

try:
    import psutil
except ImportError:  # keep the check runnable without psutil
    psutil = None


def check_ollama():
    """Return the ollama executable path, or None if the CLI is absent."""
    return shutil.which("ollama")


def available_ram_gb():
    """Total system RAM in GB via psutil, or None if psutil is unavailable."""
    if psutil is None:
        return None
    return round(psutil.virtual_memory().total / (1024 ** 3), 2)


def run():
    """Probe the environment, append an environment_check row, print a summary."""
    ollama_path = check_ollama()
    py_version = platform.python_version()
    ram_gb = available_ram_gb()

    ollama_ok = ollama_path is not None
    result = (base_runner.STATUS_SUCCESS if ollama_ok
              else base_runner.STATUS_UNAVAILABLE)
    ram_str = f"{ram_gb} GB" if ram_gb is not None else "unknown (psutil missing)"

    reason = (f"ollama_cli={'found:' + ollama_path if ollama_ok else 'NOT installed'}; "
              f"python={py_version}; total_ram={ram_str}. ")
    if ollama_ok:
        reason += "Ollama present -> real quantized/GGUF runs are possible."
    else:
        reason += ("Ollama unavailable -> real quantized/GGUF inference cannot "
                   "run in this environment; controlled analysis is used "
                   "instead (see 'controlled' command).")

    row = base_runner.make_row(
        "env-check", f"host:{platform.system()} py{py_version}",
        result=result, result_type=base_runner.TYPE_ENV,
        vram=f"~{config.MACHINE_VRAM_GB} GB",
        error_reason=reason)
    path = base_runner.append_row(row)

    print("=== Environment Check (no model inference) ===")
    print(f"ollama CLI : {'FOUND at ' + ollama_path if ollama_ok else 'NOT installed'}")
    print(f"python     : {py_version}")
    print(f"total RAM  : {ram_str}")
    print(f"status     : {result} (result_type=environment_check)")
    print(f"Appended environment_check row -> {path}")
    return row


if __name__ == "__main__":  # allow: python -m src.runners.env_runner
    run()
