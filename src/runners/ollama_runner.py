"""Optional Ollama backend runner for Assignment 05 — Phase 3B (no inference).

Checks whether the ``ollama`` CLI is installed (``shutil.which``). It does NOT
download models, does NOT start the Ollama server, and does NOT run inference.

- If Ollama is ABSENT (the case in this environment), it records a single
  ``environment_check`` row honestly documenting the missing dependency so the
  reader knows real quantized/GGUF inference could not run here.
- If Ollama were PRESENT, this file only points at how a real run *would* be
  wired (a local HTTP call to the Ollama API) — that heavy path is intentionally
  left unimplemented for this phase. (< 150 lines)
"""

import shutil

from . import base_runner

APPROACH = "ollama_quantized_backend"


def check_ollama():
    """Return the ollama executable path, or None if the CLI is absent."""
    return shutil.which("ollama")


def _placeholder_real_run(ollama_path):
    """Document (but do NOT execute) how a real Ollama run would happen.

    Left deliberately unimplemented for Phase 3B: no model is pulled and no
    network/inference call is made. A future real runner would, on capable
    hardware with Ollama installed:
      1. ensure a small quantized model is pulled (e.g. `ollama pull qwen2:0.5b`),
      2. POST the prompt to the local API at http://localhost:11434/api/generate,
      3. time load + first token (TTFT) + decode, count tokens, sample RAM,
      4. append a result_type="real" row via base_runner.make_row(...).
    """
    reason = (f"ollama_cli=found:{ollama_path}. Real quantized/GGUF inference "
              "is POSSIBLE but intentionally NOT executed in Phase 3B (no model "
              "pull, no server call). See _placeholder_real_run for the wiring.")
    return base_runner.make_row(
        APPROACH, f"ollama-cli:{ollama_path}",
        result=base_runner.STATUS_SKIPPED,
        result_type=base_runner.TYPE_ENV,
        error_reason=reason)


def run():
    """Probe for Ollama; append one row describing what was (not) run."""
    ollama_path = check_ollama()
    print("=== Ollama Backend Check (Phase 3B — no inference) ===")

    if ollama_path is None:
        row = base_runner.make_row(
            APPROACH, "ollama-cli (absent)",
            result=base_runner.STATUS_UNAVAILABLE,
            result_type=base_runner.TYPE_ENV,
            error_reason="Ollama CLI not installed")
        print("ollama CLI : NOT installed")
        print("status     : unavailable (result_type=environment_check)")
        print("note       : Ollama CLI not installed — real quantized backend "
              "cannot run here.")
    else:
        row = _placeholder_real_run(ollama_path)
        print(f"ollama CLI : FOUND at {ollama_path}")
        print("status     : skipped (result_type=environment_check)")
        print("note       : present, but real inference is not executed in "
              "Phase 3B (placeholder only).")

    path = base_runner.append_row(row)
    print(f"Appended {APPROACH} row -> {path}")
    return row


if __name__ == "__main__":  # allow: python -m src.runners.ollama_runner
    run()
