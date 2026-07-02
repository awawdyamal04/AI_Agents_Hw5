"""Optional AirLLM backend runner for Assignment 05 — Phase 3B (no inference).

Checks whether ``airllm`` can be imported, WITHOUT installing it and WITHOUT
loading or streaming any model. ``importlib.util.find_spec`` is used so the
check never actually imports the library or triggers a download.

- If AirLLM is MISSING (the case in this environment), it records a single
  ``environment_check`` row honestly documenting the gap and pointing the reader
  to the controlled AirLLM analysis (``controlled`` command), which estimates
  layer-streaming memory from transparent formulas instead of measuring it.
- If AirLLM were PRESENT, this file only points at how a real layer-streaming
  run *would* be wired; that path is intentionally left unimplemented for this
  phase. (< 150 lines)
"""

import importlib.util

from . import base_runner

APPROACH = "airllm_optional_backend"


def airllm_available():
    """True if the airllm module can be found (it is NOT imported/executed)."""
    return importlib.util.find_spec("airllm") is not None


def _placeholder_real_run():
    """Document (but do NOT execute) how a real AirLLM run would happen.

    Left deliberately unimplemented for Phase 3B: no weights are downloaded and
    no layer streaming occurs. A future real runner would, with airllm installed
    on capable storage/hardware:
      1. AirLLMModel(...) to stream one transformer layer at a time from disk,
      2. time load + first token (TTFT) + decode, sample peak RAM (~one layer),
      3. append a result_type="real" row via base_runner.make_row(...).
    """
    reason = ("airllm importable. A real layer-streaming run is POSSIBLE but "
              "intentionally NOT executed in Phase 3B (no download, no "
              "streaming). See _placeholder_real_run for the wiring.")
    return base_runner.make_row(
        APPROACH, "airllm (importable)",
        result=base_runner.STATUS_SKIPPED,
        result_type=base_runner.TYPE_ENV,
        error_reason=reason)


def run():
    """Probe for AirLLM; append one row describing what was (not) run."""
    available = airllm_available()
    print("=== AirLLM Backend Check (Phase 3B — no inference) ===")

    if not available:
        row = base_runner.make_row(
            APPROACH, "airllm (absent)",
            result=base_runner.STATUS_UNAVAILABLE,
            result_type=base_runner.TYPE_ENV,
            error_reason=("AirLLM not installed; controlled AirLLM analysis is "
                          "used instead"))
        print("airllm     : NOT installed")
        print("status     : unavailable (result_type=environment_check)")
        print("note       : AirLLM not installed — controlled AirLLM analysis "
              "is used instead (see 'controlled' command).")
    else:
        row = _placeholder_real_run()
        print("airllm     : importable")
        print("status     : skipped (result_type=environment_check)")
        print("note       : present, but real streaming is not executed in "
              "Phase 3B (placeholder only).")

    path = base_runner.append_row(row)
    print(f"Appended {APPROACH} row -> {path}")
    return row


if __name__ == "__main__":  # allow: python -m src.runners.airllm_optional_runner
    run()
