"""Optional HuggingFace/CPU backend runner for Assignment 05 — Phase 3B.

Checks whether ``transformers`` and ``torch`` can be imported, WITHOUT installing
them and WITHOUT loading or running any model. ``importlib.util.find_spec`` is
used so the check never actually imports the heavy libraries.

- If either dependency is MISSING (the case in this environment), it records a
  single ``environment_check`` row honestly documenting the gap so the reader
  knows a real tiny-CPU HuggingFace run could not happen here.
- If both were PRESENT, this file only points at how a real run *would* be wired
  (load a tiny model such as gpt2 / Qwen2.5-0.5B on CPU); that inference path is
  intentionally left unimplemented for this phase. (< 150 lines)
"""

import importlib.util

from . import base_runner

APPROACH = "huggingface_tiny_cpu_backend"
REQUIRED = ("transformers", "torch")


def missing_deps():
    """Return the list of REQUIRED modules that cannot be found (not imported)."""
    return [name for name in REQUIRED
            if importlib.util.find_spec(name) is None]


def _placeholder_real_run():
    """Document (but do NOT execute) how a real HuggingFace CPU run would happen.

    Left deliberately unimplemented for Phase 3B: no weights are downloaded and
    no forward pass is run. A future real runner would, with the deps installed:
      1. AutoTokenizer/AutoModelForCausalLM.from_pretrained a TINY model on CPU
         (e.g. gpt2, Qwen2.5-0.5B, TinyLlama-1.1B),
      2. time load + first token (TTFT) + decode, count tokens, sample RAM,
      3. append a result_type="real" row via base_runner.make_row(...).
    """
    reason = ("transformers+torch importable. A real tiny-CPU run is POSSIBLE "
              "but intentionally NOT executed in Phase 3B (no download, no "
              "forward pass). See _placeholder_real_run for the wiring.")
    return base_runner.make_row(
        APPROACH, "hf-transformers (importable)",
        result=base_runner.STATUS_SKIPPED,
        result_type=base_runner.TYPE_ENV,
        error_reason=reason)


def run():
    """Probe for transformers/torch; append one row describing what was (not) run."""
    absent = missing_deps()
    print("=== HuggingFace/CPU Backend Check (Phase 3B — no inference) ===")

    if absent:
        row = base_runner.make_row(
            APPROACH, "transformers/torch (absent)",
            result=base_runner.STATUS_UNAVAILABLE,
            result_type=base_runner.TYPE_ENV,
            error_reason="transformers/torch not installed")
        print(f"missing    : {', '.join(absent)}")
        print("status     : unavailable (result_type=environment_check)")
        print("note       : transformers/torch not installed — real HF CPU "
              "backend cannot run here.")
    else:
        row = _placeholder_real_run()
        print("deps       : transformers + torch importable")
        print("status     : skipped (result_type=environment_check)")
        print("note       : present, but real inference is not executed in "
              "Phase 3B (placeholder only).")

    path = base_runner.append_row(row)
    print(f"Appended {APPROACH} row -> {path}")
    return row


if __name__ == "__main__":  # allow: python -m src.runners.hf_runner
    run()
