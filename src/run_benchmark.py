"""CLI orchestrator for Assignment 05.

Thin argparse entrypoint. No real model inference is implemented; every
subcommand either probes the environment or records clearly-labelled non-real
rows. The available subcommands (name + help) are defined in the COMMANDS
table below and shown by ``python -m src.run_benchmark -h``.

Phase 3B adds the optional real-backend checks (``ollama-check``, ``hf-check``,
``airllm-check``, and ``backend-checks`` for all three): each safely records an
environment_check row explaining why the dependency-heavy path did NOT run —
nothing is installed, downloaded, or inferred. (< 150 lines)
"""

import argparse

from . import config
from . import economics
from . import hardware
from . import plots
from . import results_io
from . import verify
from .runners import airllm_optional_runner
from .runners import controlled_runner
from .runners import env_runner
from .runners import hf_runner
from .runners import ollama_runner


def cmd_hardware(_args):
    """Collect and save the hardware profile snapshot."""
    hardware.run()
    return 0


def cmd_dry_run(_args):
    """Write one clearly-fake benchmark row (no LLM inference) + a short log."""
    row = dict(config.DRY_RUN_ROW)
    csv_path = results_io.append_csv_row(row, config.BENCHMARK_CSV)

    results_io.append_log("Dry-run started (no model inference).",
                          config.DRY_RUN_LOG)
    results_io.append_log(
        f"Appended mock row: config={row['config']} model={row['model']} "
        f"result={row['result']}", config.DRY_RUN_LOG)
    log_path = results_io.append_log(
        f"Wrote CSV row to {csv_path}. Dry-run complete.", config.DRY_RUN_LOG)

    print("=== Dry Run (mock, no inference) ===")
    print(f"Appended mock benchmark row -> {csv_path}")
    print(f"Wrote log                   -> {log_path}")
    print("NOTE: this row is fake (result='mock'). No real model was run.")
    return 0


def cmd_env_check(_args):
    """Record an environment_check row (Ollama presence, Python, RAM)."""
    env_runner.run()
    return 0


def cmd_controlled(_args):
    """Record controlled_analysis rows for the 7B baseline/quant/AirLLM configs."""
    controlled_runner.run()
    return 0


def cmd_ollama_check(_args):
    """Phase 3B: probe for the ollama CLI; record availability (no inference)."""
    ollama_runner.run()
    return 0


def cmd_hf_check(_args):
    """Phase 3B: probe for transformers/torch; record availability (no inference)."""
    hf_runner.run()
    return 0


def cmd_airllm_check(_args):
    """Phase 3B: probe for airllm; record availability (no inference)."""
    airllm_optional_runner.run()
    return 0


def cmd_backend_checks(_args):
    """Phase 3B: run all three optional backend checks in one go."""
    print("### Backend availability checks (Phase 3B — no model inference) ###")
    for fn in (ollama_runner.run, hf_runner.run, airllm_optional_runner.run):
        fn()
        print()
    return 0


def cmd_plots(_args):
    """Generate plots from whatever rows already exist in the CSV."""
    plots.generate()
    return 0


def cmd_economics(_args):
    """Write the estimated local/cloud/API cost-comparison template."""
    economics.generate()
    return 0


def cmd_verify(_args):
    """Run structural PASS/FAIL checks and return its exit code."""
    return verify.run()


# name -> (help text, handler). Kept as a table so the parser stays compact.
COMMANDS = [
    ("hardware", "probe CPU/RAM + static profile, save JSON", cmd_hardware),
    ("dry-run", "write a fake benchmark row + log (no model)", cmd_dry_run),
    ("env-check", "record environment_check row (Ollama? RAM?)", cmd_env_check),
    ("controlled", "record controlled_analysis rows (estimates)", cmd_controlled),
    ("ollama-check", "Phase 3B: is the ollama CLI installed?", cmd_ollama_check),
    ("hf-check", "Phase 3B: transformers/torch importable?", cmd_hf_check),
    ("airllm-check", "Phase 3B: is airllm importable?", cmd_airllm_check),
    ("backend-checks", "Phase 3B: run all optional backend checks", cmd_backend_checks),
    ("plots", "plot existing CSV rows to results/*.png", cmd_plots),
    ("economics", "write estimated cost-comparison template CSV", cmd_economics),
    ("verify", "structural checks + line limits (PASS/FAIL)", cmd_verify),
]


def build_parser():
    """Build the argparse parser from the COMMANDS table."""
    parser = argparse.ArgumentParser(
        prog="python -m src.run_benchmark",
        description="Assignment 05 benchmark CLI (Phase 3B: optional backend "
                    "availability checks — no real model inference).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text, func in COMMANDS:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
