"""CLI orchestrator for Assignment 05 (Phase 1 skeleton).

Thin argparse entrypoint. Only two lightweight subcommands are wired up so far:

    python -m src.run_benchmark hardware   # probe + save hardware profile
    python -m src.run_benchmark dry-run    # write a fake benchmark row + log

No real model inference is implemented yet — baseline/quant/airllm runners are
pending in later phases. (< 150 lines)
"""

import argparse

from . import config
from . import hardware
from . import results_io


def cmd_hardware(_args):
    """Collect and save the hardware profile snapshot."""
    hardware.run()
    return 0


def cmd_dry_run(_args):
    """Write one clearly-fake benchmark row (no LLM inference) + a short log.

    This validates the CSV schema and I/O path end-to-end without downloading
    or loading any model.
    """
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


def build_parser():
    """Build the argparse parser with Phase 1 subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m src.run_benchmark",
        description="Assignment 05 benchmark CLI (Phase 1 skeleton — "
                    "no real model inference yet).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_hw = sub.add_parser("hardware",
                          help="probe CPU/RAM + static profile, save JSON")
    p_hw.set_defaults(func=cmd_hardware)

    p_dry = sub.add_parser("dry-run",
                           help="write a fake benchmark row + log (no model)")
    p_dry.set_defaults(func=cmd_dry_run)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
