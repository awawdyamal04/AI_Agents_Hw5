"""CLI orchestrator for Assignment 05.

Thin argparse entrypoint. Phase 2 wires up the measurement-infrastructure
subcommands (no real model inference is implemented yet):

    python -m src.run_benchmark hardware    # probe + save hardware profile
    python -m src.run_benchmark dry-run     # write a mock benchmark row + log
    python -m src.run_benchmark plots       # plot existing CSV rows -> PNGs
    python -m src.run_benchmark economics    # cost-comparison template CSV
    python -m src.run_benchmark verify      # structural PASS/FAIL checks

Baseline/quant/airllm runners remain pending for a later phase. (< 150 lines)
"""

import argparse

from . import config
from . import economics
from . import hardware
from . import plots
from . import results_io
from . import verify


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


def build_parser():
    """Build the argparse parser with the Phase 2 subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m src.run_benchmark",
        description="Assignment 05 benchmark CLI (Phase 2 measurement core — "
                    "no real model inference yet).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_hw = sub.add_parser("hardware",
                          help="probe CPU/RAM + static profile, save JSON")
    p_hw.set_defaults(func=cmd_hardware)

    p_dry = sub.add_parser("dry-run",
                           help="write a fake benchmark row + log (no model)")
    p_dry.set_defaults(func=cmd_dry_run)

    p_plot = sub.add_parser("plots",
                            help="plot existing CSV rows to results/*.png")
    p_plot.set_defaults(func=cmd_plots)

    p_eco = sub.add_parser("economics",
                           help="write estimated cost-comparison template CSV")
    p_eco.set_defaults(func=cmd_economics)

    p_ver = sub.add_parser("verify",
                           help="structural checks + line limits (PASS/FAIL)")
    p_ver.set_defaults(func=cmd_verify)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
