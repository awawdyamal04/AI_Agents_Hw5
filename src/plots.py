"""Plotting for Assignment 05 — Phase 2.

Reads results/benchmark_results.csv and generates simple bar charts *only*
from rows that already exist. It never invents data. If the CSV contains only
dry-run / mock rows, they are still plotted but every chart title and the
figure itself are clearly labelled "MOCK DATA" so no reader mistakes them for
real benchmark numbers.

matplotlib is a lightweight, declared dependency; if it is somehow missing we
skip gracefully with a clear message instead of crashing. (< 150 lines)
"""

from . import config
from . import results_io

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: write PNGs without a display
    import matplotlib.pyplot as plt
except ImportError:  # degrade gracefully
    plt = None


# Numeric columns we know how to chart -> (output path, y-axis label).
_CHARTS = [
    ("tokens_per_s", config.PLOT_TOKENS_PER_S, "Tokens / sec"),
    ("load_s", config.PLOT_LOAD_TIME, "Load time (s)"),
    ("peak_ram_mb", config.PLOT_PEAK_RAM, "Peak RAM (MB)"),
]


def _to_float(value):
    """Best-effort float parse; non-numeric/blank cells become 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_mock(rows):
    """True if every row is a mock/dry-run row (or there are no real rows)."""
    return all(r.get("result", "").lower() in ("mock", "")
               or r.get("config", "").lower() == "dry-run" for r in rows)


def _label(row, idx):
    """Short x-axis label combining config + model (deduped by index)."""
    cfg = row.get("config", f"row{idx}")
    model = row.get("model", "")
    return f"{cfg}\n{model}"[:40] if model else cfg


def _render_one(rows, column, out_path, ylabel, mock):
    """Render a single bar chart for `column` and save it. Returns the path."""
    labels = [_label(r, i) for i, r in enumerate(rows)]
    values = [_to_float(r.get(column)) for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.6), 4.2))
    bars = ax.bar(labels, values,
                  color="#b0b0b0" if mock else "#4c72b0")
    title = ylabel + " by configuration"
    if mock:
        title += "  —  MOCK DATA (no real inference)"
        ax.text(0.5, 0.5, "MOCK", transform=ax.transAxes, fontsize=48,
                color="red", alpha=0.18, ha="center", va="center",
                rotation=20, zorder=0)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:g}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def generate(csv_path=None):
    """Generate all charts from existing CSV rows. Returns list of paths.

    Prints a clear status line; produces no files (and returns []) if there
    are no rows yet or matplotlib is unavailable.
    """
    print("=== Plots (from existing CSV rows only) ===")
    if plt is None:
        print("matplotlib not available — skipping plot generation.")
        return []

    rows = results_io.load_benchmark_csv(csv_path)
    if not rows:
        print("No benchmark rows found — nothing to plot yet.")
        print("Run 'python -m src.run_benchmark dry-run' or real benchmarks "
              "first.")
        return []

    mock = _is_mock(rows)
    tag = "MOCK/dry-run rows" if mock else "real benchmark rows"
    print(f"Plotting {len(rows)} {tag}.")
    if mock:
        print("NOTE: all rows are mock - charts are labelled MOCK DATA.")

    written = []
    for column, out_path, ylabel in _CHARTS:
        path = _render_one(rows, column, out_path, ylabel, mock)
        written.append(path)
        print(f"Wrote {path}")
    return written


if __name__ == "__main__":  # allow: python -m src.plots
    generate()
