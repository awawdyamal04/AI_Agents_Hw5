"""Plotting for Assignment 05.

Reads results/benchmark_results.csv and charts *only* existing rows (never
invents data). Bars are coloured AND tagged by provenance (result_type):
``real`` -> blue [REAL], ``controlled_analysis`` -> orange [NON-REAL /
ESTIMATED], ``mock`` -> grey [MOCK]. ``environment_check`` rows carry no
benchmark numbers and are excluded from the charts. When no ``real`` row exists
a watermark is drawn so nobody mistakes the charts for measurements; once a real
row is present (Phase 4.5) the watermark drops and per-bar tags keep the
non-real bars clearly marked. matplotlib missing -> skip gracefully. (< 150)
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


_COLORS = {
    "real": "#4c72b0",
    "controlled_analysis": "#dd8452",
    "mock": "#b0b0b0",
}

_TAGS = {  # per-bar provenance tag so mixed charts stay unambiguous
    "real": "REAL",
    "controlled_analysis": "NON-REAL / ESTIMATED",
    "mock": "MOCK",
}


def _to_float(value):
    """Best-effort float parse; non-numeric/blank cells become 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _category(row):
    """Classify a row by provenance: real | controlled_analysis | mock."""
    rt = (row.get("result_type") or "").lower()
    if rt in ("controlled_analysis", "environment_check", "mock", "real"):
        return rt
    # Fall back for legacy rows without a result_type column.
    if (row.get("result") or "").lower() == "mock" \
            or (row.get("config") or "").lower() == "dry-run":
        return "mock"
    return "real"


def _watermark(cats):
    """Return watermark/title text for a non-real set, or '' if any real row."""
    kinds = set(cats)
    if "real" in kinds:
        return ""
    if kinds == {"controlled_analysis"}:
        return "CONTROLLED ANALYSIS (estimated — not measured)"
    if kinds == {"mock"}:
        return "MOCK DATA (no real inference)"
    return "NON-REAL (mock / estimated)"


def _label(row, idx, cat=""):
    """Short x-axis label: config name + a provenance tag (REAL/MOCK/…)."""
    cfg = (row.get("config", f"row{idx}") or f"row{idx}")[:34]
    tag = _TAGS.get(cat, "")
    return f"{cfg}\n[{tag}]" if tag else cfg


def _render_one(rows, cats, column, out_path, ylabel):
    """Render a single bar chart for `column` and save it. Returns the path."""
    labels = [_label(r, i, c) for i, (r, c) in enumerate(zip(rows, cats))]
    values = [_to_float(r.get(column)) for r in rows]
    colors = [_COLORS.get(c, "#4c72b0") for c in cats]

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.7), 4.4))
    bars = ax.bar(labels, values, color=colors)
    banner = _watermark(cats)
    title = ylabel + " by configuration"
    if banner:
        title += "  —  " + banner
        ax.text(0.5, 0.5, banner.split(" ")[0], transform=ax.transAxes,
                fontsize=44, color="red", alpha=0.16, ha="center",
                va="center", rotation=20, zorder=0)
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

    all_rows = results_io.load_benchmark_csv(csv_path)
    # environment_check rows carry no benchmark numbers -> exclude from charts.
    rows = [r for r in all_rows if _category(r) != "environment_check"]
    if not rows:
        print("No plottable benchmark rows found — nothing to plot yet.")
        print("Run 'python -m src.run_benchmark dry-run' / 'controlled' or "
              "real benchmarks first.")
        return []

    cats = [_category(r) for r in rows]
    banner = _watermark(cats)
    kinds = ", ".join(sorted(set(cats)))
    excluded = len(all_rows) - len(rows)
    print(f"Plotting {len(rows)} rows (types: {kinds}"
          f"{f'; {excluded} environment_check row(s) excluded' if excluded else ''}).")
    if banner:
        print(f"NOTE: no real rows — charts are labelled '{banner}'.")

    written = []
    for column, out_path, ylabel in _CHARTS:
        path = _render_one(rows, cats, column, out_path, ylabel)
        written.append(path)
        print(f"Wrote {path}")
    return written


if __name__ == "__main__":  # allow: python -m src.plots
    generate()
