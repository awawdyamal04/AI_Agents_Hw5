"""Economic analysis template for Assignment 05 — Phase 2.

Builds a Local vs Cloud-GPU vs API cost-comparison *template* and saves it to
results/economic_analysis.csv.

IMPORTANT: every number here is an openly-labelled ESTIMATE or PLACEHOLDER
based on public list-price assumptions and simple formulas — NOT a measured
benchmark result. The `basis` column states the assumption behind each value
so nothing is mistaken for real data. Real tokens/sec from later benchmarks
can be substituted into the formulas to refine the per-1M-token estimates.
(< 150 lines)
"""

from . import config
from . import results_io

# --- Openly-declared pricing assumptions (list-price estimates) -------------
# These are round, public-order-of-magnitude figures used only to populate the
# template. They are NOT quotes and NOT measurements. Adjust freely.
ASSUMPTIONS = {
    "laptop_hardware_usd": 500,      # est. cost of this used laptop
    "amortize_months": 36,           # est. useful-life horizon
    "power_watts": 45,               # est. avg draw under CPU inference
    "electricity_usd_per_kwh": 0.15,  # est. local tariff
    "cloud_gpu_usd_per_hour": 0.50,  # est. small rented cloud GPU
    "api_usd_per_1m_tokens": 0.50,   # est. hosted small-model output price
    "assumed_tokens_per_s": 5.0,     # PLACEHOLDER throughput (NOT measured)
}

COLUMNS = [
    "option", "upfront_usd", "ongoing_basis", "est_usd_per_1m_tokens",
    "basis", "notes",
]


def _seconds_per_1m_tokens(tps):
    """Seconds to emit 1,000,000 tokens at `tps` tokens/sec (est.)."""
    if tps <= 0:
        return 0.0
    return 1_000_000 / tps


def _local_cost_per_1m(a):
    """Estimated electricity-only marginal cost per 1M tokens (local CPU)."""
    secs = _seconds_per_1m_tokens(a["assumed_tokens_per_s"])
    hours = secs / 3600
    kwh = (a["power_watts"] / 1000) * hours
    return round(kwh * a["electricity_usd_per_kwh"], 4)


def _cloud_cost_per_1m(a):
    """Estimated rented-GPU cost per 1M tokens at assumed throughput."""
    secs = _seconds_per_1m_tokens(a["assumed_tokens_per_s"])
    hours = secs / 3600
    return round(hours * a["cloud_gpu_usd_per_hour"], 4)


def build_rows(a=None):
    """Build the three comparison rows from the ASSUMPTIONS dict."""
    a = a or ASSUMPTIONS
    tps = a["assumed_tokens_per_s"]
    return [
        {
            "option": "Local On-Prem (this laptop)",
            "upfront_usd": a["laptop_hardware_usd"],
            "ongoing_basis": f"electricity ~{a['power_watts']}W @ "
                             f"${a['electricity_usd_per_kwh']}/kWh",
            "est_usd_per_1m_tokens": _local_cost_per_1m(a),
            "basis": f"ESTIMATE: {a['power_watts']}W, assumed {tps} tok/s "
                     f"(PLACEHOLDER, not measured)",
            "notes": "Slow but ~zero marginal cloud/API cost; upfront amortized "
                     f"over ~{a['amortize_months']} months.",
        },
        {
            "option": "Cloud GPU (rented)",
            "upfront_usd": 0,
            "ongoing_basis": f"${a['cloud_gpu_usd_per_hour']}/GPU-hour",
            "est_usd_per_1m_tokens": _cloud_cost_per_1m(a),
            "basis": f"ESTIMATE: ${a['cloud_gpu_usd_per_hour']}/hr, assumed "
                     f"{tps} tok/s (PLACEHOLDER)",
            "notes": "Fast, pay-per-hour; setup overhead; cost scales with use.",
        },
        {
            "option": "API (hosted LLM)",
            "upfront_usd": 0,
            "ongoing_basis": "per-token billing",
            "est_usd_per_1m_tokens": a["api_usd_per_1m_tokens"],
            "basis": "ESTIMATE: public list-price order-of-magnitude "
                     "(PLACEHOLDER)",
            "notes": "Fastest to use; no hardware; ongoing per-call cost.",
        },
    ]


def _write_csv(rows, path):
    """Write economics rows to CSV using results_io's schema-agnostic writer."""
    import csv
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return Path(path)


def generate(path=None):
    """Build the template and save it. Returns the output path."""
    path = path or config.ECONOMICS_CSV
    rows = build_rows()
    out = _write_csv(rows, path)
    print("=== Economic Analysis (ESTIMATED TEMPLATE - not measured) ===")
    for r in rows:
        print(f"{r['option']:<28} ~${r['est_usd_per_1m_tokens']}/1M tokens "
              f"(est.)")
    print("All figures are labelled estimates/placeholders. Substitute real "
          "tokens/sec after benchmarks to refine.")
    print(f"Saved -> {out}")
    return out


if __name__ == "__main__":  # allow: python -m src.economics
    generate()
