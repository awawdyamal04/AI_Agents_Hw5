"""Hardware probe for Assignment 05.

Collects live CPU/RAM via psutil and merges it with the known static hardware
profile from config.py. Produces a plain dict and writes it to
results/hardware_profile.json. No model logic here. (< 150 lines)
"""

import platform
from datetime import datetime

from . import config
from . import results_io

try:
    import psutil
except ImportError:  # psutil is lightweight but recorded as optional here.
    psutil = None


def _bytes_to_gb(n):
    """Convert bytes to GB rounded to 2 decimals."""
    return round(n / (1024 ** 3), 2)


def collect_live():
    """Collect live CPU/RAM metrics via psutil.

    Returns a dict. If psutil is unavailable, returns a stub noting that so
    the command still succeeds instead of crashing.
    """
    if psutil is None:
        return {"psutil_available": False,
                "note": "psutil not installed; live CPU/RAM not collected"}

    vm = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    return {
        "psutil_available": True,
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": round(freq.current, 1) if freq else None,
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "ram_total_gb": _bytes_to_gb(vm.total),
        "ram_available_gb": _bytes_to_gb(vm.available),
        "ram_used_percent": vm.percent,
    }


def collect():
    """Build the full hardware snapshot: static profile + live metrics + env."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "static_profile": config.STATIC_HARDWARE,
        "live": collect_live(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }


def run():
    """Collect the snapshot, save it as JSON, and print a short summary.

    Returns the snapshot dict (handy for tests / later phases).
    """
    snapshot = collect()
    path = results_io.save_json(snapshot, config.HARDWARE_JSON)

    static = snapshot["static_profile"]
    live = snapshot["live"]
    print("=== Hardware Profile ===")
    print(f"OS      : {static['os']}")
    print(f"Laptop  : {static['laptop']}")
    print(f"CPU     : {static['cpu']}")
    print(f"RAM     : {static['ram_gb']} GB (static spec)")
    print(f"iGPU    : {static['igpu']}")
    print(f"dGPU    : {static['dgpu']} (~{static['dgpu_vram_gb_approx']} GB VRAM)")
    print(f"Disk    : {static['disk']}")
    if live.get("psutil_available"):
        print("--- live (psutil) ---")
        print(f"Cores   : {live['cpu_physical_cores']} physical / "
              f"{live['cpu_logical_cores']} logical")
        print(f"RAM live: {live['ram_available_gb']} GB free of "
              f"{live['ram_total_gb']} GB ({live['ram_used_percent']}% used)")
    else:
        print("(psutil unavailable — live CPU/RAM skipped)")
    print(f"\nSaved snapshot -> {path}")
    return snapshot


if __name__ == "__main__":  # allow: python -m src.hardware
    run()
