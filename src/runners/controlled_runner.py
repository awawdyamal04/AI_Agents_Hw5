"""Controlled-analysis runner for Assignment 05 — Phase 3A (no model inference).

Generates clearly-labelled ``controlled_analysis`` rows for three 7B configs.
It does NOT download weights, install Ollama, or run any LLM. Instead it
estimates each config's memory footprint from transparent formulas and compares
it against this laptop's real capacity (RAM=8 GB, VRAM=~2 GB), then records the
*expected* outcome and an explanatory note. Every number here is an openly
labelled estimate, never a measurement. (< 150 lines)
"""

from .. import config
from . import base_runner

PARAMS_B = 7.0  # analysed model size, in billions of parameters
_NUM_LAYERS = 32  # typical 7B transformer depth, used only for the AirLLM estimate


# --- Memory formulas (bytes-per-param -> GB of weights) ---------------------
def fp16_weight_gb(params_billion):
    """FP16 stores 2 bytes/param: params(B) * 2 = GB of weights."""
    return params_billion * 2


def int8_weight_gb(params_billion):
    """INT8 stores 1 byte/param: params(B) * 1 = GB of weights."""
    return params_billion * 1


def int4_weight_gb(params_billion):
    """INT4 stores ~0.5 byte/param: params(B) * 0.5 = GB of weights."""
    return params_billion * 0.5


def _fits_ram(gb):
    """True if an estimated footprint fits inside the machine's RAM."""
    return gb <= config.MACHINE_RAM_GB


def _scenarios():
    """Build the three analysis scenarios (as dicts) for this machine."""
    ram = config.MACHINE_RAM_GB
    vram = config.MACHINE_VRAM_GB
    fp16 = fp16_weight_gb(PARAMS_B)
    int4 = int4_weight_gb(PARAMS_B)
    # AirLLM peak ~ one fp16 layer resident at a time + activations headroom.
    airllm_peak = round(fp16 / _NUM_LAYERS + 1.0, 2)

    return [
        {
            "config": "baseline_fp16_7b",
            "model": "generic-7B (fp16, controlled analysis)",
            "precision": "fp16",
            "peak_gb": fp16,
            "result": base_runner.STATUS_FAILED,
            "note": (f"fp16 weights = {PARAMS_B:g}B x 2 = {fp16:g} GB. "
                     f"Machine RAM={ram} GB, VRAM=~{vram} GB. "
                     f"{fp16:g} GB > {ram} GB RAM and >> {vram} GB VRAM -> "
                     "expected OOM / model too large to load. A real run would "
                     "fail on this hardware (this failure IS the result)."),
        },
        {
            "config": "quantized_int4_7b",
            "model": "generic-7B (int4/GGUF, controlled analysis)",
            "precision": "int4",
            "peak_gb": int4,
            "result": base_runner.STATUS_SUCCESS,
            "note": (f"int4 weights = {PARAMS_B:g}B x 0.5 = {int4:g} GB "
                     f"(fp16 would be {fp16:g} GB). {int4:g} GB < {ram} GB RAM "
                     f"-> may fit in system RAM. Only ~{vram} GB VRAM, so "
                     "inference runs on CPU and is expected to be SLOW (low "
                     "tokens/sec). No throughput number is claimed here — it is "
                     "not measured, only estimated as feasible-but-slow."),
        },
        {
            "config": "airllm_layer_streaming_7b",
            "model": "generic-7B (AirLLM layer streaming, controlled analysis)",
            "precision": "fp16-streamed",
            "peak_gb": airllm_peak,
            "result": base_runner.STATUS_SUCCESS,
            "note": (f"AirLLM streams one of ~{_NUM_LAYERS} transformer layers "
                     f"at a time from disk, so peak memory (est. ~{airllm_peak:g} "
                     f"GB) is far below the full {fp16:g} GB fp16 footprint and "
                     f"fits in {ram} GB RAM. Cost: every token re-reads layer "
                     "weights from disk -> high disk-I/O latency, so it is "
                     "expected to run but be EXTREMELY slow. Controlled "
                     "analysis only; not executed."),
        },
    ]


def run():
    """Append one controlled_analysis row per scenario; print a summary."""
    print("=== Controlled Analysis (estimates only — no model inference) ===")
    print(f"Machine limits: RAM={config.MACHINE_RAM_GB} GB, "
          f"VRAM=~{config.MACHINE_VRAM_GB} GB. Analysed size: {PARAMS_B:g}B params.")
    rows = []
    for s in _scenarios():
        peak_mb = round(s["peak_gb"] * 1024, 2)
        row = base_runner.make_row(
            s["config"], s["model"],
            result=s["result"], result_type=base_runner.TYPE_CONTROLLED,
            params=f"{PARAMS_B:g}B", precision=s["precision"],
            peak_ram_mb=peak_mb,
            vram=f"~{config.MACHINE_VRAM_GB} GB avail",
            error_reason=s["note"])
        path = base_runner.append_row(row)
        rows.append(row)
        fits = "fits RAM" if _fits_ram(s["peak_gb"]) else "exceeds RAM"
        print(f"  - {s['config']:<28} est_peak={s['peak_gb']:>5g} GB "
              f"({fits}) -> {s['result']}")
    print(f"Appended {len(rows)} controlled_analysis rows -> "
          f"{config.BENCHMARK_CSV}")
    return rows


if __name__ == "__main__":  # allow: python -m src.runners.controlled_runner
    run()
