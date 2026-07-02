# PLAN — Technical Plan & Architecture

This plan defines the modular architecture for Assignment 05. **No implementation code is written yet** — this document specifies *what will be built*.

Hard rules:
- All Python source lives in `src/`.
- Every Python file is **< 150 lines**.
- No single giant file — responsibilities are split into small modules.
- The project runs **from the terminal**.
- Results, plots, logs, CSVs go in `results/`.

---

## 1. Modular Architecture (Overview)
```
Idea → PRD → Plan → TODO → Verify → Execute → Push
                                   │
                                   ▼
           ┌─────────────────────────────────────────┐
           │  CLI entrypoint (run_benchmark.py)        │
           └───────────────┬───────────────────────────┘
                           │
     ┌─────────────┬───────┴────────┬──────────────┬───────────────┐
     ▼             ▼                ▼              ▼               ▼
 hardware.py   metrics.py     runners/*      results_io.py     plots.py
 (probe HW)   (timers/mem)  (model backends) (CSV/log save)  (matplotlib)
```

---

## 2. File & Folder Structure (Target)
```
AI_Agents_HW5/
├── prd.md
├── plan.md
├── todo.md
├── README.md                 # final technical report
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py             # models, prompts, run settings (constants)
│   ├── hardware.py           # collect CPU/RAM/GPU/OS info via psutil
│   ├── metrics.py            # timers, TTFT, tokens/sec, RAM sampler
│   ├── results_io.py         # write/append CSV, write logs, load results
│   ├── plots.py              # build comparison plots (PNG)
│   ├── economics.py          # local vs cloud GPU vs API cost model
│   ├── report.py             # aggregate CSV -> summary tables (pandas)
│   ├── run_benchmark.py      # CLI orchestrator (argparse subcommands)
│   └── runners/
│       ├── __init__.py
│       ├── base_runner.py    # abstract runner interface + shared timing
│       ├── baseline_runner.py# HF transformers CPU baseline (small model)
│       ├── quant_runner.py   # GGUF/quantized via llama-cpp / Ollama
│       └── airllm_runner.py  # AirLLM run OR controlled documented analysis
├── models/                   # (gitignored) downloaded weights
└── results/
    ├── *.csv                 # benchmark rows
    ├── *.png                 # plots
    └── *.log                 # raw run logs / errors
```

---

## 3. Explanation of Each Future Module
- **config.py** — Central constants: model list (id, backend, size), benchmark prompts, `max_new_tokens`, `runs_per_config`, output paths. Keeps everything reproducible. (< 150 lines)
- **hardware.py** — Uses `psutil` for CPU count, total/available RAM; static dict for GPU/VRAM specs; OS/Python version. Returns a dict + writes a hardware snapshot. (< 150 lines)
- **metrics.py** — Context-manager timers; TTFT capture hook; tokens/sec computation; background RAM peak sampler thread. Pure measurement, no model logic. (< 150 lines)
- **results_io.py** — Append a run-result dict as a CSV row; create logs; load CSVs for reporting. Single source of truth for I/O. (< 150 lines)
- **plots.py** — Reads results CSV via pandas, produces PNGs (tokens/sec bar chart, load-time chart, RAM usage chart). (< 150 lines)
- **economics.py** — Cost model: local amortized (hardware + electricity), cloud GPU hourly, API per-token. Emits a comparison table CSV. (< 150 lines)
- **report.py** — Aggregates raw CSV into mean/min/max summary per configuration; prints markdown tables. (< 150 lines)
- **runners/base_runner.py** — Defines `Runner` interface: `load()`, `generate(prompt)`, `teardown()`, plus shared timing wrappers and standardized result schema. (< 150 lines)
- **runners/baseline_runner.py** — Loads a small HF model on CPU (FP32/FP16), runs generation, records metrics; catches OOM/errors. (< 150 lines)
- **runners/quant_runner.py** — Runs a quantized GGUF model (via `llama-cpp-python` or Ollama HTTP through `requests`), records metrics. (< 150 lines)
- **runners/airllm_runner.py** — Attempts AirLLM layer-by-layer inference; if not feasible on hardware, runs a **controlled analysis mode** that documents expected behavior, memory math, and why AirLLM helps. (< 150 lines)
- **run_benchmark.py** — argparse CLI with subcommands (`hardware`, `baseline`, `quant`, `airllm`, `all`, `plots`, `economics`, `report`). Thin orchestration only. (< 150 lines)

---

## 4. Data Flow
```
config.py (models, prompts)
      │
      ▼
run_benchmark.py ──> selects runner ──> runner.load() / generate()
      │                                     │
      ▼                                     ▼
hardware.py snapshot                 metrics.py (time, TTFT, tokens/s, RAM)
      │                                     │
      └────────────► results_io.py (append CSV row + log) ◄────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
        report.py            plots.py           economics.py
     (summary tables)     (PNG charts)      (cost comparison)
                                 │
                                 ▼
                          README.md (report)
```

---

## 5. Execution Flow
1. `python src/run_benchmark.py hardware` → write hardware snapshot.
2. `python src/run_benchmark.py baseline` → run + log baseline small model.
3. `python src/run_benchmark.py quant` → run + log quantized model.
4. `python src/run_benchmark.py airllm` → run OR analysis; log result.
5. `python src/run_benchmark.py report` → aggregate CSVs → summary tables.
6. `python src/run_benchmark.py plots` → generate PNG charts.
7. `python src/run_benchmark.py economics` → cost comparison CSV.
8. Paste tables/plots into `README.md` report sections.

---

## 6. Benchmarking Flow
For each configuration and each prompt, repeat `runs_per_config` times:
1. Snapshot RAM baseline.
2. Start RAM sampler thread.
3. `load()` → record load time (+ any OOM/error).
4. `generate(prompt)` → record TTFT (prefill), total time, generated tokens.
5. Compute decode tokens/sec and overall tokens/sec.
6. Stop sampler → record peak RAM.
7. Record success/slow/failure + `error_reason`.
8. Append one CSV row; append raw text to log.

---

## 7. Result-Saving Strategy
- **Raw runs** → `results/benchmark_runs.csv` (append-only, one row per run).
- **Aggregated** → `results/summary.csv`.
- **Economics** → `results/economics.csv`.
- **Plots** → `results/tokens_per_sec.png`, `results/load_time.png`, `results/ram_usage.png`.
- **Logs** → `results/run_YYYYMMDD_HHMMSS.log`.
- Logs and tmp files are gitignored; CSVs and PNGs are committed as evidence.

---

## 8. Testing Strategy
- **Smoke test**: `hardware` subcommand runs and prints a valid dict.
- **Dry-run mode**: a `--dry-run` flag runs the pipeline with a mock runner (no model download) to validate CSV schema, plotting, and I/O.
- **Schema test**: verify every CSV row has all expected columns.
- **Failure-path test**: confirm an intentionally-too-big model records `failure` + `error_reason` instead of crashing the whole run.
- **Line-count check**: assert each `src/**/*.py` file is < 150 lines.

---

## 9. CLI Commands (Planned)
```bash
# environment
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# pipeline
python src/run_benchmark.py hardware
python src/run_benchmark.py baseline
python src/run_benchmark.py quant
python src/run_benchmark.py airllm
python src/run_benchmark.py all --dry-run
python src/run_benchmark.py report
python src/run_benchmark.py plots
python src/run_benchmark.py economics
```

---

## 10. GitHub Submission Plan
1. `git init` (already a repo on `main`).
2. Commit documentation first (this stage): PRD, Plan, TODO, README, requirements, .gitignore.
3. Implement `src/` modules in small commits (one module ≈ one commit).
4. Run benchmarks → commit `results/` CSVs + PNGs (not logs).
5. Finalize README report with real numbers.
6. Push to a **public** GitHub repository.
7. Submit repository URL per assignment requirements.
