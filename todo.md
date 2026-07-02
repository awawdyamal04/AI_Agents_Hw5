# TODO — AI-Agent Task Checklist

Small, verifiable tasks following the Vibe Coding lifecycle:
**Idea → PRD → Plan → TODO → Verify → Execute → Push**

Legend: `[ ]` pending · `[x]` done · `[~]` in progress

---

## Phase 0 — Documentation (current stage)
- [x] Define idea and assignment scope.
- [x] Write `prd.md`.
- [x] Write `plan.md`.
- [x] Write `todo.md`.
- [x] Write `README.md` (report skeleton with placeholders).
- [x] Write `requirements.txt` (lightweight deps only).
- [x] Write `.gitignore`.
- [ ] Commit documentation stage to git.

## Phase 1 — Project Skeleton
- [x] Create `src/` package with `__init__.py`.
- [x] Create `results/` folder with `.gitkeep`.
- [x] Add `src/config.py` — paths, static hardware profile, CSV schema, run settings (< 150 lines).
- [x] Add `src/hardware.py` — psutil CPU/RAM + static profile → `results/hardware_profile.json` (< 150 lines).
- [x] Add `src/results_io.py` — JSON/CSV/log writers (< 150 lines).
- [x] Add `src/run_benchmark.py` — argparse CLI with `hardware` and `dry-run` subcommands (< 150 lines).
- [x] Verify `python -m src.run_benchmark hardware` runs and writes JSON.
- [x] Verify `python -m src.run_benchmark dry-run` writes mock CSV row + log.
- [ ] Create `src/runners/` package with `__init__.py` (pending — later phase).
- [ ] Create `models/` folder (pending — created when first weights are downloaded).

## Phase 2 — Measurement Core
- [ ] Implement `src/hardware.py` — collect CPU/RAM via psutil + GPU static specs.
- [ ] Implement `src/metrics.py` — timers, TTFT, tokens/sec, RAM peak sampler.
- [ ] Implement `src/results_io.py` — CSV append, log writer, CSV loader.
- [ ] Unit-check: hardware snapshot prints valid dict.

## Phase 3 — Runners
- [ ] Implement `src/runners/base_runner.py` — interface + shared timing + result schema.
- [ ] Implement `src/runners/baseline_runner.py` — small HF model on CPU.
- [ ] Implement `src/runners/quant_runner.py` — GGUF/quantized via llama-cpp or Ollama.
- [ ] Implement `src/runners/airllm_runner.py` — AirLLM run OR controlled analysis mode.
- [ ] Ensure every runner catches OOM/errors and records `error_reason`.

## Phase 4 — CLI Orchestrator
- [ ] Implement `src/run_benchmark.py` — argparse subcommands (hardware/baseline/quant/airllm/all/report/plots/economics).
- [ ] Add `--dry-run` mock mode.
- [ ] Verify all files are < 150 lines (line-count check).

## Phase 5 — Reporting & Analysis
- [ ] Implement `src/report.py` — aggregate raw CSV → summary tables.
- [ ] Implement `src/plots.py` — tokens/sec, load-time, RAM usage PNGs.
- [ ] Implement `src/economics.py` — Local vs Cloud GPU vs API cost model → CSV.

## Phase 6 — Run Experiments (generate real results)
- [ ] Run `hardware` → confirm hardware snapshot.
- [ ] Run `baseline` on a small model (e.g., gpt2 / Qwen2.5-0.5B / TinyLlama).
- [ ] Run `quant` on a quantized GGUF model.
- [ ] Run `airllm` (real run if feasible; else controlled analysis).
- [ ] Attempt an intentionally "too big" model → record expected failure + reason.
- [ ] Run `report`, `plots`, `economics`.
- [ ] Collect CSVs + PNGs into `results/`.

## Phase 7 — Finalize Report
- [ ] Fill README results section with **real** numbers (no fabrication).
- [ ] Fill discussion: prefill vs decode, VRAM/RAM limits, quantization effect, AirLLM effect.
- [ ] Confirm every result maps to a Lecture 08L concept.
- [ ] Add AI prompts used section.
- [ ] Verify Mermaid diagrams render.

## Phase 8 — Verify & Push
- [ ] Line-count check passes (all `.py` < 150 lines).
- [ ] Dry-run pipeline passes end-to-end.
- [ ] `git status` clean except intended files.
- [ ] Push to **public** GitHub repo.
- [ ] Submit repository URL.

---

### Guardrails (apply to every task)
- No fake results — unrun items are labelled "to be generated".
- Keep each Python file under 150 lines; split if needed.
- Everything must run from the terminal.
- Log failures honestly; a documented failure is a valid result.
