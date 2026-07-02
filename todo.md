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
- [x] Create `src/runners/` package with `__init__.py` (Phase 3A).
- [ ] Create `models/` folder (pending — created when first weights are downloaded).

## Phase 2 — Measurement Core (COMPLETE)
- [x] Implement `src/hardware.py` — collect CPU/RAM via psutil + GPU static specs.
- [x] Implement `src/metrics.py` — timers, RAM peak sampler, whitespace token count, tokens/sec, result-row builder (< 150 lines).
- [x] Implement `src/results_io.py` — CSV append, JSON save, log writer, CSV loader (< 150 lines).
- [x] Implement `src/plots.py` — bar charts from existing CSV rows; mock rows labelled clearly (< 150 lines).
- [x] Implement `src/economics.py` — estimated local vs cloud GPU vs API cost template → `results/economic_analysis.csv` (< 150 lines).
- [x] Implement `src/verify.py` + `verify` CLI — required-file / dir / line-count / CSV checks with PASS/FAIL (< 150 lines).
- [x] Wire CLI subcommands: `hardware`, `dry-run`, `plots`, `economics`, `verify`.
- [x] Unit-check: hardware snapshot prints valid dict.
- [x] Verify all measurement files < 150 lines (line-count check passes).
- [ ] Model runners (`baseline` / `quant` / `airllm`) real execution — pending Phase 6.
- [ ] Real benchmark numbers — pending Phase 6 (no fabrication).

## Phase 3A — Controlled Benchmark Runners (COMPLETE)
- [x] Implement `src/runners/base_runner.py` — shared schema helper + status constants (`success` / `failed` / `skipped` / `unavailable`).
- [x] Implement `src/runners/env_runner.py` — check `ollama` CLI presence, Python version, RAM (psutil) → `environment_check` row.
- [x] Implement `src/runners/controlled_runner.py` — fp16/int8/int4 memory formulas vs RAM=8 GB / VRAM=2 GB → `controlled_analysis` rows for baseline_fp16_7b / quantized_int4_7b / airllm_layer_streaming_7b.
- [x] Every controlled/env row tagged `result_type` (never `"real"`) with an explanatory `error_reason`.
- [x] Wire CLI subcommands `env-check` and `controlled`.
- [x] Update `plots.py` to colour/label `controlled_analysis` rows and exclude `environment_check` rows.
- [x] Update `verify.py` to require the `src/runners/` files (all still < 150 lines).
- [x] Run `env-check` (Ollama confirmed NOT installed) + `controlled` (3 estimate rows).

## Phase 3B — Optional Backend Availability Checks (COMPLETE) + Real Runners (PENDING)
Optional backend availability runners — implemented, safe-skipping, no inference:
- [x] Implement `src/runners/ollama_runner.py` — `shutil.which("ollama")`; if absent record `environment_check` row (`unavailable`, "Ollama CLI not installed"); placeholder documents a future real call.
- [x] Implement `src/runners/hf_runner.py` — `importlib.util.find_spec` for transformers+torch (no import); if missing record `environment_check` row (`unavailable`, "transformers/torch not installed").
- [x] Implement `src/runners/airllm_optional_runner.py` — `find_spec("airllm")`; if missing record `environment_check` row (`unavailable`, points to controlled AirLLM analysis).
- [x] Wire CLI subcommands `ollama-check` / `hf-check` / `airllm-check` / `backend-checks`.
- [x] Update `verify.py` to require the three new runner files (all still < 150 lines; verify PASS 23/23).
- [x] Confirm plots exclude `environment_check` rows; keep `controlled_analysis` non-real and `mock` labelled mock.
- [x] Run `backend-checks` in current env — all three backends confirmed **unavailable** (Ollama / transformers+torch / AirLLM not installed). Documented limitation, not a fake result.

Real inference runners — still OPTIONAL / PENDING (needs heavy deps + capable HW):
- [ ] Implement `src/runners/baseline_runner.py` — small HF model on CPU (real).
- [ ] Implement `src/runners/quant_runner.py` — GGUF/quantized via llama-cpp or Ollama (real).
- [ ] Implement `src/runners/airllm_runner.py` — real AirLLM run (falls back to controlled analysis).
- [ ] Ensure every real runner catches OOM/errors and records `error_reason`.
- [ ] NOTE: blocked in current environment — `ollama` not installed and heavy deps (torch/transformers/AirLLM/llama-cpp) intentionally not installed this phase.

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
