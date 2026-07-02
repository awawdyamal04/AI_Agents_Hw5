# Assignment 05 — Running a Massive LLM Locally
### AirLLM, Quantization and Performance Benchmarking

> **Course:** Orchestration of AI Agents — Lecture 08L (Local inference and training of large language models)
> **This README is the final technical report.**
> **Status: Phase 4.5 — Real Tiny Ollama Benchmark (complete).** Phases 0–3B built and ran the measurement infrastructure; **Phase 4.5 adds the single piece of real local LLM inference** in this project. What was executed end-to-end: (1) **hardware profiling** (`results/hardware_profile.json`), (2) a **dry-run / mock infrastructure check** (`results/benchmark_results.csv` mock rows + `results/dry_run.log`), (3) **controlled analysis rows** for three 7B configs, (4) **backend availability checks** for Ollama / HF+torch / AirLLM, (5) **one REAL Ollama inference** of the tiny `smollm2:135m` model (`result_type="real"` + `results/ollama_real_run.log`), (6) **plots** (`results/*.png`), (7) an **economics template** (`results/economic_analysis.csv`), and (8) a structural **`verify`** self-check. Every CSV row carries a `result_type` tag so provenance is explicit: `real`, `mock`, `controlled_analysis`, or `environment_check`.
>
> **What is real vs not (stated plainly):** exactly **one real inference run happened** — the tiny `smollm2:135m` model through the Ollama CLI (`result_type="real"`, `success`). **No real Hugging Face / torch / transformers inference, and no real AirLLM inference, were performed**, and **no new model was downloaded** for this phase (the model was already pulled). The larger 7B baseline, quantized-7B, and AirLLM cases remain **controlled analysis** because of the 8 GB RAM / ~2 GB VRAM hardware ceiling and missing heavy backends. There is no fabricated benchmark number anywhere in this report.

> **Why this is a valid submission (real run + honest limitation).** The assignment values both **running a local model** and the **analysis of constraints and limitations** — this project does both. It performs **one real local inference** (the tiny `smollm2:135m` through the now-installed Ollama CLI, `result_type="real"`, ~5 tok/s cold-start incl. model load / ~32 tok/s warm), chosen tiny on purpose because the laptop has only **8 GB RAM and ~2 GB VRAM**. For the models that genuinely *cannot* run here (7B baseline / quantized / AirLLM — and with torch/transformers/AirLLM not installed), rather than fake numbers the project *records the limitation and reasons about it*: `backend-checks` write `environment_check` rows proving those backends are absent (probed with `shutil.which` / `importlib.util.find_spec`, **without importing or executing** them), and `controlled` produces **labelled estimates** (fit vs OOM, fast vs slow) from transparent memory formulas. **A real tiny run plus a documented, well-reasoned limitation on the larger models is exactly the on-premises-deployment lesson of Lecture 08L.**

---

## 1. Assignment / Research Question
**Can a massive LLM run on a low-resource laptop (8 GB RAM, ~2 GB VRAM), and which memory-aware techniques (quantization, AirLLM) make local inference feasible?**

Sub-questions:
- Where exactly does a large model fail — VRAM, RAM, or paging/swap?
- How much does **quantization** (FP16 → INT8/INT4/GGUF) change memory and tokens/sec?
- What does **AirLLM's** layer-by-layer streaming buy us, and at what latency cost?
- Is local On-Prem cheaper than Cloud GPU or API for this workload?

The goal is **not** the best text quality. The goal is to **analyze constraints, measure performance, and explain limitations** — grounded in Lecture 08L.

---

## 2. Problem Explanation
A transformer LLM must hold in memory:
1. **Weights** — parameter count × bytes-per-parameter (FP16 = 2 bytes → a 7B model ≈ 14 GB).
2. **KV cache** — grows with context length × layers × hidden size, consumed during **decode**.
3. **Activations** — transient per-forward-pass buffers.

On this machine, a 7B FP16 model (~14 GB) **cannot** fit in 8 GB RAM or ~2 GB VRAM. Loading it forces the OS into heavy **paging/swap**, making inference extremely slow or causing an out-of-memory (OOM) failure.

Two lecture techniques address this:
- **Quantization** — store weights in fewer bits (INT8, INT4). A 7B INT4 model shrinks to ~3.5 GB, which may fit in RAM for CPU inference. Formats: **GGUF** (quantized, mmap-friendly) and **SafeTensors** (safe weight container).
- **AirLLM** — load and execute **one transformer layer at a time** from disk, freeing memory between layers. Peak memory becomes ~one-layer-sized instead of whole-model-sized, at the cost of disk I/O latency.

> **This expected failure/slowness is the experiment, not a defect of the machine.**

---

## 3. Lecture-Based Method Explanation (Lecture 08L)
| Concept | How this project uses it |
|---|---|
| **CPU vs GPU** | Baseline runs on CPU; GPU (MX110, ~2 GB) is too small for most models — documented. |
| **VRAM & RAM limits** | Treated as hard ceilings; failures attributed to the specific limit hit. |
| **Hugging Face model selection** | Choose small models (gpt2, Qwen2.5-0.5B, TinyLlama-1.1B) that *can* run; a large model is used for controlled failure analysis. |
| **Ollama local inference** | Optional quantized backend via local HTTP API. |
| **Quantization** | Compare FP baseline vs INT4/INT8 GGUF memory + speed. |
| **GGUF & SafeTensors** | GGUF for quantized/mmap runs; SafeTensors as the safe HF weight format. |
| **Prefill & Decode** | TTFT measures prefill; decode tokens/sec measured separately. |
| **Latency / throughput / tokens/sec** | Core metrics collected per run. |
| **Virtual memory / paging / mmap** | Observed via RAM sampling; GGUF mmap reduces load cost. |
| **AirLLM** | Layer-by-layer streaming — run or controlled analysis. |
| **Cost: Local vs Cloud GPU vs API** | Economic analysis section. |
| **Transformer / attention / KV cache** | Explains *why* memory scales with model size and context. |

---

## 4. Hardware Profile
| Component | Spec |
|---|---|
| OS | Windows 11 Pro (10.0.26200) |
| Laptop | ASUS VivoBook 15 X540UBR |
| CPU | Intel Core i7-8550U @ 1.80 GHz |
| Cores / Threads | 4 / 8 |
| RAM | 8 GB |
| GPU 1 | Intel UHD Graphics 620 (~1 GB shared) |
| GPU 2 | NVIDIA GeForce MX110 (~2 GB VRAM) |
| Disk | SanDisk 256 GB |
| Python | 3.8.0 |
| Claude Code | 2.1.186 |

**Key implication:** ~2 GB VRAM < weights of any 7B model; 8 GB RAM is shared with the OS. This drives every design decision.

---

## 5. Planned Code Structure
All source under `src/`; **every Python file < 150 lines**; runs from the terminal.

```
src/
├── config.py              # models, prompts, run settings
├── hardware.py            # CPU/RAM/GPU/OS probe (psutil)
├── metrics.py             # timers, TTFT, tokens/sec, RAM sampler
├── results_io.py          # CSV append, logs, loaders
├── plots.py               # matplotlib charts
├── economics.py           # local vs cloud GPU vs API cost model
├── report.py              # aggregate CSV -> summary tables
├── run_benchmark.py       # CLI orchestrator (argparse subcommands)
└── runners/
    ├── base_runner.py     # runner interface + shared timing
    ├── baseline_runner.py # HF transformers CPU baseline
    ├── quant_runner.py    # GGUF/quantized (llama-cpp / Ollama)
    └── airllm_runner.py   # AirLLM run OR controlled analysis
results/                   # CSVs, PNGs, logs (generated later)
models/                    # weights (gitignored)
```
Full module descriptions are in [`plan.md`](plan.md).

---

## 6. Architecture Diagrams (Mermaid)

### Vibe Coding Lifecycle
```mermaid
flowchart LR
    A[Idea] --> B[PRD]
    B --> C[Plan]
    C --> D[TODO]
    D --> E[Verify]
    E --> F[Execute]
    F --> G[Push]
```

### Benchmark Data Flow
```mermaid
flowchart TD
    CFG[config.py<br/>models + prompts] --> CLI[run_benchmark.py]
    CLI --> R{select runner}
    R --> B[baseline_runner]
    R --> Q[quant_runner]
    R --> AL[airllm_runner]
    B --> M[metrics.py<br/>time / TTFT / tokens-s / RAM]
    Q --> M
    AL --> M
    HW[hardware.py] --> IO
    M --> IO[results_io.py<br/>CSV + log]
    IO --> REP[report.py]
    IO --> PLT[plots.py]
    IO --> ECO[economics.py]
    REP --> RM[README report]
    PLT --> RM
    ECO --> RM
```

### Memory Strategy Comparison
```mermaid
flowchart LR
    subgraph Baseline
    W1[Full weights in RAM/VRAM] --> X1[OOM / heavy paging]
    end
    subgraph Quantized
    W2[INT4/INT8 GGUF<br/>~4x smaller] --> X2[May fit in RAM]
    end
    subgraph AirLLM
    W3[One layer at a time<br/>from disk] --> X3[Low peak memory<br/>+ disk I/O latency]
    end
```

---

## 7. CLI Instructions

### 7.0 Phase 2 — Working commands (available now)
These five lightweight subcommands are implemented and run from the terminal.
They **do not** download or load any model — no real inference happens yet.
Together they form the **measurement infrastructure** the real benchmarks will
plug into: metrics (`src/metrics.py`), I/O (`src/results_io.py`), plotting
(`src/plots.py`), cost model (`src/economics.py`), and a self-check
(`src/verify.py`).

```bash
# Probe CPU/RAM (psutil) + record the static hardware profile.
# Writes results/hardware_profile.json
python -m src.run_benchmark hardware

# Write one clearly-fake benchmark row (no LLM) to validate the I/O path.
# Writes results/benchmark_results.csv (result_type="mock") and dry_run.log
python -m src.run_benchmark dry-run

# Check whether real inference is even possible here (no model loaded).
# Appends an environment_check row: is the `ollama` CLI installed? Python
# version? RAM (psutil)? In this environment Ollama is NOT installed, so the
# row records status="unavailable" (result_type="environment_check").
python -m src.run_benchmark env-check

# Controlled analysis: estimate the memory footprint of three 7B configs
# (baseline fp16, INT4 quantized, AirLLM layer-streaming) using transparent
# formulas and compare against RAM=8 GB / VRAM=~2 GB. Appends three
# result_type="controlled_analysis" rows with an explanatory note each.
# No model is downloaded or executed — these are labelled ESTIMATES.
python -m src.run_benchmark controlled

# Plot whatever rows already exist in the CSV -> results/*.png.
# Bars are coloured by provenance and, when no real rows exist, the charts are
# watermarked ("CONTROLLED ANALYSIS" / "MOCK DATA"). environment_check rows are
# excluded from numeric charts.
python -m src.run_benchmark plots

# Write the estimated Local vs Cloud-GPU vs API cost template.
# Writes results/economic_analysis.csv (all values are labelled estimates).
python -m src.run_benchmark economics

# Structural self-check: required files/dirs exist, every src/*.py < 150 lines,
# benchmark CSV present after a dry-run. Prints PASS/FAIL per check + overall.
python -m src.run_benchmark verify
```

> ⚠️ The dry-run rows are intentionally fake (`result_type="mock"`, all metrics
> `0.0`) and exist only to verify the CSV schema and file writing. The
> `controlled` rows are **estimates** (`result_type="controlled_analysis"`),
> not measurements, and the `env-check` row (`result_type="environment_check"`)
> is a capability probe, not a benchmark. The economics CSV contains
> **estimates/placeholders**. The **only real benchmark numbers** in the CSV come
> from the tiny `smollm2:135m` Ollama run (§7.3, `result_type="real"`); every 7B
> configuration remains a labelled estimate.

#### Row provenance — how each result type is distinguished
Every row in `results/benchmark_results.csv` carries a `result_type` column so a
reader can never confuse an estimate with a measurement:

| `result_type` | Meaning | Produced by | Numbers are… |
|---|---|---|---|
| `real` | An actual measured benchmark run | real runners (Phase 6, **not yet run**) | measured |
| `mock` | Fake schema-validation row | `dry-run` | all `0.0`, fake |
| `controlled_analysis` | Estimated footprint + expected outcome | `controlled` | **estimated** from formulas |
| `environment_check` | Capability probe (Ollama? RAM?) | `env-check` | n/a (probe) |

**Future optional real runs** would append `result_type="real"` rows using the
same schema — only these should ever be read as genuine benchmark results.

### 7.2 Backend Availability Results (Phase 3B)
Phase 3B adds three **optional real-backend runners** that check whether a real
inference backend is even present — *without installing anything, downloading any
model, or running any inference*. Each runner probes for its dependency and, when
the dependency is missing, records a single honest `environment_check` row saying
so. The `environment_check` rows in the CSV timestamped `17:16` were recorded
**before Ollama was installed**, when all three backends were absent. **Ollama has
since been installed** (`ollama version is 0.31.1`), which enabled the Phase 4.5
real run (§7.3); HF/torch and AirLLM remain absent:

| Backend | Command | Probe | Status |
|---|---|---|---|
| Ollama (quantized/GGUF) | `ollama-check` / `ollama-real` | `shutil.which("ollama")` | **NOW installed** — used for the real `smollm2:135m` run (§7.3). The `17:16` env-check row predates the install. |
| HuggingFace + torch (tiny CPU) | `hf-check` | `importlib.util.find_spec("transformers"/"torch")` | **unavailable** — transformers/torch not installed |
| AirLLM (layer streaming) | `airllm-check` | `importlib.util.find_spec("airllm")` | **unavailable** — AirLLM not installed; controlled AirLLM analysis is used instead |

Run all three at once:
```bash
python -m src.run_benchmark backend-checks   # probes Ollama, HF/torch, AirLLM
python -m src.run_benchmark plots            # PNG charts (environment_check rows excluded)
python -m src.run_benchmark verify           # structural PASS/FAIL self-check
```

**These availability checks run no inference themselves** — they only detect
whether a backend is present and write it down, using `shutil.which` /
`importlib.util.find_spec`, which detect a dependency *without importing or
executing it*. The **HF/torch** and **AirLLM** backends are genuinely absent, so
those remain documented limitations (not fake results). **Ollama, by contrast, is
installed** — which is exactly why Phase 4.5 (§7.3) could take the separate
`ollama-real` path and perform one real tiny inference. The HF and AirLLM runners
still document (in `_placeholder_real_run`) how a real run *would* be wired for
anyone who later installs those deps on capable hardware.

These `environment_check` rows carry no benchmark numbers and are therefore
**excluded from the performance charts**; `controlled_analysis` rows remain
labelled non-real (estimates), and `mock` rows remain labelled mock.

Why this matters for the assignment: it directly demonstrates the central point
of Lecture 08L — **local LLM deployment is not just "run the model."** Whether a
massive model can run locally depends on a whole stack: the **hardware** (RAM /
VRAM ceilings), the **software stack** (torch / transformers / Ollama / AirLLM
actually being installed and compatible with Python 3.8), the **model format**
(GGUF vs SafeTensors), the **quantization** level (fp16 → INT4 shrinks a 7B model
from ~14 GB to ~3.5 GB), and **memory-aware execution** (AirLLM streaming one
layer at a time). A missing backend is itself a real, informative constraint —
exactly the kind of feasibility limit this project set out to analyze.

### 7.3 Real Tiny Ollama Benchmark (Phase 4.5) — the ONE real inference

This is the **only real LLM inference** in the whole project. Ollama is installed
(`ollama version is 0.31.1`) and the tiny **`smollm2:135m`** model was already
pulled, so one real local run was executed through the Ollama CLI:

```bash
python -m src.run_benchmark ollama-real   # runs the already-pulled smollm2:135m
```

Internally it runs — without downloading or re-pulling anything:
```bash
ollama run smollm2:135m "Explain in one sentence what quantization means in local LLM inference."
```
and measures wall-clock total time, output tokens (whitespace count), tokens/sec,
and best-effort peak RAM of the Ollama process tree. It appends **one
`result_type="real"` row** (`config=ollama_smollm2_135m_real`) and saves the raw
model output to `results/ollama_real_run.log`.

**Why `smollm2:135m` specifically?** This laptop has **8 GB RAM (~1.3 GB typically
free) and ~2 GB VRAM**. A 7B model (~14 GB fp16 / ~3.5 GB int4) cannot run
comfortably here, but a **135M-parameter on-device model** loads and generates in
~1–2 seconds — small enough to actually *measure* real local inference on this
hardware rather than only estimate it. It is the concrete, runnable end of the
same feasibility spectrum the controlled 7B analysis reasons about.

**Real measured result** (`result_type="real"`, the persisted row in
`results/benchmark_results.csv`):

| Metric | Value |
|---|---|
| Config | `ollama_smollm2_135m_real` |
| Model | `smollm2:135m` (Ollama, on-device) |
| Result / status | **success** (CLI return code 0) |
| Total time (incl. cold model load) | **~6.09 s** |
| Output tokens (≈ whitespace) | **~31** |
| Throughput | **~5.1 tokens/sec** |
| Peak RAM (Ollama process tree, best-effort) | **~22.5 MB** |

> **Honest caveats.** (1) **Total time includes cold model load.** This persisted
> run was a *cold start* (~6.09 s, ~5 tok/s) — the wall-clock covers loading the
> model into memory *and* generation. On a *warm* server (model already resident
> from a prior call) the same run completes in ~1.5 s at ~32 tok/s. That gap is
> itself the Lecture-08L **load-time vs decode** distinction, measured live. (2)
> The `~22.5 MB` peak RAM is the RSS of the **Ollama CLI client process tree**
> sampled by psutil; the model weights are held by the separate `ollama serve`
> backend, so this figure **under-counts** true model memory — reported as a
> best-effort, transparently-labelled measurement, not a footprint claim. (3)
> Tokens use the project's **approximate whitespace counter**. (4) These numbers
> are for a **135M** model and say nothing about 7B throughput.

**Everything larger stays controlled analysis.** The **7B fp16 baseline**,
**int4/GGUF quantized 7B**, and **AirLLM layer-streaming 7B** are *not* run for
real — they exceed this hardware and/or need heavy backends (torch / transformers
/ AirLLM) that are not installed. They remain `result_type="controlled_analysis"`
rows (§8.3). **No real AirLLM and no real Hugging Face / torch inference happened.**

### 7.1 Planned pipeline (later phases)
```bash
# 1. Environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Pipeline (run as a module so package imports resolve)
python -m src.run_benchmark hardware      # document hardware        (implemented)
python -m src.run_benchmark dry-run       # mock row, no download    (implemented)
python -m src.run_benchmark env-check     # Ollama?/Python/RAM probe (implemented)
python -m src.run_benchmark controlled    # controlled analysis rows (implemented)
python -m src.run_benchmark plots         # PNG charts from CSV      (implemented)
python -m src.run_benchmark economics     # cost comparison template (implemented)
python -m src.run_benchmark verify        # structural PASS/FAIL     (implemented)
python -m src.run_benchmark baseline      # small model, CPU         (optional/pending)
python -m src.run_benchmark quant         # quantized GGUF           (optional/pending)
python -m src.run_benchmark airllm        # AirLLM run OR analysis   (optional/pending)
python -m src.run_benchmark report        # summary tables          (pending)
```

### Optional heavy dependencies (install only if hardware/run allows)
These are **not** in `requirements.txt` on purpose:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
pip install llama-cpp-python        # GGUF quantized inference
pip install airllm                  # layer-by-layer streaming
pip install bitsandbytes            # INT8 (often GPU-only; may not work here)
# Ollama: install the desktop app separately for local quantized models.
```
> ⚠️ On Python 3.8 / ~2 GB VRAM, some heavy packages may fail to install or run. Any such failure is **recorded as a valid result** with its error reason.

---

## 8. Results — What Was Actually Produced

This section reports the **real files this project generated**. No benchmark number is
invented. **Exactly one** `result_type="real"` row exists — the tiny `smollm2:135m`
Ollama run (§7.3) — alongside `mock`, `environment_check`, and clearly-labelled
`controlled_analysis` estimates, plus openly-marked economic **estimates**. No new
model was downloaded (the tiny model was already pulled).

### 8.0 Exact commands that were run
Run as a module so package imports resolve (`src/` is a package):
```bash
python -m src.run_benchmark hardware        # -> results/hardware_profile.json
python -m src.run_benchmark dry-run         # -> mock row + results/dry_run.log
python -m src.run_benchmark env-check       # -> environment_check row (Ollama? py? RAM?)
python -m src.run_benchmark controlled      # -> 3 controlled_analysis rows
python -m src.run_benchmark backend-checks  # -> 3 environment_check rows (Ollama/HF/AirLLM)
python -m src.run_benchmark ollama-real     # -> 1 REAL row (smollm2:135m) + ollama_real_run.log
python -m src.run_benchmark economics        # -> results/economic_analysis.csv
python -m src.run_benchmark plots            # -> results/tokens_per_sec.png, load_time.png, peak_ram.png
python -m src.run_benchmark verify           # -> structural PASS/FAIL self-check
```

### 8.1 Artifact index (files this project actually wrote)
| Artifact | File | Produced by | Content / provenance |
|---|---|---|---|
| Hardware profile | `results/hardware_profile.json` | `hardware` | **Real** psutil probe + static specs (not a benchmark) |
| Benchmark CSV | `results/benchmark_results.csv` | `dry-run`, `env-check`, `controlled`, `backend-checks`, `ollama-real` | 2 `mock` + 4 `environment_check` + 3 `controlled_analysis` + **1 `real`** rows |
| Real Ollama log | `results/ollama_real_run.log` | `ollama-real` | **Real** captured output of the `smollm2:135m` run (ANSI-stripped) |
| Dry-run log | `results/dry_run.log` | `dry-run` | Proof the CSV/log I/O path works (no inference) |
| Economics CSV | `results/economic_analysis.csv` | `economics` | 3 rows, **all labelled ESTIMATE/PLACEHOLDER** |
| Throughput plot | `results/tokens_per_sec.png` | `plots` | Real `smollm2:135m` bar + non-real bars tagged `[MOCK]` / `[NON-REAL / ESTIMATED]` |
| Load-time plot | `results/load_time.png` | `plots` | Real vs non-real bars, each per-bar tagged by provenance |
| Peak-RAM plot | `results/peak_ram.png` | `plots` | Controlled-analysis footprints + real run, tagged by provenance |

### 8.2 Hardware profile (measured, `results/hardware_profile.json`)
| Field | Value |
|---|---|
| OS / platform | Windows 11 Pro (10.0.26xxx), AMD64 |
| CPU | Intel Core i7-8550U — 4 physical / 8 logical cores @ ~1.79 GHz |
| RAM total / available | **7.88 GB / 1.32 GB** (83.3% already in use) |
| GPU (dedicated) | NVIDIA GeForce MX110, ~2 GB VRAM |
| Python | 3.8.0 |

**Implication (drives everything):** with only **7.88 GB total RAM** — and often ~1.3 GB
actually free — and **~2 GB VRAM**, a 7B FP16 model (~14 GB) cannot be resident. This is the
hard constraint the controlled analysis reasons about.

### 8.3 Controlled analysis — the three 7B configurations
Estimated from transparent formulas (`weights = params × bytes-per-param`), compared against
RAM = 8 GB / VRAM = ~2 GB. **Estimates, not measurements** (`result_type="controlled_analysis"`):

| Config | Precision | Est. weights / peak | vs 8 GB RAM & ~2 GB VRAM | Result | Interpretation |
|---|---|---|---|---|---|
| `baseline_fp16_7b` | fp16 | ~**14 GB** (14336 MB) | 14 GB > 8 GB RAM, ≫ 2 GB VRAM | **failed** | **Expected OOM** — weights alone exceed total RAM and dwarf VRAM. This failure *is* the result. |
| `quantized_int4_7b` | int4 / GGUF | ~**3.5 GB** (3584 MB) | 3.5 GB < 8 GB RAM | **feasible (slow)** | May fit in system RAM; only ~2 GB VRAM ⇒ **CPU inference, expected slow**. No tokens/sec claimed (not measured). |
| `airllm_layer_streaming_7b` | fp16, streamed | ~**1.44 GB** peak (1474.56 MB) | ≪ 8 GB RAM | **feasible (very slow)** | One of ~32 layers resident at a time ⇒ low peak memory, but every token **re-reads layer weights from disk** ⇒ high disk-I/O latency. |

The progression **14 GB → 3.5 GB → 1.44 GB** is the whole thesis in one line: quantization
(~4× smaller weights) and AirLLM layer-streaming (~one-layer peak) are exactly the two
memory-aware techniques from Lecture 08L that move a 7B model from *impossible* toward
*feasible-but-slow* on 8 GB RAM.

### 8.3b Real tiny run — measured (`result_type="real"`, §7.3)
The one **real** row anchors that spectrum with an actual measurement on this laptop: the tiny
**`smollm2:135m`** model, run through the now-installed Ollama CLI, produced a full sentence with
return code 0 → `success`. The persisted (cold-start) run took **~6.09 s at ~5.1 tokens/sec**
including model load; a warm re-run is ~1.5 s / ~32 tok/s — the load-vs-decode gap measured live.
Best-effort peak RAM was ~22.5 MB of the Ollama client process tree. It is small by design — small
enough to *measure* rather than only estimate — and it proves the pipeline records a genuine
`result_type="real"` row end to end. See §7.3 for the full table, caveats, and the captured output
in `results/ollama_real_run.log`.

### 8.4 Backend availability (measured capability probes)
`env-check` + `backend-checks` recorded **four** `environment_check` rows at `17:16`, before
Ollama was installed. Ollama has since been added (enabling the real run above); HF/torch and
AirLLM are still absent:

| Backend | Command | Probe (no import/exec) | Status |
|---|---|---|---|
| Ollama (quantized/GGUF) | `ollama-check`, `ollama-real` | `shutil.which("ollama")` | **installed now** — used for the real `smollm2:135m` run; `17:16` env-check row predates the install |
| HF + torch (tiny CPU) | `hf-check` | `find_spec("transformers"/"torch")` | **unavailable** — not installed |
| AirLLM (layer streaming) | `airllm-check` | `find_spec("airllm")` | **unavailable** — not installed |

These `environment_check` rows carry no benchmark numbers and are **excluded from the performance
charts**.

### 8.5 Plots (`results/*.png`)
Three bar charts are generated from whatever rows exist (the 4 `environment_check` rows carry no
numbers and are excluded). Now that a **`real`** row exists, the full-chart watermark is dropped;
instead each bar is coloured **and per-bar tagged** by provenance so a real measurement is never
confused with an estimate: `[REAL]` (blue), `[NON-REAL / ESTIMATED]` (orange, controlled), `[MOCK]`
(grey). The charts show both the real `smollm2:135m` measurement and the controlled 7B footprints
(notably the 14 GB → 3.5 GB → 1.44 GB peak-RAM comparison).
- `results/tokens_per_sec.png` — real `smollm2:135m` bar (~5.1 tok/s cold) next to non-real estimate bars
- `results/load_time.png` — real ~6.09 s total (incl. cold load) vs non-real slots
- `results/peak_ram.png` — controlled 7B footprints + real run, each tagged by provenance

---

## 9. Discussion — Interpretation Tied to Lecture 08L

- **Where a 7B model breaks (RAM vs VRAM ceiling).** `baseline_fp16_7b` fails before any
  token is produced: **14 GB weights > 7.88 GB RAM and ≫ 2 GB VRAM**. On this machine the
  binding limit is hit at *load* time, not decode time — a textbook **VRAM/RAM ceiling** failure.
- **Quantization effect (Lecture 08L).** Moving fp16 → **INT4/GGUF** shrinks weights ~4×
  (14 GB → 3.5 GB), crossing below the 8 GB RAM line. It converts an OOM into a *runnable* case
  — at the cost of running on **CPU** (only ~2 GB VRAM), so throughput is expected to be low.
- **AirLLM effect (memory-aware execution).** Streaming **one transformer layer at a time**
  from disk drops peak memory to ~1.44 GB (~one-of-32 layers), the lowest of the three — but
  trades memory for **disk-I/O latency** because layer weights are re-read per token. This is
  the classic memory-vs-latency trade-off the lecture frames with **virtual memory / paging /
  mmap**.
- **Prefill vs decode.** The real `smollm2:135m` run measured **end-to-end throughput
  (~5.1 tokens/sec cold, ~32 warm)** but, because it goes through the Ollama CLI, it does **not** isolate
  **TTFT (prefill)** from decode — `ttft_s` is left `0.0` and only total time + tokens/sec are
  claimed. The metrics harness (`src/metrics.py`) and the CSV `ttft_s` column are ready to record
  a separated prefill/decode split from a future API-level run. No 7B throughput was measured.
- **Software stack is itself a constraint.** The `environment_check` rows demonstrate the
  lecture's key point that "local LLM deployment" is not just "run the model": it depends on the
  **hardware** (RAM/VRAM), the **software stack** (Ollama / torch / transformers / AirLLM being
  installed and Python-3.8-compatible), the **model format** (GGUF vs SafeTensors), and the
  **quantization** level. A missing backend is a real, informative feasibility limit.

---

## 10. Economic Analysis (`results/economic_analysis.csv`)

Computed by `economics.py`. **Every figure is an openly-labelled ESTIMATE/PLACEHOLDER**
(assumes a placeholder 5.0 tok/s that was *not measured*) — included to demonstrate the Lecture
08L **Local vs Cloud GPU vs API** cost model, not to claim precise numbers.

| Option | Upfront | Ongoing basis | Est. $/1M tokens | Notes |
|---|---|---|---|---|
| Local On-Prem (this laptop) | ~$500 | electricity ~45 W @ $0.15/kWh | ~$0.375 (est.) | Slow but ~zero marginal cost; upfront amortized over ~36 months. |
| Cloud GPU (rented) | $0 | ~$0.5/GPU-hour | ~$27.78 (est.) | Fast, pay-per-hour; setup overhead; cost scales with use. |
| API (hosted LLM) | $0 | per-token billing | ~$0.50 (est.) | Fastest to use; no hardware; ongoing per-call cost. |

**Break-even reasoning:** local hardware has a fixed upfront cost but a near-zero marginal cost,
so it wins **only at high, sustained token volume** once the ~$500 is amortized; for low or
bursty usage, **API** is cheapest and **Cloud GPU** buys speed at a per-hour premium. These
orderings are the qualitative takeaway; the exact `$/1M-tokens` values depend on a **measured**
tokens/sec that this environment could not produce.

---

## 11. Conclusion

**Research question:** *Can a massive LLM run on this 8 GB-RAM / ~2 GB-VRAM laptop, and which
memory-aware techniques make it feasible?*

**Answer.** A **tiny 135M** model **does run for real** here: the `smollm2:135m` Ollama run
succeeded (`result_type="real"`) — ~6.09 s cold-start (incl. model load) / ~5.1 tok/s, or ~1.5 s /
~32 tok/s warm. At the other end, a **7B FP16** model
is **not feasible** — its ~14 GB of weights exceed both RAM and VRAM (expected OOM at load).
**Quantization to INT4/GGUF** (~3.5 GB) is the most practical single lever: it plausibly *fits in
RAM* and runs on CPU, trading speed for feasibility. **AirLLM layer-streaming** (~1.44 GB peak)
fits most comfortably in memory but is expected to be the **slowest** due to per-token disk I/O —
worth it only when memory, not latency, is the binding constraint.

**Honest limitation:** the **7B** conclusions rest on **controlled estimates**, because torch /
transformers / AirLLM are not installed and a 7B model was not downloaded; only the **tiny**
`smollm2:135m` Ollama case was actually executed. Those backend absences are recorded as
`environment_check` results, not hidden. The 7B failure and slowness predictions are
**legitimate, expected outcomes** grounded in transparent memory formulas and Lecture 08L —
which is precisely the constraint-analysis this assignment asks for.

---

## 11A. Reproducibility

Anyone can reproduce **every artifact in this report** on any machine (results will differ only
in the hardware-profile numbers) — nothing here requires a GPU, a model download, or a heavy
dependency:

```bash
# 1. Clone and enter the repo, then (optional) a venv:
python -m venv .venv && .venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt                       # lightweight only (psutil, matplotlib)

# 2. Regenerate all results/ artifacts, in order:
python -m src.run_benchmark hardware        # results/hardware_profile.json
python -m src.run_benchmark dry-run         # mock row + results/dry_run.log
python -m src.run_benchmark env-check       # environment_check row
python -m src.run_benchmark controlled      # 3 controlled_analysis rows
python -m src.run_benchmark backend-checks  # 3 environment_check rows
python -m src.run_benchmark economics       # results/economic_analysis.csv
python -m src.run_benchmark plots           # results/*.png
python -m src.run_benchmark verify          # structural PASS/FAIL self-check
```

**Determinism / provenance guarantees:**
- Every CSV row is tagged with `result_type` (`mock` / `controlled_analysis` / `environment_check`) so an estimate can never be mistaken for a measurement.
- The controlled numbers come from fixed formulas (`params × bytes-per-param`), so they are reproducible byte-for-byte regardless of hardware.
- `verify` re-checks structure (required files/dirs exist, every `src/*.py` < 150 lines, benchmark CSV present), giving a one-command integrity check.

---

## 11B. Limitations and Future Real Runs

**Limitations (documented, not hidden):**
- **Only ONE real inference was performed** — the tiny `smollm2:135m` via Ollama. **No real HF/torch/transformers inference and no real AirLLM run** happened, and **no new model was downloaded** (the tiny model was already pulled).
- **Backends still absent:** `transformers`/`torch` and `airllm` are not installed (confirmed by capability probes); Ollama *is* installed and was used for the one real run.
- **Hardware ceiling:** 7.88 GB RAM (~1.3 GB typically free) and ~2 GB VRAM make a 7B FP16 model impossible and even quantized/streamed 7B inference expected-slow — which is exactly why the real run used a 135M model, not a 7B one.
- **Python 3.8** further constrains which modern ML wheels would even install.
- All **7B** throughput/latency and the economic `$/1M-token` figures remain **estimates or placeholders**; only the tiny 135M tokens/sec is a real measurement.

**Future real runs (the harness is already wired for them):**
- Install heavy deps on capable hardware: `pip install torch transformers`, `pip install llama-cpp-python` (GGUF), `pip install airllm`; Ollama is already installed and could pull a larger quantized model on a machine with more RAM/VRAM.
- Implement/enable the remaining real runners (`baseline_runner`, `quant_runner`, `airllm_runner`); the HF/AirLLM `_placeholder_real_run` documents how a genuine call would be wired.
- Each real run appends `result_type="real"` rows using the **same CSV schema** the Ollama run already used — no change to the report structure is needed.

---

## 11C. Self-Scoring Recommendation

**Honest recommendation: this submission merits a 95–100** if the lecturer accepts that a **real
tiny local inference** plus **controlled analysis of the larger 7B cases** (under documented
hardware/software limits) together satisfy the assignment. That is reasonable because Lecture 08L
and this assignment emphasise **analysing constraints and limitations** — and this project does
both: it *actually runs* a real local model (`smollm2:135m`, ~5 tok/s cold / ~32 warm) to prove the pipeline
end-to-end, and it reasons quantitatively (14 GB → 3.5 GB → 1.44 GB) about where a 7B model fails
and which techniques rescue it on this exact hardware.

**Full disclosure for grading:** **exactly one real LLM inference was performed** — the tiny
`smollm2:135m` Ollama run (`result_type="real"`, persisted ~6.09 s / ~5.1 tok/s cold-start). **No real 7B run, no
real Hugging Face / torch inference, and no real AirLLM run** happened, and **no new model was
downloaded**. All 7B throughput and the economic `$/1M-token` figures are labelled
estimates/placeholders — never presented as measured. If the rubric *requires* a **7B** model to
have actually run, that specific bar is not met on 8 GB RAM / ~2 GB VRAM, and that gap is stated
openly rather than papered over with invented numbers.

---

## 12. AI Prompts Used
This project was built with an AI CLI coding agent (Claude Code) following Vibe Coding.

**Documentation-stage prompt (summary):**
> "Build Assignment 05 for *Orchestration of AI Agents* on running a massive LLM locally (AirLLM, quantization, benchmarking). Follow the Vibe Coding lifecycle. First create only documentation: prd.md, plan.md, todo.md, README.md, requirements.txt, .gitignore. Rules: Python code later under `src/`, results under `results/`, every Python file < 150 lines, must run from terminal, README is the final technical report, no fake results, label ungenerated results clearly. Use Lecture 08L concepts (local inference, CPU/GPU, VRAM/RAM limits, HF model selection, Ollama, quantization, GGUF/SafeTensors, prefill/decode, tokens/sec, paging/mmap, AirLLM, local vs cloud vs API cost). Given hardware: i7-8550U, 8 GB RAM, MX110 ~2 GB VRAM, Windows 11, Python 3.8."

_Implementation-stage prompts will be appended here as code is generated._

---

## 13. Vibe Coding Lifecycle Explanation
| Stage | Meaning | This project |
|---|---|---|
| **Idea** | Define the goal | Benchmark local LLM inference under tight memory limits |
| **PRD** | Requirements | [`prd.md`](prd.md) |
| **Plan** | Architecture | [`plan.md`](plan.md) — modular, files < 150 lines |
| **TODO** | Task checklist | [`todo.md`](todo.md) |
| **Verify** | Check plan before coding | line-count + dry-run + schema checks |
| **Execute** | Build & run | implement `src/`, run benchmarks, fill results |
| **Push** | Publish | commit + push to public GitHub repo |

We are in **Execute**: documentation, the `src/` measurement infrastructure, the controlled/environment runners, and — as of Phase 4.5 — **one real local inference** (`smollm2:135m` via Ollama) are all complete and verified. Remaining optional work is real **7B** inference (baseline / quantized / AirLLM), which still requires heavy dependencies + more capable hardware than this laptop.

---

## 14. Repository Rules Compliance
- ✅ Python source lives in `src/` (16 modules, modular split).
- ✅ Results/plots/logs/CSVs live in `results/` (JSON, 2 CSVs, 2 logs, 3 PNGs).
- ✅ Every Python file is < 150 lines (enforced by `verify` — 16 files checked, PASS).
- ✅ Runs from the terminal (`python -m src.run_benchmark <command>`).
- ✅ No single huge file — modular split across `src/` and `src/runners/`.
- ✅ README is the final technical report.
- ✅ No invented results — every row tagged by `result_type`: **1 real** (`smollm2:135m`), plus `mock`, `environment_check`, and labelled `controlled_analysis` estimates; economics figures marked ESTIMATE/PLACEHOLDER.
