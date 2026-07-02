# PRD — Product Requirements Document

## Project Title
**Assignment 05 — Running a Massive LLM Locally: AirLLM, Quantization and Performance Benchmarking**

Course: *Orchestration of AI Agents* — Lecture 08L (Local inference and training of large language models).

---

## 1. Assignment Goal
Prove **practical and technical understanding** of running a large language model (LLM) locally on limited hardware.

The goal is **NOT** to build the best text-generation model. The goal is to:
- Analyze local inference constraints (VRAM, RAM, CPU vs GPU).
- Measure real performance (latency, throughput, tokens/sec, memory).
- Explain failures and limitations honestly.
- Compare three approaches: **baseline local**, **quantization**, and **AirLLM-style layer-by-layer** inference.
- Connect every observation back to Lecture 08L concepts.

> On this hardware a large LLM is *expected* to fail, be extremely slow, or not fit in VRAM/RAM without quantization or memory-aware techniques. **That expected failure is the core of the experiment, not a weakness.**

---

## 2. Problem Definition
Large transformer models require memory proportional to their parameter count plus the KV cache, which grows with context length. Modern LLMs (7B+ parameters) need several GB of memory just to hold weights in FP16.

The target machine has **8 GB system RAM** and only **~2 GB VRAM** (NVIDIA MX110). Loading a 7B FP16 model (~14 GB) is impossible without:
- **Quantization** (reduce bits per weight: FP16 → INT8/INT4 → GGUF).
- **Memory-aware execution** such as **AirLLM** (load and run one transformer layer at a time from disk, freeing memory between layers).

The problem this project solves: **systematically document, benchmark, and explain what happens when you push a laptop past its memory limits, and which techniques make local inference feasible.**

---

## 3. Inputs
- **Hardware profile** (auto-collected via `psutil` + static specs).
- **Model identifiers** from Hugging Face / Ollama (e.g., a small model such as `TinyLlama-1.1B`, `Qwen2.5-0.5B`, or `gpt2`; and a "too big" reference model for the controlled AirLLM analysis).
- **Prompts**: a fixed set of benchmark prompts (short + medium length) stored in config so runs are reproducible.
- **Generation settings**: `max_new_tokens`, temperature, number of runs per configuration.

---

## 4. Outputs
- **Benchmark CSV files** in `results/` (one row per run/configuration).
- **Plots** (PNG) in `results/`: tokens/sec comparison, load-time comparison, memory usage.
- **Logs** in `results/*.log` capturing raw run output, errors, and failure reasons.
- **README.md** acting as the **final technical report**.
- **Economic analysis table**: Local On-Prem vs Cloud GPU vs API.

---

## 5. Required Methods / Tools
| Concept (Lecture 08L) | Tool / Method |
|---|---|
| Local inference | `transformers` (CPU), Ollama (optional) |
| Quantization | GGUF models via `llama-cpp-python` / Ollama; bitsandbytes INT8 (optional) |
| AirLLM | `airllm` package if runnable, else controlled documented analysis |
| Memory / paging / mmap | `psutil`, OS virtual memory observation |
| Metrics collection | Custom Python timing + `psutil` sampling |
| Results & plots | `pandas`, `matplotlib` |
| API/HTTP checks | `requests` |

Lightweight-only dependencies are forced now (`psutil`, `pandas`, `matplotlib`, `requests`). Heavy/optional packages (`transformers`, `torch`, `airllm`, `llama-cpp-python`, `bitsandbytes`) are documented as optional in the README and installed only if the hardware/run allows.

---

## 6. Formulas / Metrics
- **Load time** = `t_model_ready − t_load_start` (seconds).
- **Time to first token (TTFT)** = `t_first_token − t_generate_start` (seconds) — the *prefill* phase indicator.
- **Total generation time** = `t_generation_end − t_generate_start` (seconds).
- **Decode throughput (tokens/sec)** = `generated_tokens / (total_generation_time − TTFT)`.
- **Overall throughput** = `generated_tokens / total_generation_time`.
- **Peak RAM** = max resident memory sampled during run (MB).
- **VRAM limitation** = recorded as fit / OOM / not-attempted with reason.
- **Success flag** = `success | slow | failure`, with `error_reason` text.

Prefill vs Decode distinction (Lecture 08L) is captured via TTFT (prefill) vs decode throughput.

---

## 7. Hardware Description
| Component | Spec |
|---|---|
| OS | Windows 11 Pro (10.0.26200) |
| Laptop | ASUS VivoBook 15 X540UBR |
| CPU | Intel Core i7-8550U @ 1.80 GHz |
| Cores / Threads | 4 cores / 8 logical processors |
| RAM | 8 GB |
| GPU 1 | Intel UHD Graphics 620 (~1 GB shared) |
| GPU 2 | NVIDIA GeForce MX110 (~2 GB VRAM) |
| Disk | SanDisk 256 GB |
| Python | 3.8.0 |

**Implication:** ~2 GB VRAM cannot hold a 7B model. 8 GB RAM is shared with the OS, so even CPU inference of medium models risks paging/swap. This constraint drives the entire experiment design.

---

## 8. Lecture Constraints (08L Concepts to Demonstrate)
- CPU vs GPU trade-off on constrained hardware.
- VRAM and RAM limits as hard ceilings.
- Hugging Face model selection for size feasibility.
- Ollama local inference path.
- Quantization (bits, GGUF, SafeTensors formats).
- Prefill vs Decode phases (TTFT vs decode tokens/sec).
- Latency, throughput, tokens/sec definitions.
- Virtual memory, paging, and mmap in loading large weights.
- AirLLM layer-by-layer streaming from disk.
- Cost comparison: Local On-Prem vs Cloud GPU vs API.
- Background: transformer architecture, attention, context window, KV cache → why memory scales.

---

## 9. Evaluation Method
1. Run each configuration (baseline / quantized / AirLLM-or-analysis) N times.
2. Collect metrics into CSV.
3. Aggregate (mean, min, max) with `pandas`.
4. Plot comparisons.
5. Interpret results against expectations and lecture concepts.
6. Report failures honestly with `error_reason` — a failure is a valid, expected result.

---

## 10. Success Criteria
- [ ] Hardware is auto-documented and reproducible.
- [ ] At least one small model runs locally and is benchmarked.
- [ ] Baseline vs quantized behavior is compared (numbers or documented failure).
- [ ] AirLLM is either run or rigorously analyzed as a controlled experiment.
- [ ] All required metrics are collected or explicitly marked "not measurable + reason".
- [ ] Results tables + plots exist in `results/` (after running commands).
- [ ] Economic analysis (Local vs Cloud GPU vs API) is included.
- [ ] Every result is tied to a Lecture 08L concept.
- [ ] No fake results — unrun items are clearly labelled "to be generated".

---

## 11. Final Deliverables
- `prd.md`, `plan.md`, `todo.md` — lifecycle documentation.
- `README.md` — final technical report (with Mermaid diagrams, results, discussion, conclusion, AI prompts, lifecycle).
- `requirements.txt` — lightweight deps.
- `.gitignore`.
- `src/` — modular Python code (each file < 150 lines) — **implemented later**.
- `results/` — CSVs, plots, logs — **generated later**.
- Public GitHub repository submission.
