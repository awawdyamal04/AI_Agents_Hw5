"""Benchmark runners for Assignment 05.

Phase 3A ships two *non-inference* runners that document expected behaviour and
this machine's limitations without downloading models or running real LLMs:

- ``env_runner``        — records an ``environment_check`` row (is Ollama
                          installed? Python version? available RAM?).
- ``controlled_runner`` — records ``controlled_analysis`` rows estimating the
                          memory footprint of baseline / quantized / AirLLM 7B
                          configs and the expected outcome on this hardware.

``base_runner`` holds the shared status constants and schema helper. Real
runners (baseline / quant / AirLLM) remain optional/pending — see ``todo.md``.
"""
