"""Assignment 05 — Running a Massive LLM Locally.

Phase 1 skeleton package. This package currently provides:
- config      : central constants (paths, static hardware profile, schema)
- hardware    : CPU/RAM probe (psutil) + static hardware profile snapshot
- results_io  : CSV / JSON / log writers (single source of I/O truth)
- run_benchmark : lightweight argparse CLI (hardware / dry-run)

No real model inference is implemented yet. Model runners are pending.
"""

__version__ = "0.1.0"  # Phase 1: skeleton only, no model inference
