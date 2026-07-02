"""Assignment 05 — Running a Massive LLM Locally.

Phase 2 measurement-core package. This package currently provides:
- config      : central constants (paths, static hardware profile, schema)
- hardware    : CPU/RAM probe (psutil) + static hardware profile snapshot
- metrics     : timers, RAM sampler, token/throughput metrics, row builder
- results_io  : CSV / JSON / log writers + CSV loader (source of I/O truth)
- plots       : bar charts from existing CSV rows (mock rows labelled)
- economics   : estimated local/cloud/API cost-comparison template
- verify      : structural PASS/FAIL self-checks (incl. line limits)
- run_benchmark : argparse CLI (hardware/dry-run/plots/economics/verify)

No real model inference is implemented yet. Model runners are pending.
"""

__version__ = "0.2.0"  # Phase 2: measurement core, still no model inference
