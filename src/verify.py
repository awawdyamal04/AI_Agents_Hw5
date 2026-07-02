"""Project self-check for Assignment 05 — Phase 2.

`verify` runs a set of structural checks and prints a clear PASS/FAIL per
check plus an overall verdict. It does NOT run models. Checks:
- required source + doc files exist,
- src/ and results/ directories exist,
- every Python file under src/ is < 150 lines,
- the benchmark CSV exists (created by dry-run).

Returns process exit code 0 (all pass) or 1 (any fail). (< 150 lines)
"""

from pathlib import Path

from . import config

MAX_LINES = 150

REQUIRED_FILES = [
    "src/__init__.py",
    "src/config.py",
    "src/hardware.py",
    "src/metrics.py",
    "src/results_io.py",
    "src/plots.py",
    "src/economics.py",
    "src/run_benchmark.py",
    "src/verify.py",
    # Phase 3A runners package (non-inference: env-check + controlled analysis).
    "src/runners/__init__.py",
    "src/runners/base_runner.py",
    "src/runners/env_runner.py",
    "src/runners/controlled_runner.py",
    "README.md",
    "todo.md",
    "requirements.txt",
]


def _check(label, ok, detail=""):
    """Print one PASS/FAIL line and return ok (bool)."""
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {label}"
    if detail:
        line += f" - {detail}"
    print(line)
    return ok


def _count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def run():
    """Run all checks; print PASS/FAIL; return 0 if all pass else 1."""
    root = config.PROJECT_ROOT
    print("=== Verify (structural checks - no model inference) ===")
    results = []

    # 1. Directories.
    results.append(_check("src/ directory exists", (root / "src").is_dir()))
    results.append(_check("results/ directory exists",
                          config.RESULTS_DIR.is_dir()))

    # 2. Required files.
    for rel in REQUIRED_FILES:
        results.append(_check(f"file exists: {rel}", (root / rel).is_file()))

    # 3. Python line-count limit (< 150 lines) across src/.
    py_files = sorted((root / "src").rglob("*.py"))
    py_files = [p for p in py_files if "__pycache__" not in p.parts]
    over = []
    for p in py_files:
        n = _count_lines(p)
        if n >= MAX_LINES:
            over.append(f"{p.relative_to(root)} ({n})")
    results.append(_check(
        f"all src/*.py under {MAX_LINES} lines",
        not over,
        "over limit: " + ", ".join(over) if over else
        f"{len(py_files)} files checked"))

    # 4. Benchmark CSV exists (produced by dry-run).
    csv_ok = config.BENCHMARK_CSV.is_file()
    results.append(_check(
        "benchmark CSV exists (run dry-run first)",
        csv_ok,
        str(config.BENCHMARK_CSV.relative_to(root)) if csv_ok else
        "missing — run: python -m src.run_benchmark dry-run"))

    passed = sum(1 for r in results if r)
    total = len(results)
    all_ok = passed == total
    print("-" * 48)
    print(f"{'OVERALL: PASS' if all_ok else 'OVERALL: FAIL'}  "
          f"({passed}/{total} checks passed)")
    return 0 if all_ok else 1


if __name__ == "__main__":  # allow: python -m src.verify
    raise SystemExit(run())
