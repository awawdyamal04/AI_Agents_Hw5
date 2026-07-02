"""Result I/O for Assignment 05 — the single source of truth for writing files.

Provides small helpers to:
- save a dict as pretty JSON (hardware snapshot),
- append a benchmark result dict as a CSV row (fixed schema),
- append lines to a log file,
- load a benchmark CSV back into a list of dict rows.

Uses only the standard library so it stays lightweight. (< 150 lines)
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from . import config


def _ensure_parent(path):
    """Make sure the parent directory of `path` exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    """Write `data` as indented JSON to `path`. Returns the path."""
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return Path(path)


def _migrate_header(path):
    """Rewrite an existing CSV whose header predates the current schema.

    If the file's header does not match config.CSV_COLUMNS (e.g. an older run
    lacked the `result_type` column), reload every row and rewrite the file
    with the current columns, filling any newly-added column with a blank.
    Existing rows are preserved; nothing is fabricated.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == config.CSV_COLUMNS:
            return
        rows = list(reader)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in config.CSV_COLUMNS})


def append_csv_row(row, path=None):
    """Append one benchmark row to the CSV, writing a header if new.

    `row` is a dict; missing columns are left blank and unknown keys ignored,
    so every file written matches config.CSV_COLUMNS exactly.
    """
    path = Path(path or config.BENCHMARK_CSV)
    _ensure_parent(path)
    _migrate_header(path)

    # Fill/normalize against the fixed schema.
    normalized = {col: row.get(col, "") for col in config.CSV_COLUMNS}
    if not normalized.get("timestamp"):
        normalized["timestamp"] = datetime.now().isoformat(timespec="seconds")

    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized)
    return path


def append_log(message, path=None):
    """Append a timestamped line to a log file. Returns the path."""
    path = Path(path or config.DRY_RUN_LOG)
    _ensure_parent(path)
    stamp = datetime.now().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")
    return path


def load_benchmark_csv(path=None):
    """Load the benchmark CSV into a list of dict rows (in file order).

    Returns [] if the file does not exist or is empty. Uses the stdlib csv
    module so no pandas dependency is required just to read results.
    """
    path = Path(path or config.BENCHMARK_CSV)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
