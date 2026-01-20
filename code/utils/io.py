"""Utility functions for writing experiment artefacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence


def ensure_dir(path: Path) -> None:
    """Create the parent directory for a file path if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Mapping) -> None:
    """Write a JSON payload with UTF-8 encoding."""
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_csv_row(path: Path, fieldnames: Sequence[str], row: Mapping) -> None:
    """Append a single row to a CSV file, creating it with headers if needed."""
    ensure_dir(path)
    is_new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_log(path: Path, message: str, *, timestamp: bool = True) -> None:
    """Append a line to a plain-text log file."""
    ensure_dir(path)
    stamp = f"[{datetime.now().isoformat(timespec='seconds')}] " if timestamp else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}{message}\n")
