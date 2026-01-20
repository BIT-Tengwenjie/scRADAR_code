"""Centralised logging helpers used across scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger

from config import CONFIG


def setup_logging(
    run_name: str,
    per_dataset: Optional[Iterable[str]] = None,
    *,
    level: str = "INFO",
) -> Path:
    """Configure loguru sinks for a run and return the base log directory."""
    base_dir = (Path(CONFIG["log_root"]).expanduser() / run_name).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=level)

    logger.add(
        base_dir / "run.log",
        level=level,
        rotation="10 MB",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    if per_dataset:
        for dataset in per_dataset:
            safe_name = str(dataset).replace("/", "_")
            logger.add(
                base_dir / f"{safe_name}.log",
                level=level,
                rotation="5 MB",
                enqueue=True,
                backtrace=False,
                diagnose=False,
            )

    return base_dir
