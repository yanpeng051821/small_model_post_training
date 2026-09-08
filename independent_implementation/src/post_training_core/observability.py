"""Durable metrics and failure evidence for SFT runs."""

from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

from post_training_core.experiment import atomic_write_json, utc_now


class JsonlMetricWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        for name, value in event.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"metric {name} must be finite")
        payload = {"recorded_at": utc_now(), **event}
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


def write_failure_snapshot(
    run_dir: str | Path,
    error: BaseException,
    **context: Any,
) -> Path:
    path = Path(run_dir) / "failure.json"
    atomic_write_json(
        path,
        {
            "context": context,
            "error_type": type(error).__name__,
            "message": str(error),
            "notes": list(getattr(error, "__notes__", [])),
            "recorded_at": utc_now(),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        },
    )
    return path
