"""Run-directory and evidence helpers for controlled experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from post_training_core.config import ExperimentConfig


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _command_output(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_environment(project_dir: Path | None = None) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    project_dir = project_dir or Path(__file__).resolve().parents[2]
    git_status = _command_output(["git", "status", "--porcelain"], cwd=project_dir)
    return {
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "cuda_driver": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ),
        "packages": {
            package: _installed_version(package)
            for package in (
                "accelerate",
                "datasets",
                "flash-attn",
                "lighteval",
                "math-verify",
                "trl",
            )
        },
        "git": {
            "commit": _command_output(["git", "rev-parse", "HEAD"], cwd=project_dir),
            "dirty": bool(git_status) if git_status is not None else None,
        },
        "uv_lock_sha256": _file_sha256(project_dir / "uv.lock"),
    }


def initialize_run(config: ExperimentConfig) -> Path:
    run_dir = config.output_dir.resolve() / config.run_name
    if config.resume_from_checkpoint is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"cannot resume because run manifest is missing: {manifest_path}"
            )
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest["config_hash"] != config.semantic_hash():
            raise ValueError("resume config does not match the original run")
        update_run_status(
            run_dir,
            "resuming",
            resume_from_checkpoint=str(config.resume_from_checkpoint.resolve()),
        )
        return run_dir

    atomic_write_json(run_dir / "config.resolved.json", config.to_dict())
    atomic_write_json(run_dir / "environment.json", collect_environment())
    atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "config_hash": config.semantic_hash(),
            "created_at": utc_now(),
            "run_name": config.run_name,
            "status": "initialized",
        },
    )
    return run_dir


def update_run_status(
    run_dir: str | Path,
    status: str,
    **details: Any,
) -> None:
    manifest_path = Path(run_dir) / "run_manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    manifest.update(details)
    manifest["status"] = status
    manifest["updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
