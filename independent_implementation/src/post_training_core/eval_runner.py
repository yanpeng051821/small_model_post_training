"""Frozen LightEval command construction and invocation evidence."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import yaml

from post_training_core.experiment import atomic_write_json, utc_now


def _load_tasks(suite: dict[str, Any], config_dir: Path) -> str:
    if "task" in suite:
        return str(suite["task"])
    task_file = config_dir / suite["task_file"]
    tasks = [
        line.strip()
        for line in task_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not tasks:
        raise ValueError(f"evaluation task file is empty: {task_file}")
    return ",".join(tasks)


def _render_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, dict):
        body = ",".join(f"{key}:{_render_value(item)}" for key, item in value.items())
        return f"{{{body}}}"
    return str(value)


def build_lighteval_command(
    *,
    config_path: str | Path,
    suite_name: str,
    model: str,
    model_revision: str | None,
    output_dir: str | Path,
    max_samples: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as stream:
        contract = yaml.safe_load(stream)
    suites = contract.get("suites", {})
    if suite_name not in suites:
        raise ValueError(f"unknown evaluation suite: {suite_name}")
    suite = suites[suite_name]
    model_args = {"model_name": model, **suite["model_args"]}
    if model_revision is not None:
        model_args["revision"] = model_revision
    rendered_model_args = ",".join(
        f"{key}={_render_value(value)}" for key, value in model_args.items()
    )
    tasks = _load_tasks(suite, path.parent)
    command = ["lighteval", "vllm", rendered_model_args, tasks]
    if suite.get("use_chat_template", False):
        command.append("--use-chat-template")
    command.extend(["--output-dir", str(Path(output_dir).resolve()), "--save-details"])
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        command.extend(["--max-samples", str(max_samples)])
    evidence = {
        "contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "contract_path": str(path),
        "contract_version": contract["contract_version"],
        "lighteval_commit": contract["software"]["lighteval_commit"],
        "math_verify": contract["software"]["math_verify"],
        "model": model,
        "model_revision": model_revision,
        "model_args": suite["model_args"],
        "suite": suite_name,
        "tasks": tasks.split(","),
        "use_chat_template": suite.get("use_chat_template", False),
        "command": command,
    }
    return command, evidence


def run_lighteval(
    *,
    command: list[str],
    evidence: dict[str, Any],
    output_dir: str | Path,
    dry_run: bool,
) -> int:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "invocation_manifest.json"
    manifest = {
        **evidence,
        "created_at": utc_now(),
        "status": "dry_run" if dry_run else "running",
    }
    atomic_write_json(manifest_path, manifest)
    if dry_run:
        return 0
    result = subprocess.run(command, check=False)
    atomic_write_json(
        manifest_path,
        {
            **manifest,
            "completed_at": utc_now(),
            "returncode": result.returncode,
            "status": "completed" if result.returncode == 0 else "failed",
        },
    )
    return result.returncode
