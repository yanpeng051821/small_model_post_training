"""Evidence checks that gate paid GPU work."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata, util
from pathlib import Path
from typing import Any

import torch

from post_training_core.data import sha256_file
from post_training_core.experiment import atomic_write_json, utc_now
from post_training_core.provenance import runtime_file_hashes, runtime_tree_hash

EXPECTED_RUNTIME_VERSIONS = {
    "accelerate": "1.6.0",
    "datasets": "3.6.0",
    "torch": "2.6.0",
    "transformers": "4.52.3",
    "trl": "0.18.0",
}
EXPECTED_LIGHTEVAL_COMMIT = "d3da6b9bbf38104c8b5e1acc86f83541f9a502d1"


def _eval_stack_identity() -> tuple[bool, dict[str, Any]]:
    identity: dict[str, Any] = {}
    for package in ("vllm", "math-verify", "lighteval"):
        try:
            distribution = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            identity[package] = None
            continue
        record: dict[str, Any] = {"version": distribution.version}
        if package == "lighteval":
            direct_url = distribution.read_text("direct_url.json")
            if direct_url:
                try:
                    payload = json.loads(direct_url)
                    record["commit"] = payload.get("vcs_info", {}).get("commit_id")
                    record["url"] = payload.get("url")
                except json.JSONDecodeError:
                    record["commit"] = None
        identity[package] = record
    passed = (
        (identity.get("vllm") or {}).get("version") == "0.8.5.post1"
        and (identity.get("math-verify") or {}).get("version") == "0.5.2"
        and (identity.get("lighteval") or {}).get("commit")
        == EXPECTED_LIGHTEVAL_COMMIT
    )
    return passed, identity


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    evidence: Any


def _run_check(name: str, command: list[str], cwd: Path) -> GateCheck:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return GateCheck(
        name=name,
        passed=result.returncode == 0,
        evidence={
            "command": command,
            "returncode": result.returncode,
            "output_tail": output[-4_000:],
        },
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _artifact_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        name: manifest[name]["sha256"]
        for name in ("train", "validation", "rejected", "review")
    }


def _audit_check(first: Path, second: Path) -> GateCheck:
    first_manifest = _load_json(first)
    second_manifest = _load_json(second)
    first_hashes = _artifact_hashes(first_manifest)
    second_hashes = _artifact_hashes(second_manifest)
    deterministic_fields = (
        "accepted_count",
        "length_statistics",
        "reason_counts",
        "rejected_count",
        "settings",
        "source",
        "train",
        "validation",
        "rejected",
        "review",
    )
    mismatched_fields = [
        field
        for field in deterministic_fields
        if first_manifest.get(field) != second_manifest.get(field)
    ]
    full_frozen_scan = (
        first_manifest.get("source", {}).get("frozen_candidate") is True
        and second_manifest.get("source", {}).get("frozen_candidate") is True
    )
    return GateCheck(
        name="real_data_audit_is_reproducible",
        passed=not mismatched_fields and full_frozen_scan,
        evidence={
            "first_manifest": str(first.resolve()),
            "second_manifest": str(second.resolve()),
            "first_hashes": first_hashes,
            "second_hashes": second_hashes,
            "full_frozen_scan": full_frozen_scan,
            "mismatched_fields": mismatched_fields,
        },
    )


def _qwen_preflight_check(
    run_dir: Path,
    *,
    expected_project_tree_sha256: str,
    expected_train_sha256: str,
) -> GateCheck:
    summary = _load_json(run_dir / "preflight_summary.json")
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_steps = [row["optimizer_step"] for row in metrics if row["event"] == "train"]
    validation_steps = [
        row["optimizer_step"] for row in metrics if row["event"] == "validation"
    ]
    finite_metrics = all(
        isinstance(row.get("mean_loss"), (int, float))
        and isinstance(row.get("grad_norm"), (int, float))
        for row in metrics
        if row["event"] == "train"
    )
    passed = (
        summary.get("final_status") == "completed"
        and summary.get("optimizer_steps") == 2
        and train_steps == [1, 2]
        and validation_steps == [0, 1, 2]
        and finite_metrics
        and summary.get("checkpoint_inference_assets_complete") is True
        and summary.get("project_tree_sha256") == expected_project_tree_sha256
        and summary.get("audited_train_sha256") == expected_train_sha256
    )
    return GateCheck(
        name="real_qwen_bf16_train_and_cross_process_resume",
        passed=passed,
        evidence={
            "run_dir": str(run_dir.resolve()),
            "summary": summary,
            "train_steps": train_steps,
            "validation_steps": validation_steps,
        },
    )


def generate_local_readiness_report(
    *,
    project_dir: Path,
    first_audit_manifest: Path,
    second_audit_manifest: Path,
    qwen_preflight_run: Path,
    output_path: Path,
) -> dict[str, Any]:
    project_tree_sha256 = runtime_tree_hash(runtime_file_hashes(project_dir))
    first_audit = _load_json(first_audit_manifest)
    checks = [
        _run_check(
            "ruff",
            [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
            project_dir,
        ),
        _run_check(
            "unit_tests",
            [sys.executable, "-m", "pytest", "-q"],
            project_dir,
        ),
        _run_check(
            "pinned_qwen_tokenizer_integration",
            [sys.executable, "-m", "pytest", "-q", "-m", "integration"],
            project_dir,
        ),
        _audit_check(first_audit_manifest, second_audit_manifest),
        _qwen_preflight_check(
            qwen_preflight_run,
            expected_project_tree_sha256=project_tree_sha256,
            expected_train_sha256=first_audit["train"]["sha256"],
        ),
    ]
    checkpoint_bytes = _load_json(
        qwen_preflight_run / "preflight_summary.json"
    )["checkpoint_bytes_before_cleanup"]
    report = {
        "created_at": utc_now(),
        "gate": "local_readiness",
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            ),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": {
                package: metadata.version(package)
                for package in EXPECTED_RUNTIME_VERSIONS
            },
        },
        "frozen_inputs": {
            "project_tree_sha256": project_tree_sha256,
            "uv_lock_sha256": sha256_file(project_dir / "uv.lock"),
        },
        "resource_observations": {
            "checkpoint_bytes": checkpoint_bytes,
            "minimum_recommended_server_free_bytes": checkpoint_bytes * 4,
            "project_disk_free_bytes": shutil.disk_usage(project_dir).free,
        },
    }
    atomic_write_json(output_path, report)
    return report


def generate_server_preflight_report(
    *,
    project_dir: Path,
    train_artifact: Path,
    validation_artifact: Path,
    expected_train_sha256: str,
    expected_validation_sha256: str,
    expected_uv_lock_sha256: str,
    expected_server_requirements_sha256: str | None,
    data_manifest_path: Path | None,
    expected_data_manifest_sha256: str | None,
    expected_project_tree_sha256: str | None,
    minimum_free_bytes: int,
    output_path: Path,
    require_flash_attention: bool = True,
    require_eval_stack: bool = True,
    human_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    free_bytes = shutil.disk_usage(output_path.parent).free
    installed_versions = {}
    package_versions_match = True
    for package, expected in EXPECTED_RUNTIME_VERSIONS.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = None
        installed_versions[package] = {"expected": expected, "installed": installed}
        package_versions_match &= (
            installed is not None and installed.split("+", maxsplit=1)[0] == expected
        )
    eval_stack_passed, eval_stack_identity = _eval_stack_identity()
    flash_attention_version = None
    try:
        flash_attention_version = metadata.version("flash-attn")
    except metadata.PackageNotFoundError:
        pass
    server_requirements = project_dir / "requirements-server.txt"
    actual_project_tree_sha256 = (
        runtime_tree_hash(runtime_file_hashes(project_dir))
        if expected_project_tree_sha256 is not None
        else None
    )
    checks = [
        GateCheck(
            "human_data_review",
            human_review is None
            or (
                human_review.get("passed") is True
                and human_review.get("status") == "passed"
                and human_review.get("unresolved_count") == 0
            ),
            human_review,
        ),
        GateCheck(
            "runtime_package_versions",
            package_versions_match,
            installed_versions,
        ),
        GateCheck(
            "flash_attention_available",
            not require_flash_attention
            or (
                util.find_spec("flash_attn") is not None
                and flash_attention_version == "2.7.4.post1"
            ),
            {
                "required": require_flash_attention,
                "expected_version": "2.7.4.post1",
                "installed_version": flash_attention_version,
            },
        ),
        GateCheck(
            "frozen_evaluation_stack",
            not require_eval_stack or eval_stack_passed,
            {"required": require_eval_stack, **eval_stack_identity},
        ),
        GateCheck("cuda_available", torch.cuda.is_available(), torch.version.cuda),
        GateCheck(
            "bf16_supported",
            torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        ),
        GateCheck(
            "train_artifact_hash",
            train_artifact.is_file()
            and sha256_file(train_artifact) == expected_train_sha256,
            str(train_artifact.resolve()),
        ),
        GateCheck(
            "validation_artifact_hash",
            validation_artifact.is_file()
            and sha256_file(validation_artifact) == expected_validation_sha256,
            str(validation_artifact.resolve()),
        ),
        GateCheck(
            "uv_lock_hash",
            (project_dir / "uv.lock").is_file()
            and sha256_file(project_dir / "uv.lock") == expected_uv_lock_sha256,
            (
                sha256_file(project_dir / "uv.lock")
                if (project_dir / "uv.lock").is_file()
                else None
            ),
        ),
        GateCheck(
            "server_requirements_hash",
            expected_server_requirements_sha256 is None
            or (
                server_requirements.is_file()
                and sha256_file(server_requirements)
                == expected_server_requirements_sha256
            ),
            {
                "required": expected_server_requirements_sha256 is not None,
                "actual": (
                    sha256_file(server_requirements)
                    if server_requirements.is_file()
                    else None
                ),
                "expected": expected_server_requirements_sha256,
            },
        ),
        GateCheck(
            "data_manifest_hash",
            expected_data_manifest_sha256 is None
            or (
                data_manifest_path is not None
                and data_manifest_path.is_file()
                and sha256_file(data_manifest_path) == expected_data_manifest_sha256
            ),
            {
                "path": (
                    str(data_manifest_path.resolve())
                    if data_manifest_path is not None
                    else None
                ),
                "expected": expected_data_manifest_sha256,
            },
        ),
        GateCheck(
            "project_tree_hash",
            expected_project_tree_sha256 is None
            or actual_project_tree_sha256 == expected_project_tree_sha256,
            {
                "actual": actual_project_tree_sha256,
                "expected": expected_project_tree_sha256,
            },
        ),
        GateCheck(
            "free_disk",
            free_bytes >= minimum_free_bytes,
            {"available_bytes": free_bytes, "required_bytes": minimum_free_bytes},
        ),
    ]
    report = {
        "created_at": utc_now(),
        "gate": "server_preflight",
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        },
    }
    atomic_write_json(output_path, report)
    return report
