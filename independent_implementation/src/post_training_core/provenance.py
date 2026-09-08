"""Deterministic project and data identities for server handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from post_training_core.data import sha256_file
from post_training_core.experiment import atomic_write_json, utc_now

ROOT_RUNTIME_FILES = ("pyproject.toml", "requirements-server.txt", "uv.lock")
RUNTIME_TREES = ("configs", "scripts", "src", "tests")


def runtime_file_hashes(project_dir: Path) -> dict[str, str]:
    project_dir = project_dir.resolve()
    files = []
    for name in ROOT_RUNTIME_FILES:
        path = project_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"required runtime file is missing: {path}")
        files.append(path)
    for tree in RUNTIME_TREES:
        root = project_dir / tree
        if not root.is_dir():
            raise FileNotFoundError(f"required runtime tree is missing: {root}")
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    return {
        path.relative_to(project_dir).as_posix(): sha256_file(path)
        for path in sorted(files, key=lambda item: item.relative_to(project_dir).as_posix())
    }


def runtime_tree_hash(file_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative_path, file_hash in sorted(file_hashes.items()):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_server_bundle_manifest(
    *,
    project_dir: Path,
    data_manifest_path: Path,
    review_summary_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    with data_manifest_path.open(encoding="utf-8") as stream:
        data_manifest = json.load(stream)
    if data_manifest.get("source", {}).get("frozen_candidate") is not True:
        raise ValueError("data manifest is not a full frozen candidate")
    artifact_dir = data_manifest_path.parent
    for name in ("train", "validation", "rejected", "review"):
        filename = "review_samples.jsonl" if name == "review" else f"{name}.jsonl"
        actual = sha256_file(artifact_dir / filename)
        if actual != data_manifest[name]["sha256"]:
            raise ValueError(f"data artifact hash mismatch while bundling: {filename}")

    with review_summary_path.open(encoding="utf-8") as stream:
        review_summary = json.load(stream)
    decision_artifact = Path(review_summary.get("decision_artifact", ""))
    review_passed = (
        review_summary.get("passed") is True
        and review_summary.get("status") == "passed"
        and review_summary.get("completed_count") == data_manifest["review"]["count"]
        and review_summary.get("expected_count") == data_manifest["review"]["count"]
        and review_summary.get("unresolved_count") == 0
        and review_summary.get("review_artifact_sha256")
        == data_manifest["review"]["sha256"]
        and decision_artifact.is_file()
        and review_summary.get("decision_artifact_sha256")
        == sha256_file(decision_artifact)
    )
    if not review_passed:
        raise ValueError("human review is incomplete, unresolved, or unverifiable")

    file_hashes = runtime_file_hashes(project_dir)
    manifest = {
        "created_at": utc_now(),
        "data": {
            "manifest_sha256": sha256_file(data_manifest_path),
            "source": data_manifest["source"],
            **{
                name: data_manifest[name]
                for name in ("train", "validation", "rejected", "review")
            },
        },
        "human_review": {
            **review_summary,
            "summary_path": str(review_summary_path.resolve()),
            "summary_sha256": sha256_file(review_summary_path),
        },
        "project": {
            "files": file_hashes,
            "tree_sha256": runtime_tree_hash(file_hashes),
        },
    }
    atomic_write_json(output_path, manifest)
    return manifest
