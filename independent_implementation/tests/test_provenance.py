import json

import pytest

from post_training_core.data import sha256_file
from post_training_core.provenance import (
    build_server_bundle_manifest,
    runtime_file_hashes,
    runtime_tree_hash,
)


def _project(tmp_path):
    for name in ("pyproject.toml", "requirements-server.txt", "uv.lock"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    for tree in ("configs", "scripts", "src", "tests"):
        (tmp_path / tree).mkdir()
        (tmp_path / tree / "file.txt").write_text(tree, encoding="utf-8")


def test_runtime_tree_hash_changes_with_runtime_code(tmp_path):
    _project(tmp_path)
    before = runtime_tree_hash(runtime_file_hashes(tmp_path))

    (tmp_path / "src" / "file.txt").write_text("changed", encoding="utf-8")
    after = runtime_tree_hash(runtime_file_hashes(tmp_path))

    assert before != after


def test_bundle_verifies_every_data_artifact(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _project(project)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    entries = {}
    for name in ("train", "validation", "rejected", "review"):
        filename = "review_samples.jsonl" if name == "review" else f"{name}.jsonl"
        path = artifacts / filename
        path.write_text(f'{{"name":"{name}"}}\n', encoding="utf-8")
        entries[name] = {"count": 1, "sha256": sha256_file(path)}
    manifest_path = artifacts / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"source": {"frozen_candidate": True}, **entries},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text('{"verdict":"keep"}\n', encoding="utf-8")
    review_summary = tmp_path / "decisions.summary.json"
    review_summary.write_text(
        json.dumps(
            {
                "completed_count": 1,
                "decision_artifact": str(decisions.resolve()),
                "decision_artifact_sha256": sha256_file(decisions),
                "expected_count": 1,
                "passed": True,
                "review_artifact_sha256": entries["review"]["sha256"],
                "status": "passed",
                "unresolved_count": 0,
            }
        ),
        encoding="utf-8",
    )

    bundle = build_server_bundle_manifest(
        project_dir=project,
        data_manifest_path=manifest_path,
        review_summary_path=review_summary,
        output_path=tmp_path / "bundle.json",
    )

    assert bundle["data"]["manifest_sha256"] == sha256_file(manifest_path)
    assert bundle["project"]["tree_sha256"]

    (artifacts / "train.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train.jsonl"):
        build_server_bundle_manifest(
            project_dir=project,
            data_manifest_path=manifest_path,
            review_summary_path=review_summary,
            output_path=tmp_path / "other.json",
        )


def test_bundle_refuses_incomplete_human_review(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _project(project)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    entries = {}
    for name in ("train", "validation", "rejected", "review"):
        filename = "review_samples.jsonl" if name == "review" else f"{name}.jsonl"
        path = artifacts / filename
        path.write_text("{}\n", encoding="utf-8")
        entries[name] = {"count": 1, "sha256": sha256_file(path)}
    manifest = artifacts / "data_manifest.json"
    manifest.write_text(
        json.dumps({"source": {"frozen_candidate": True}, **entries}),
        encoding="utf-8",
    )
    summary = tmp_path / "review.summary.json"
    summary.write_text(json.dumps({"passed": False}), encoding="utf-8")

    with pytest.raises(ValueError, match="human review"):
        build_server_bundle_manifest(
            project_dir=project,
            data_manifest_path=manifest,
            review_summary_path=summary,
            output_path=tmp_path / "bundle.json",
        )
