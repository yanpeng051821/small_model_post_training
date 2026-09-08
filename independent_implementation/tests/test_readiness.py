import json

import torch

from post_training_core.readiness import (
    _audit_check,
    _qwen_preflight_check,
    generate_server_preflight_report,
)


def _audit_manifest(*, review_hash="review", frozen_candidate=True):
    return {
        "accepted_count": 3,
        "length_statistics": {"sequence_tokens": {"count": 3}},
        "reason_counts": {"invalid": 1},
        "rejected_count": 1,
        "settings": {"validation_size": 1},
        "source": {"frozen_candidate": frozen_candidate, "revision": "abc"},
        "train": {"count": 2, "sha256": "train"},
        "validation": {"count": 1, "sha256": "validation"},
        "rejected": {"count": 1, "sha256": "rejected"},
        "review": {"count": 2, "sha256": review_hash},
    }


def test_audit_check_requires_full_manifest_equality_and_frozen_scan(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_audit_manifest()), encoding="utf-8")
    second.write_text(json.dumps(_audit_manifest()), encoding="utf-8")

    assert _audit_check(first, second).passed is True

    second.write_text(
        json.dumps(_audit_manifest(review_hash="different")),
        encoding="utf-8",
    )
    mismatch = _audit_check(first, second)
    assert mismatch.passed is False
    assert mismatch.evidence["mismatched_fields"] == ["review"]

    second.write_text(
        json.dumps(_audit_manifest(frozen_candidate=False)),
        encoding="utf-8",
    )
    assert _audit_check(first, second).passed is False


def test_qwen_preflight_is_bound_to_code_data_and_inference_assets(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "preflight_summary.json").write_text(
        json.dumps(
            {
                "audited_train_sha256": "train-hash",
                "checkpoint_inference_assets_complete": True,
                "final_status": "completed",
                "optimizer_steps": 2,
                "project_tree_sha256": "tree-hash",
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {"event": "validation", "optimizer_step": 0},
        {"event": "train", "optimizer_step": 1, "mean_loss": 1.0, "grad_norm": 1.0},
        {"event": "validation", "optimizer_step": 1},
        {"event": "train", "optimizer_step": 2, "mean_loss": 0.9, "grad_norm": 0.8},
        {"event": "validation", "optimizer_step": 2},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert _qwen_preflight_check(
        run_dir,
        expected_project_tree_sha256="tree-hash",
        expected_train_sha256="train-hash",
    ).passed
    assert not _qwen_preflight_check(
        run_dir,
        expected_project_tree_sha256="changed-tree",
        expected_train_sha256="train-hash",
    ).passed


def test_server_preflight_rejects_wrong_hash_and_writes_report(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    validation.write_text("{}\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock", encoding="utf-8")
    output = tmp_path / "evidence" / "server_preflight.json"
    output.parent.mkdir()

    report = generate_server_preflight_report(
        project_dir=tmp_path,
        train_artifact=train,
        validation_artifact=validation,
        expected_train_sha256="wrong",
        expected_validation_sha256="wrong",
        expected_uv_lock_sha256="wrong",
        expected_server_requirements_sha256=None,
        data_manifest_path=None,
        expected_data_manifest_sha256=None,
        expected_project_tree_sha256=None,
        minimum_free_bytes=1,
        output_path=output,
        require_flash_attention=False,
        require_eval_stack=False,
    )

    assert report["passed"] is False
    written = json.loads(output.read_text(encoding="utf-8"))
    checks = {check["name"]: check for check in written["checks"]}
    assert checks["train_artifact_hash"]["passed"] is False
    assert checks["validation_artifact_hash"]["passed"] is False
    assert checks["uv_lock_hash"]["passed"] is False
    assert checks["free_disk"]["passed"] is True
    assert checks["runtime_package_versions"]["passed"] is True
    assert checks["flash_attention_available"]["passed"] is True
    assert checks["frozen_evaluation_stack"]["passed"] is True
    assert checks["cuda_available"]["passed"] is torch.cuda.is_available()
