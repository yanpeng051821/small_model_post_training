"""Fail-fast environment and artifact checks before paid GPU work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from post_training_core.readiness import generate_server_preflight_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-artifact", type=Path, required=True)
    parser.add_argument("--validation-artifact", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=int, default=30)
    parser.add_argument(
        "--allow-missing-flash-attention",
        action="store_true",
        help="Only valid when the selected training config does not use flash_attention_2.",
    )
    parser.add_argument(
        "--allow-missing-eval-stack",
        action="store_true",
        help="Only valid for a training-only host that will not run B0/S1 evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/server_preflight.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    project_dir = Path(__file__).resolve().parents[1]
    with args.bundle_manifest.open(encoding="utf-8") as stream:
        bundle = json.load(stream)
    expected_files = bundle["project"]["files"]
    report = generate_server_preflight_report(
        project_dir=project_dir,
        train_artifact=args.train_artifact,
        validation_artifact=args.validation_artifact,
        expected_train_sha256=bundle["data"]["train"]["sha256"],
        expected_validation_sha256=bundle["data"]["validation"]["sha256"],
        expected_uv_lock_sha256=expected_files["uv.lock"],
        expected_server_requirements_sha256=expected_files[
            "requirements-server.txt"
        ],
        data_manifest_path=args.data_manifest,
        expected_data_manifest_sha256=bundle["data"]["manifest_sha256"],
        expected_project_tree_sha256=bundle["project"]["tree_sha256"],
        minimum_free_bytes=args.minimum_free_gib * 1024**3,
        output_path=args.output,
        require_flash_attention=not args.allow_missing_flash_attention,
        require_eval_stack=not args.allow_missing_eval_stack,
        human_review=bundle.get("human_review"),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
