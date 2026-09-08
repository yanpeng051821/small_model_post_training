"""Freeze code/config and audited-data identities before server upload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from post_training_core.provenance import build_server_bundle_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--review-summary", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/server_bundle_manifest.json"),
    )
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_server_bundle_manifest(
        project_dir=project_dir,
        data_manifest_path=args.data_manifest.resolve(),
        review_summary_path=args.review_summary.resolve(),
        output_path=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
