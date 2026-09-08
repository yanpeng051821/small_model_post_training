"""Generate the evidence gate required before renting a GPU server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from post_training_core.readiness import generate_local_readiness_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-audit-manifest", type=Path, required=True)
    parser.add_argument("--second-audit-manifest", type=Path, required=True)
    parser.add_argument("--qwen-preflight-run", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/local_readiness_report.json"),
    )
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    report = generate_local_readiness_report(
        project_dir=project_dir,
        first_audit_manifest=args.first_audit_manifest,
        second_audit_manifest=args.second_audit_manifest,
        qwen_preflight_run=args.qwen_preflight_run,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
