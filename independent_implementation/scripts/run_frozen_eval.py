"""Run or dry-run one frozen Gate 0B LightEval suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from post_training_core.eval_runner import build_lighteval_command, run_lighteval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/gate0b/evaluation.yaml"),
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="Suite key from the selected evaluation contract.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--allow-expensive-suite",
        action="store_true",
        help="Acknowledge a recorded decision to run a guarded evaluation suite.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command, evidence = build_lighteval_command(
        config_path=args.contract,
        suite_name=args.suite,
        model=args.model,
        model_revision=args.model_revision,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        allow_expensive_suite=args.allow_expensive_suite,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return run_lighteval(
        command=command,
        evidence=evidence,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
