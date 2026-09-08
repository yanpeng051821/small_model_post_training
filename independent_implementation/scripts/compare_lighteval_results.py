"""Create paired metric and changed-sample evidence for B0 versus S1 LightEval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from post_training_core.eval_comparison import compare_lighteval_runs
from post_training_core.experiment import atomic_write_json, utc_now


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--trained-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary, changed_records = compare_lighteval_runs(
        args.baseline_root,
        args.trained_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    changed_path = args.output_dir / "changed_samples.jsonl"
    changed_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
            for record in changed_records
        ),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        **summary,
        "created_at": utc_now(),
        "changed_samples": str(changed_path.resolve()),
    }
    atomic_write_json(args.output_dir / "summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
