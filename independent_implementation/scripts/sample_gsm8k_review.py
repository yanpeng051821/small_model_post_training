"""Build a deterministic, stratified GSM8K paired-review bundle."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from post_training_core.data import sha256_file
from post_training_core.experiment import atomic_write_json, utc_now
from post_training_core.gsm8k_analysis import classify_pair, fixed_stratified_sample


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _complete_pairs(
    changed_records: list[dict[str, Any]], trained_details: Path
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to reconstruct identical pairs") from exc

    by_index = {record["row_index"]: record for record in changed_records}
    rows = pq.read_table(
        trained_details, columns=["example", "gold", "predictions", "metrics"]
    ).to_pylist()
    complete = []
    for row_index, trained in enumerate(rows):
        if row_index in by_index:
            complete.append({**by_index[row_index], "pair_source": "changed_samples"})
            continue
        complete.append(
            {
                "task": "leaderboard|gsm8k|4",
                "row_index": row_index,
                "example": trained["example"],
                "gold": trained["gold"],
                "baseline_predictions": trained["predictions"],
                "trained_predictions": trained["predictions"],
                "baseline_metrics": trained["metrics"],
                "trained_metrics": trained["metrics"],
                "metric_deltas_s1_minus_b0": {"qem": 0.0},
                "pair_source": "reconstructed_identical_from_s1",
            }
        )
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed-samples", type=Path, required=True)
    parser.add_argument("--trained-details", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-outcome", type=int, default=18)
    parser.add_argument("--seed", default="gsm8k-target-capability-v1")
    args = parser.parse_args()

    complete = _complete_pairs(_read_jsonl(args.changed_samples), args.trained_details)
    if len(complete) != 1319:
        raise ValueError(f"expected 1319 complete GSM8K pairs, found {len(complete)}")
    selected, strata = fixed_stratified_sample(
        complete, per_outcome=args.per_outcome, seed=args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.output_dir / "review_samples.jsonl"
    review_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
            for record in selected
        ),
        encoding="utf-8",
        newline="\n",
    )
    outcomes = Counter(
        classify_pair(record["baseline_metrics"]["qem"], record["trained_metrics"]["qem"])
        for record in complete
    )
    manifest = {
        "created_at": utc_now(),
        "contract_version": "gsm8k-target-capability-review-v1",
        "seed": args.seed,
        "sampling_rule": (
            "For each target outcome, select one stable-hash-ranked record from every "
            "non-empty outcome x marker-transition x length bucket, then fill the "
            "remaining quota by stable hash rank."
        ),
        "length_buckets_chars": {
            "shorter_by_50_or_more": "delta <= -50",
            "within_49_chars": "-49 <= delta <= 49",
            "longer_by_50_or_more": "delta >= 50",
        },
        "target_outcomes": ["improved", "regressed", "both_wrong"],
        "per_outcome": args.per_outcome,
        "population_count": len(complete),
        "population_by_outcome": dict(sorted(outcomes.items())),
        "reconstructed_identical_pair_count": sum(
            record["pair_source"] == "reconstructed_identical_from_s1"
            for record in complete
        ),
        "selected_count": len(selected),
        "review_samples": str(review_path.resolve()),
        "review_samples_sha256": sha256_file(review_path),
        "sources": {
            "changed_samples": str(args.changed_samples.resolve()),
            "changed_samples_sha256": sha256_file(args.changed_samples),
            "trained_details": str(args.trained_details.resolve()),
            "trained_details_sha256": sha256_file(args.trained_details),
        },
        **strata,
    }
    atomic_write_json(args.output_dir / "sampling_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
