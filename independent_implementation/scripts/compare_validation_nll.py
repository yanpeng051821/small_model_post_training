"""Compare B0 and S1 per-record validation NLL with a paired bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from post_training_core.data import sha256_file
from post_training_core.experiment import atomic_write_json, utc_now
from post_training_core.nll_comparison import compare_paired_nll, load_nll_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--trained-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    result = compare_paired_nll(
        load_nll_records(args.baseline_records),
        load_nll_records(args.trained_records),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result.update(
        {
            "baseline_records": str(args.baseline_records.resolve()),
            "baseline_records_sha256": sha256_file(args.baseline_records),
            "created_at": utc_now(),
            "trained_records": str(args.trained_records.resolve()),
            "trained_records_sha256": sha256_file(args.trained_records),
        }
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
