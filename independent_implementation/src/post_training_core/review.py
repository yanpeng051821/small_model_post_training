"""Auditable human review workflow for deterministic data-audit samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from post_training_core.data import sha256_file
from post_training_core.experiment import atomic_write_json, utc_now


@dataclass(frozen=True)
class ReviewBundle:
    records: list[dict[str, Any]]
    sha256: str


def _record_sha256(record: dict[str, Any]) -> str:
    encoded = json.dumps(
        record,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_review_bundle(
    review_path: str | Path,
    data_manifest_path: str | Path,
) -> ReviewBundle:
    review_path = Path(review_path)
    data_manifest_path = Path(data_manifest_path)
    with data_manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected = manifest.get("review")
    if not isinstance(expected, dict):
        raise ValueError("data manifest has no review artifact identity")

    actual_sha256 = sha256_file(review_path)
    if actual_sha256 != expected.get("sha256"):
        raise ValueError("review artifact hash does not match data manifest")

    records = []
    with review_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank review record at line {line_number}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"review record {line_number} must be an object")
            status = record.get("status")
            if status not in {"accepted", "rejected"}:
                raise ValueError(f"invalid review status at line {line_number}")
            if not isinstance(record.get("problem"), str) or not isinstance(
                record.get("generation"), str
            ):
                raise ValueError(f"review record {line_number} has no text pair")
            if status == "rejected":
                issues = record.get("issues")
                if (
                    not isinstance(issues, list)
                    or record.get("review_reason") not in issues
                ):
                    raise ValueError(
                        "rejected review reason is inconsistent at line "
                        f"{line_number}"
                    )
            records.append(record)
    if len(records) != expected.get("count"):
        raise ValueError("review artifact count does not match data manifest")
    return ReviewBundle(records=records, sha256=actual_sha256)


def _allowed_verdicts(status: str) -> set[str]:
    return {"keep", "flag"} if status == "accepted" else {"agree", "disagree"}


def load_review_decisions(
    path: str | Path,
    *,
    bundle: ReviewBundle,
    reviewer: str,
) -> dict[int, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    decisions = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            decision = json.loads(line)
            index = decision.get("review_index")
            if not isinstance(index, int) or not 0 <= index < len(bundle.records):
                raise ValueError(f"invalid decision index at line {line_number}")
            if index in decisions:
                raise ValueError(f"duplicate decision index: {index}")
            if decision.get("review_artifact_sha256") != bundle.sha256:
                raise ValueError("existing decisions belong to another review artifact")
            if decision.get("reviewer") != reviewer:
                raise ValueError("existing decisions belong to another reviewer")
            record = bundle.records[index]
            if decision.get("record_sha256") != _record_sha256(record):
                raise ValueError(f"review record changed for decision index {index}")
            if decision.get("status") != record["status"]:
                raise ValueError(f"status mismatch for decision index {index}")
            if decision.get("verdict") not in _allowed_verdicts(record["status"]):
                raise ValueError(f"invalid verdict for decision index {index}")
            decisions[index] = decision
    return decisions


def _write_decisions(
    path: Path,
    decisions: dict[int, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for index in sorted(decisions):
            stream.write(
                json.dumps(decisions[index], ensure_ascii=True, sort_keys=True)
            )
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _summary_path(decisions_path: Path) -> Path:
    return decisions_path.with_suffix(".summary.json")


def write_review_state(
    decisions_path: str | Path,
    *,
    bundle: ReviewBundle,
    reviewer: str,
    decisions: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    decisions_path = Path(decisions_path)
    _write_decisions(decisions_path, decisions)
    verdict_counts = Counter(
        decision["verdict"] for decision in decisions.values()
    )
    complete = len(decisions) == len(bundle.records)
    unresolved = verdict_counts["flag"] + verdict_counts["disagree"]
    if not complete:
        status = "incomplete"
    elif unresolved:
        status = "needs_resolution"
    else:
        status = "passed"
    summary = {
        "completed_count": len(decisions),
        "created_at": utc_now(),
        "decision_artifact": str(decisions_path.resolve()),
        "decision_artifact_sha256": sha256_file(decisions_path),
        "expected_count": len(bundle.records),
        "passed": status == "passed",
        "review_artifact_sha256": bundle.sha256,
        "reviewer": reviewer,
        "status": status,
        "unresolved_count": unresolved,
        "verdict_counts": dict(sorted(verdict_counts.items())),
    }
    atomic_write_json(_summary_path(decisions_path), summary)
    return summary


def _show_record(
    index: int,
    total: int,
    record: dict[str, Any],
    output_fn: Callable[[str], None],
) -> None:
    output_fn("\n" + "=" * 88)
    output_fn(f"Review {index + 1}/{total}")
    output_fn(
        " | ".join(
            (
                f"status={record['status']}",
                f"reason={record.get('review_reason')}",
                f"row_index={record.get('row_index')}",
                f"source={record.get('source')}",
            )
        )
    )
    output_fn("\n[PROBLEM]\n" + record["problem"])
    output_fn("\n[GENERATION]\n" + record["generation"])


def run_interactive_review(
    *,
    bundle: ReviewBundle,
    decisions_path: str | Path,
    reviewer: str,
    redo_indices: Iterable[int] = (),
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    if not reviewer.strip():
        raise ValueError("reviewer must not be empty")
    decisions_path = Path(decisions_path)
    decisions = load_review_decisions(
        decisions_path,
        bundle=bundle,
        reviewer=reviewer,
    )
    for display_index in redo_indices:
        if not 1 <= display_index <= len(bundle.records):
            raise ValueError(f"redo index is out of range: {display_index}")
        decisions.pop(display_index - 1, None)
    write_review_state(
        decisions_path,
        bundle=bundle,
        reviewer=reviewer,
        decisions=decisions,
    )

    for index, record in enumerate(bundle.records):
        if index in decisions:
            continue
        _show_record(index, len(bundle.records), record, output_fn)
        if record["status"] == "accepted":
            choices = {"k": "keep", "f": "flag"}
            prompt = "[k] keep  [f] flag  [s] skip  [q] quit: "
        else:
            choices = {"a": "agree", "d": "disagree"}
            prompt = "[a] agree  [d] disagree  [s] skip  [q] quit: "
        while True:
            try:
                choice = input_fn(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                output_fn("\nReview paused; completed decisions remain saved.")
                choice = "q"
            if choice in choices or choice in {"s", "q"}:
                break
            output_fn("Invalid choice.")
        if choice == "q":
            break
        if choice == "s":
            continue
        try:
            comment = input_fn("Comment (optional): ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nReview paused; the current item was not recorded.")
            break
        decisions[index] = {
            "comment": comment,
            "record_sha256": _record_sha256(record),
            "review_artifact_sha256": bundle.sha256,
            "review_index": index,
            "review_reason": record.get("review_reason"),
            "reviewed_at": utc_now(),
            "reviewer": reviewer,
            "row_index": record.get("row_index"),
            "status": record["status"],
            "verdict": choices[choice],
        }
        write_review_state(
            decisions_path,
            bundle=bundle,
            reviewer=reviewer,
            decisions=decisions,
        )

    summary = write_review_state(
        decisions_path,
        bundle=bundle,
        reviewer=reviewer,
        decisions=decisions,
    )
    output_fn(json.dumps(summary, indent=2, sort_keys=True))
    if summary["passed"]:
        return 0
    return 2 if summary["status"] == "needs_resolution" else 1


def entrypoint() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-samples", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--redo",
        action="append",
        default=[],
        type=int,
        help="Revisit a displayed one-based review number; may be repeated.",
    )
    args = parser.parse_args()
    bundle = load_review_bundle(args.review_samples, args.data_manifest)
    return run_interactive_review(
        bundle=bundle,
        decisions_path=args.output,
        reviewer=args.reviewer,
        redo_indices=args.redo,
    )
