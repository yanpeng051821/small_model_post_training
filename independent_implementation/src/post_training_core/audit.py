"""Deterministic OpenR1-Math filtering and SFT artifact construction."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from post_training_core.data import (
    _validate_tokenized_record,
    build_single_turn_sft_sample,
    sha256_file,
)
from post_training_core.experiment import atomic_write_json, utc_now


@dataclass(frozen=True)
class AuditSettings:
    validation_size: int = 2_000
    max_length: int = 32_768
    split_salt: str = "independent-sft-v1"
    ngram_size: int = 8
    accepted_review_size: int = 20
    rejected_review_size_per_reason: int = 3

    def __post_init__(self) -> None:
        if self.validation_size <= 0:
            raise ValueError("validation_size must be positive")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2")
        if not self.split_salt:
            raise ValueError("split_salt must not be empty")
        if self.ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        if self.accepted_review_size <= 0:
            raise ValueError("accepted_review_size must be positive")
        if self.rejected_review_size_per_reason <= 0:
            raise ValueError("rejected_review_size_per_reason must be positive")


@dataclass(frozen=True)
class AuditResult:
    output_dir: Path
    train_count: int
    validation_count: int
    rejected_count: int
    manifest_path: Path


def _write_success_manifest(
    *,
    destination: Path,
    accepted_sequence_lengths: list[int],
    accepted_supervised_tokens: list[int],
    reason_counts: Counter[str],
    settings: AuditSettings,
    source_identity: Mapping[str, Any],
    train_count: int,
    validation_count: int,
    rejected_count: int,
    review_count: int,
) -> Path:
    train_path = destination / "train.jsonl"
    validation_path = destination / "validation.jsonl"
    rejected_path = destination / "rejected.jsonl"
    review_path = destination / "review_samples.jsonl"
    manifest_path = destination / "data_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "accepted_count": train_count + validation_count,
            "created_at": utc_now(),
            "reason_counts": dict(sorted(reason_counts.items())),
            "rejected_count": rejected_count,
            "length_statistics": {
                "sequence_tokens": _integer_distribution(
                    accepted_sequence_lengths
                ),
                "supervised_tokens": _integer_distribution(
                    accepted_supervised_tokens
                ),
            },
            "settings": {
                "max_length": settings.max_length,
                "ngram_size": settings.ngram_size,
                "accepted_review_size": settings.accepted_review_size,
                "rejected_review_size_per_reason": (
                    settings.rejected_review_size_per_reason
                ),
                "split_salt": settings.split_salt,
                "validation_size": settings.validation_size,
            },
            "source": dict(source_identity),
            "train": {"count": train_count, "sha256": sha256_file(train_path)},
            "validation": {
                "count": validation_count,
                "sha256": sha256_file(validation_path),
            },
            "rejected": {
                "count": rejected_count,
                "sha256": sha256_file(rejected_path),
            },
            "review": {
                "count": review_count,
                "sha256": sha256_file(review_path),
            },
        },
    )
    return manifest_path


def normalize_problem(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.casefold().split())


def problem_hash(problem: str) -> str:
    return hashlib.sha256(normalize_problem(problem).encode("utf-8")).hexdigest()


def sample_content_hash(problem: str, generation: str) -> str:
    payload = json.dumps(
        {"generation": generation, "problem": problem},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _integer_distribution(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty integer distribution")
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _word_ngrams(text: str, size: int) -> set[tuple[str, ...]]:
    # Match the pinned Open-R1 decontamination contract: normalize whitespace,
    # then keep punctuation attached while splitting on whitespace.
    words = normalize_problem(text).split()
    return {
        tuple(words[index : index + size]) for index in range(len(words) - size + 1)
    }


class ContaminationIndex:
    def __init__(
        self,
        evaluation_problems: Mapping[str, Iterable[str]],
        *,
        ngram_size: int,
    ) -> None:
        self.ngram_size = ngram_size
        self.exact: dict[str, set[str]] = defaultdict(set)
        self.ngrams: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for suite, problems in evaluation_problems.items():
            for problem in problems:
                normalized = normalize_problem(problem)
                if not normalized:
                    continue
                self.exact[normalized].add(suite)
                for ngram in _word_ngrams(normalized, ngram_size):
                    self.ngrams[ngram].add(suite)

    def matches(self, problem: str) -> list[str]:
        normalized = normalize_problem(problem)
        exact_suites = self.exact.get(normalized)
        if exact_suites:
            return [f"evaluation_exact_match:{suite}" for suite in sorted(exact_suites)]

        suites: set[str] = set()
        for ngram in _word_ngrams(normalized, self.ngram_size):
            suites.update(self.ngrams.get(ngram, ()))
        return [
            f"evaluation_{self.ngram_size}gram_overlap:{suite}"
            for suite in sorted(suites)
        ]


def _is_closed_reasoning(text: str) -> bool:
    starts = [match.start() for match in re.finditer(r"<think>", text)]
    ends = [match.start() for match in re.finditer(r"</think>", text)]
    return (
        bool(starts)
        and len(starts) == len(ends)
        and all(start < end for start, end in zip(starts, ends, strict=True))
    )


def _validate_row_shape(row: Mapping[str, Any]) -> list[str]:
    issues = []
    problem = row.get("problem")
    messages = row.get("messages")
    if not isinstance(problem, str) or not problem.strip():
        issues.append("empty_problem")
    if not isinstance(messages, list) or len(messages) != 2:
        issues.append("invalid_messages")
    elif (
        not isinstance(messages[0], dict)
        or not isinstance(messages[1], dict)
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
        or not isinstance(messages[1].get("content"), str)
        or not messages[1]["content"].strip()
    ):
        issues.append("invalid_messages")
    return issues


def _select_verified_generation(row: Mapping[str, Any]) -> tuple[int, str] | str:
    messages = row["messages"]
    assistant_content = messages[1]["content"]
    generations = row.get("generations")
    verified = row.get("correctness_math_verify")
    complete = row.get("is_reasoning_complete")
    if not all(isinstance(value, list) for value in (generations, verified, complete)):
        return "invalid_generation_fields"
    if not (len(generations) == len(verified) == len(complete)):
        return "generation_field_length_mismatch"

    matching = [
        index
        for index, generation in enumerate(generations)
        if generation == assistant_content
    ]
    if not matching:
        return "assistant_not_in_generations"
    generation_index = matching[0]
    if verified[generation_index] is not True:
        return "math_verify_failed"
    if complete[generation_index] is not True:
        return "incomplete_reasoning"
    generation = generations[generation_index]
    if not isinstance(generation, str) or not _is_closed_reasoning(generation):
        return "unclosed_think_tags"
    if "\\boxed{" not in generation:
        return "missing_boxed_answer"
    return generation_index, generation


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _materialize_sorted_artifacts(
    candidate_path: Path,
    accepted_locations: list[tuple[str, int]],
    *,
    validation_size: int,
    train_path: Path,
    validation_path: Path,
    max_chunk_bytes: int = 256 * 1024 * 1024,
) -> None:
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")
    train_temporary = train_path.with_suffix(f"{train_path.suffix}.tmp")
    validation_temporary = validation_path.with_suffix(
        f"{validation_path.suffix}.tmp"
    )
    offset_to_key = {offset: key for key, offset in accepted_locations}
    chunk_paths: list[Path] = []

    def write_chunk(records: list[tuple[str, bytes]]) -> None:
        records.sort(key=lambda item: item[0])
        chunk_path = candidate_path.with_name(
            f".sorted-chunk-{len(chunk_paths):05d}.tmp"
        )
        chunk_paths.append(chunk_path)
        with chunk_path.open("wb") as chunk_stream:
            for key, line in records:
                chunk_stream.write(key.encode("ascii") + b"\t" + line)
            chunk_stream.flush()
            os.fsync(chunk_stream.fileno())

    try:
        chunk: list[tuple[str, bytes]] = []
        chunk_bytes = 0
        with candidate_path.open("rb") as candidates:
            while True:
                offset = candidates.tell()
                line = candidates.readline()
                if not line:
                    break
                try:
                    key = offset_to_key.pop(offset)
                except KeyError as error:
                    raise RuntimeError(
                        f"candidate offset has no split key: {offset}"
                    ) from error
                chunk.append((key, line))
                chunk_bytes += len(line) + len(key) + 1
                if chunk_bytes >= max_chunk_bytes:
                    write_chunk(chunk)
                    chunk = []
                    chunk_bytes = 0
        if offset_to_key:
            raise RuntimeError("some accepted offsets were absent from candidate file")
        if chunk:
            write_chunk(chunk)

        chunk_streams = [path.open("rb") for path in chunk_paths]
        try:
            merge_heap: list[tuple[str, int, bytes]] = []
            for chunk_index, stream in enumerate(chunk_streams):
                line = stream.readline()
                if line:
                    key, payload = line.split(b"\t", maxsplit=1)
                    heapq.heappush(
                        merge_heap,
                        (key.decode("ascii"), chunk_index, payload),
                    )
            with (
                train_temporary.open("wb") as train_stream,
                validation_temporary.open("wb") as validation_stream,
            ):
                output_index = 0
                while merge_heap:
                    _key, chunk_index, payload = heapq.heappop(merge_heap)
                    destination = (
                        validation_stream
                        if output_index < validation_size
                        else train_stream
                    )
                    destination.write(payload)
                    output_index += 1
                    line = chunk_streams[chunk_index].readline()
                    if line:
                        key, next_payload = line.split(b"\t", maxsplit=1)
                        heapq.heappush(
                            merge_heap,
                            (key.decode("ascii"), chunk_index, next_payload),
                        )
                if output_index != len(accepted_locations):
                    raise RuntimeError("external merge lost accepted records")
                for stream in (train_stream, validation_stream):
                    stream.flush()
                    os.fsync(stream.fileno())
        finally:
            for stream in chunk_streams:
                stream.close()
        train_temporary.replace(train_path)
        validation_temporary.replace(validation_path)
    finally:
        for path in chunk_paths:
            path.unlink(missing_ok=True)


def _split_key(sample_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{sample_id}".encode()).hexdigest()


def _review_key(sample_id: str | None, row_index: int, salt: str) -> str:
    identity = f"{sample_id or 'missing'}:{row_index}"
    return hashlib.sha256(f"{salt}:review:{identity}".encode()).hexdigest()


def _retain_smallest(
    heap: list[tuple[int, dict[str, Any]]],
    *,
    key: str,
    record: dict[str, Any],
    limit: int,
) -> None:
    numeric_key = int(key, 16)
    item = (-numeric_key, record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif numeric_key < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _ordered_review_records(
    heap: list[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [record for _, record in sorted(heap, key=lambda item: -item[0])]


def audit_openr1_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    tokenizer,
    output_dir: str | Path,
    settings: AuditSettings,
    evaluation_problems: Mapping[str, Iterable[str]],
    source_identity: Mapping[str, Any],
) -> AuditResult:
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"audit output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    contamination = ContaminationIndex(
        evaluation_problems,
        ngram_size=settings.ngram_size,
    )
    accepted_ids: set[str] = set()
    accepted_locations: list[tuple[str, int]] = []
    rejected = []
    reason_counts: Counter[str] = Counter()
    accepted_review: list[tuple[int, dict[str, Any]]] = []
    rejected_review: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    accepted_sequence_lengths: list[int] = []
    accepted_supervised_tokens: list[int] = []
    candidate_path = destination / ".accepted-candidates.jsonl.tmp"
    with candidate_path.open("wb") as candidate_stream:
        for row_index, row in enumerate(rows):
            issues = _validate_row_shape(row)
            if issues:
                selected = None
            else:
                selected = _select_verified_generation(row)
                if isinstance(selected, str):
                    issues.append(selected)

            problem = row.get("problem")
            sample_id = problem_hash(problem) if isinstance(problem, str) else None
            if not issues and sample_id in accepted_ids:
                issues.append("duplicate_problem")
            if not issues:
                issues.extend(contamination.matches(problem))

            sample = None
            if not issues:
                generation_index, generation = selected
                try:
                    sample = build_single_turn_sft_sample(
                        tokenizer,
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": generation},
                    )
                except ValueError as error:
                    issues.append(f"tokenization_error:{error}")
                else:
                    if len(sample["input_ids"]) > settings.max_length:
                        issues.append("over_max_length")

            if issues:
                reason_counts.update(issues)
                rejection = {
                    "issues": issues,
                    "problem_hash": sample_id,
                    "row_index": row_index,
                    "source": row.get("source"),
                    "token_count": len(sample["input_ids"]) if sample else None,
                    "uuid": row.get("uuid"),
                }
                rejected.append(rejection)
                review_record = {
                    **rejection,
                    "problem": problem,
                    "status": "rejected",
                }
                if isinstance(selected, tuple):
                    review_record["generation"] = selected[1]
                else:
                    messages = row.get("messages")
                    if (
                        isinstance(messages, list)
                        and len(messages) > 1
                        and isinstance(messages[1], Mapping)
                        and isinstance(messages[1].get("content"), str)
                    ):
                        review_record["generation"] = messages[1]["content"]
                review_key = _review_key(sample_id, row_index, settings.split_salt)
                for reason in issues:
                    _retain_smallest(
                        rejected_review[reason],
                        key=review_key,
                        record={**review_record, "review_reason": reason},
                        limit=settings.rejected_review_size_per_reason,
                    )
                continue

            record = {
                "input_ids": sample["input_ids"],
                "labels": sample["labels"],
                "metadata": {
                    "content_sha256": sample_content_hash(problem, generation),
                    "correctness_math_verify": row["correctness_math_verify"][
                        generation_index
                    ],
                    "generation_index": generation_index,
                    "is_reasoning_complete": row["is_reasoning_complete"][
                        generation_index
                    ],
                    "row_index": row_index,
                    "source": row.get("source"),
                    "supervised_tokens": sum(
                        label != -100 for label in sample["labels"][1:]
                    ),
                    "token_count": len(sample["input_ids"]),
                    "uuid": row.get("uuid"),
                },
                "sample_id": sample_id,
            }
            offset = candidate_stream.tell()
            candidate_stream.write(
                (json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            )
            accepted_ids.add(sample_id)
            accepted_sequence_lengths.append(len(sample["input_ids"]))
            accepted_supervised_tokens.append(
                sum(label != -100 for label in sample["labels"][1:])
            )
            accepted_locations.append(
                (_split_key(sample_id, settings.split_salt), offset)
            )
            _retain_smallest(
                accepted_review,
                key=_review_key(sample_id, row_index, settings.split_salt),
                record={
                    "generation": generation,
                    "problem": problem,
                    "row_index": row_index,
                    "sample_id": sample_id,
                    "source": row.get("source"),
                    "status": "accepted",
                    "uuid": row.get("uuid"),
                },
                limit=settings.accepted_review_size,
            )
        candidate_stream.flush()
        os.fsync(candidate_stream.fileno())

    accepted_count = len(accepted_locations)
    rejected_path = destination / "rejected.jsonl"
    _write_jsonl(rejected_path, rejected)
    if accepted_count <= settings.validation_size:
        atomic_write_json(
            destination / "data_manifest.json",
            {
                "accepted_count": accepted_count,
                "created_at": utc_now(),
                "failure": "accepted sample count must be larger than validation_size",
                "reason_counts": dict(sorted(reason_counts.items())),
                "rejected_count": len(rejected),
                "source": dict(source_identity),
                "status": "failed",
            },
        )
        candidate_path.unlink()
        raise ValueError("accepted sample count must be larger than validation_size")

    train_path = destination / "train.jsonl"
    validation_path = destination / "validation.jsonl"
    _materialize_sorted_artifacts(
        candidate_path,
        accepted_locations,
        validation_size=settings.validation_size,
        train_path=train_path,
        validation_path=validation_path,
    )
    candidate_path.unlink()
    validation_count = settings.validation_size
    train_count = accepted_count - validation_count
    review_records = _ordered_review_records(accepted_review)
    for reason in sorted(rejected_review):
        review_records.extend(_ordered_review_records(rejected_review[reason]))
    review_path = destination / "review_samples.jsonl"
    _write_jsonl(review_path, review_records)
    manifest_path = _write_success_manifest(
        destination=destination,
        accepted_sequence_lengths=accepted_sequence_lengths,
        accepted_supervised_tokens=accepted_supervised_tokens,
        reason_counts=reason_counts,
        settings=settings,
        source_identity=source_identity,
        train_count=train_count,
        validation_count=validation_count,
        rejected_count=len(rejected),
        review_count=len(review_records),
    )
    return AuditResult(
        output_dir=destination,
        train_count=train_count,
        validation_count=validation_count,
        rejected_count=len(rejected),
        manifest_path=manifest_path,
    )


def finalize_completed_audit_scan(
    rows,
    *,
    output_dir: str | Path,
    settings: AuditSettings,
    source_identity: Mapping[str, Any],
) -> AuditResult:
    """Recover final artifacts after scanning finished but materialization stopped."""
    destination = Path(output_dir)
    candidate_path = destination / ".accepted-candidates.jsonl.tmp"
    rejected_path = destination / "rejected.jsonl"
    manifest_path = destination / "data_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"audit manifest already exists: {manifest_path}")
    for required in (candidate_path, rejected_path):
        if not required.is_file() or required.stat().st_size == 0:
            raise ValueError(f"completed scan artifact is missing or empty: {required}")
    for final_name in ("train.jsonl", "validation.jsonl", "review_samples.jsonl"):
        if (destination / final_name).exists():
            raise FileExistsError(f"final audit artifact already exists: {final_name}")
    for temporary in (
        destination / "train.jsonl.tmp",
        destination / "validation.jsonl.tmp",
        *destination.glob(".sorted-chunk-*.tmp"),
    ):
        temporary.unlink(missing_ok=True)

    accepted_locations = []
    accepted_sequence_lengths = []
    accepted_supervised_tokens = []
    accepted_review: list[tuple[int, dict[str, Any]]] = []
    accepted_ids: set[str] = set()
    with candidate_path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            record = json.loads(line)
            _validate_tokenized_record(record, context=f"candidate offset {offset}")
            sample_id = record["sample_id"]
            if sample_id in accepted_ids:
                raise ValueError(f"duplicate candidate sample_id: {sample_id}")
            accepted_ids.add(sample_id)
            metadata = record.get("metadata", {})
            row_index = metadata.get("row_index")
            token_count = metadata.get("token_count")
            supervised_tokens = metadata.get("supervised_tokens")
            if not all(
                isinstance(value, int) and value >= 0
                for value in (row_index, token_count, supervised_tokens)
            ):
                raise ValueError(f"candidate metadata is incomplete at offset {offset}")
            if token_count != len(record["input_ids"]) or supervised_tokens <= 0:
                raise ValueError(f"candidate token metadata is invalid at offset {offset}")
            accepted_sequence_lengths.append(token_count)
            accepted_supervised_tokens.append(supervised_tokens)
            accepted_locations.append(
                (_split_key(sample_id, settings.split_salt), offset)
            )
            _retain_smallest(
                accepted_review,
                key=_review_key(sample_id, row_index, settings.split_salt),
                record={
                    "generation_index": metadata.get("generation_index"),
                    "row_index": row_index,
                    "sample_id": sample_id,
                },
                limit=settings.accepted_review_size,
            )

    reason_counts: Counter[str] = Counter()
    rejected_count = 0
    rejected_review: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    with rejected_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            rejection = json.loads(line)
            issues = rejection.get("issues")
            row_index = rejection.get("row_index")
            if not isinstance(issues, list) or not issues or not isinstance(row_index, int):
                raise ValueError(f"invalid rejected record at line {line_number}")
            rejected_count += 1
            reason_counts.update(issues)
            review_key = _review_key(
                rejection.get("problem_hash"),
                row_index,
                settings.split_salt,
            )
            for reason in issues:
                _retain_smallest(
                    rejected_review[reason],
                    key=review_key,
                    record={**rejection, "review_reason": reason},
                    limit=settings.rejected_review_size_per_reason,
                )

    accepted_count = len(accepted_locations)
    if accepted_count <= settings.validation_size:
        raise ValueError("accepted sample count must be larger than validation_size")

    review_records = []
    for marker in _ordered_review_records(accepted_review):
        row = rows[marker["row_index"]]
        generation_index = marker["generation_index"]
        generation = row["generations"][generation_index]
        review_records.append(
            {
                "generation": generation,
                "problem": row["problem"],
                "row_index": marker["row_index"],
                "sample_id": marker["sample_id"],
                "source": row.get("source"),
                "status": "accepted",
                "uuid": row.get("uuid"),
            }
        )
    for reason in sorted(rejected_review):
        for rejection in _ordered_review_records(rejected_review[reason]):
            row = rows[rejection["row_index"]]
            record = {
                **rejection,
                "problem": row.get("problem"),
                "status": "rejected",
            }
            messages = row.get("messages")
            if (
                isinstance(messages, list)
                and len(messages) > 1
                and isinstance(messages[1], Mapping)
                and isinstance(messages[1].get("content"), str)
            ):
                record["generation"] = messages[1]["content"]
            review_records.append(record)

    train_path = destination / "train.jsonl"
    validation_path = destination / "validation.jsonl"
    _materialize_sorted_artifacts(
        candidate_path,
        accepted_locations,
        validation_size=settings.validation_size,
        train_path=train_path,
        validation_path=validation_path,
    )
    review_path = destination / "review_samples.jsonl"
    _write_jsonl(review_path, review_records)
    train_count = accepted_count - settings.validation_size
    manifest_path = _write_success_manifest(
        destination=destination,
        accepted_sequence_lengths=accepted_sequence_lengths,
        accepted_supervised_tokens=accepted_supervised_tokens,
        reason_counts=reason_counts,
        settings=settings,
        source_identity=source_identity,
        train_count=train_count,
        validation_count=settings.validation_size,
        rejected_count=rejected_count,
        review_count=len(review_records),
    )
    candidate_path.unlink()
    return AuditResult(
        output_dir=destination,
        train_count=train_count,
        validation_count=settings.validation_size,
        rejected_count=rejected_count,
        manifest_path=manifest_path,
    )
