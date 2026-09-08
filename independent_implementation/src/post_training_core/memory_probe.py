"""Longest-record selection for the paid-host training memory gate."""

from __future__ import annotations

import json
import re
from pathlib import Path

_TOKEN_COUNT_PATTERN = re.compile(rb'"token_count"\s*:\s*(\d+)')


def _load_extreme_tokenized_record(artifact: str | Path, *, longest: bool) -> dict:
    path = Path(artifact)
    selected_line = None
    selected_line_number = None
    selected_length = None
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            token_count_match = _TOKEN_COUNT_PATTERN.search(line)
            if token_count_match is not None:
                length = int(token_count_match.group(1))
            else:
                record = json.loads(line)
                input_ids = record.get("input_ids")
                labels = record.get("labels")
                sample_id = record.get("sample_id")
                if (
                    not isinstance(sample_id, str)
                    or not isinstance(input_ids, list)
                    or not isinstance(labels, list)
                    or len(input_ids) != len(labels)
                    or not input_ids
                ):
                    raise ValueError(f"invalid tokenized record at line {line_number}")
                length = len(input_ids)
            should_select = selected_length is None or (
                length > selected_length if longest else length < selected_length
            )
            if should_select:
                selected_line = line
                selected_line_number = line_number
                selected_length = length
    if selected_line is None:
        raise ValueError("tokenized artifact must contain at least one record")

    selected = json.loads(selected_line)
    input_ids = selected.get("input_ids")
    labels = selected.get("labels")
    sample_id = selected.get("sample_id")
    metadata = selected.get("metadata", {})
    if (
        not isinstance(sample_id, str)
        or not isinstance(input_ids, list)
        or not isinstance(labels, list)
        or len(input_ids) != len(labels)
        or not input_ids
        or (
            isinstance(metadata, dict)
            and "token_count" in metadata
            and metadata["token_count"] != len(input_ids)
        )
    ):
        raise ValueError(f"invalid tokenized record at line {selected_line_number}")
    return selected


def load_longest_tokenized_record(artifact: str | Path) -> dict:
    return _load_extreme_tokenized_record(artifact, longest=True)


def load_shortest_tokenized_record(artifact: str | Path) -> dict:
    return _load_extreme_tokenized_record(artifact, longest=False)
