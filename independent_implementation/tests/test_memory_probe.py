import json

import pytest

from post_training_core.memory_probe import (
    load_longest_tokenized_record,
    load_shortest_tokenized_record,
)


def test_loads_first_longest_valid_record(tmp_path):
    artifact = tmp_path / "train.jsonl"
    records = [
        {"sample_id": "short", "input_ids": [1, 2], "labels": [-100, 2]},
        {
            "sample_id": "first-longest",
            "input_ids": [1, 2, 3],
            "labels": [-100, 2, 3],
        },
        {
            "sample_id": "second-longest",
            "input_ids": [3, 4, 5],
            "labels": [-100, 4, 5],
        },
    ]
    artifact.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    assert load_longest_tokenized_record(artifact)["sample_id"] == "first-longest"
    assert load_shortest_tokenized_record(artifact)["sample_id"] == "short"


def test_rejects_malformed_record_instead_of_skipping_it(tmp_path):
    artifact = tmp_path / "train.jsonl"
    artifact.write_text(
        json.dumps({"sample_id": "bad", "input_ids": [1], "labels": []}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1"):
        load_longest_tokenized_record(artifact)


def test_rejects_selected_record_with_incorrect_token_count_metadata(tmp_path):
    artifact = tmp_path / "train.jsonl"
    artifact.write_text(
        json.dumps(
            {
                "sample_id": "bad",
                "input_ids": [1],
                "labels": [1],
                "metadata": {"token_count": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1"):
        load_longest_tokenized_record(artifact)
