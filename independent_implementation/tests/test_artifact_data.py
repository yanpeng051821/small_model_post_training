import json
import pickle

import pytest

from post_training_core.data import (
    IndexedTokenizedSFTDataset,
    StatefulRandomSampler,
    TokenizedSFTDataset,
)


def _records():
    return [
        {
            "sample_id": "sample-a",
            "input_ids": [1, 2, 3],
            "labels": [-100, 2, 3],
        },
        {
            "sample_id": "sample-b",
            "input_ids": [2, 3, 4, 5],
            "labels": [-100, -100, 4, 5],
        },
        {
            "sample_id": "sample-c",
            "input_ids": [3, 4],
            "labels": [-100, 4],
        },
    ]


def test_indexed_jsonl_dataset_pickle_separates_open_file_streams(tmp_path):
    path = tmp_path / "train.jsonl"
    records = _records()
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    dataset = IndexedTokenizedSFTDataset(path)

    # Open the source stream before serializing the Dataset.
    assert dataset[1] == records[1]
    original_stream = dataset._stream

    assert original_stream is not None
    assert not original_stream.closed

    restored = pickle.loads(pickle.dumps(dataset))

    # Pickling must leave the original stream untouched.
    assert dataset._stream is original_stream
    assert not dataset._stream.closed

    # The restored Dataset must lazily open an independent stream.
    assert restored._stream is None

    assert restored[2] == records[2]
    assert restored._stream is not None
    assert not restored._stream.closed
    assert restored._stream is not original_stream


def test_loads_frozen_tokenized_jsonl(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in _records()),
        encoding="utf-8",
    )

    dataset = TokenizedSFTDataset.from_jsonl(path)

    assert len(dataset) == 3
    assert dataset[1]["sample_id"] == "sample-b"


def test_indexed_jsonl_dataset_reads_lazily_and_survives_pickle(tmp_path):
    path = tmp_path / "train.jsonl"
    records = _records()
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    dataset = IndexedTokenizedSFTDataset(path)
    restored = pickle.loads(pickle.dumps(dataset))

    assert len(dataset) == 3
    assert dataset[1] == records[1]
    assert restored[0] == records[0]
    assert not hasattr(dataset, "_records")


def test_indexed_jsonl_dataset_can_freeze_a_prefix_for_smoke(tmp_path):
    path = tmp_path / "train.jsonl"
    records = _records()
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    dataset = IndexedTokenizedSFTDataset(path, limit=2)

    assert len(dataset) == 2
    assert [dataset[index]["sample_id"] for index in range(2)] == [
        "sample-a",
        "sample-b",
    ]


@pytest.mark.parametrize(
    "record",
    [
        {"sample_id": "a", "input_ids": [], "labels": []},
        {"sample_id": "a", "input_ids": [1, 2], "labels": [-100]},
        {"sample_id": "a", "input_ids": [1, 2], "labels": [-100, -100]},
    ],
)
def test_rejects_invalid_tokenized_records(record):
    with pytest.raises(ValueError):
        TokenizedSFTDataset([record])


def test_stateful_sampler_resumes_exact_remaining_order():
    dataset = TokenizedSFTDataset(_records())
    first = StatefulRandomSampler(dataset, seed=42)
    iterator = iter(first)
    consumed = [next(iterator), next(iterator)]
    first.commit(len(consumed))
    state = first.state_dict()
    remaining = list(iterator)

    resumed = StatefulRandomSampler(dataset, seed=42)
    resumed.load_state_dict(state)

    assert consumed + remaining == list(StatefulRandomSampler(dataset, seed=42))
    assert list(resumed) == remaining


def test_stateful_sampler_checkpoints_consumed_not_prefetched_position():
    dataset = TokenizedSFTDataset(_records())
    expected = list(StatefulRandomSampler(dataset, seed=42))
    sampler = StatefulRandomSampler(dataset, seed=42)
    iterator = iter(sampler)

    consumed = [next(iterator)]
    _prefetched = [next(iterator), next(iterator)]
    sampler.commit(len(consumed))

    resumed = StatefulRandomSampler(dataset, seed=42)
    resumed.load_state_dict(sampler.state_dict())

    assert consumed + list(resumed) == expected


def test_stateful_sampler_rejects_changed_dataset():
    original = TokenizedSFTDataset(_records())
    state = StatefulRandomSampler(original, seed=42).state_dict()
    changed = TokenizedSFTDataset(_records()[:2])

    with pytest.raises(ValueError, match="dataset size"):
        StatefulRandomSampler(changed, seed=42).load_state_dict(state)
