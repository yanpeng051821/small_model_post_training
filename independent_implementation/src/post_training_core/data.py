from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler


def collate_sft_batch(
    samples: list[dict[str, Any]],
    pad_token_id: int,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor | list[str]]:
    """Right-pad tokenized SFT samples into a batch of integer tensors."""
    if not samples:
        raise ValueError("samples must not be empty")

    input_list = []
    label_list = []
    attention_mask_list = []
    max_length = 0

    for index, sample in enumerate(samples):
        input_ids = sample["input_ids"]
        labels_len = len(sample["labels"])

        if len(input_ids) != labels_len:
            raise ValueError(
                f"input_ids and labels must have equal length for sample {index}"
            )

        max_length = max(max_length, len(input_ids))

    for sample in samples:
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        attention_mask = torch.ones(max_length, dtype=torch.long)

        padded = input_ids + [pad_token_id] * (max_length - len(input_ids))
        input_list.append(torch.tensor(padded, dtype=torch.long))

        labels = labels + [ignore_index] * (max_length - len(labels))
        label_list.append(torch.tensor(labels, dtype=torch.long))

        attention_mask[len(input_ids) :] = 0
        attention_mask_list.append(attention_mask)

    batch = {}
    batch["input_ids"] = torch.stack(input_list, dim=0)
    batch["labels"] = torch.stack(label_list, dim=0)
    batch["attention_mask"] = torch.stack(attention_mask_list, dim=0)
    sample_ids = [sample.get("sample_id") for sample in samples]
    if any(sample_id is not None for sample_id in sample_ids):
        if not all(isinstance(sample_id, str) and sample_id for sample_id in sample_ids):
            raise ValueError("sample_id must be present and non-empty for every sample")
        batch["sample_ids"] = sample_ids

    return batch


def build_assistant_only_labels(
    input_ids: list[int],
    assistant_mask: list[bool],
    ignore_index: int = -100,
) -> list[int]:
    """Mask every token that is not an assistant training target."""
    if len(input_ids) != len(assistant_mask):
        raise ValueError("input_ids and assistant_mask must have equal length")

    if not input_ids:
        raise ValueError("input_ids must not be empty")

    if not any(assistant_mask):
        raise ValueError("assistant_mask must have at least one True")

    labels = []
    for input_id, should_train in zip(input_ids, assistant_mask, strict=True):
        if should_train:
            labels.append(input_id)
        else:
            labels.append(ignore_index)

    return labels


def build_single_turn_sft_sample(
    tokenizer,
    user_message: dict[str, str],
    assistant_message: dict[str, str],
    assistant_end_token: str = "<|im_end|>",
    ignore_index: int = -100,
) -> dict[str, list[int]]:
    """Build assistant-only labels for one user-to-assistant conversation."""
    prompt_ids = tokenizer.apply_chat_template(
        [user_message],
        tokenize=True,
        add_generation_prompt=True,
    )

    full_ids = tokenizer.apply_chat_template(
        [user_message, assistant_message],
        tokenize=True,
        add_generation_prompt=False,
    )

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt token ids must be a prefix of full token ids")

    assistant_start = len(prompt_ids)

    assistant_end_token_id = tokenizer.convert_tokens_to_ids(assistant_end_token)

    if assistant_end_token_id is None:
        raise ValueError(f"unknown assistant end token: {assistant_end_token}")

    try:
        assistant_end = full_ids.index(
            assistant_end_token_id,
            assistant_start,
        )
    except ValueError as exc:
        raise ValueError(
            "assistant end token must appear after assistant start"
        ) from exc

    assistant_mask = [False] * len(full_ids)
    for index in range(assistant_start, assistant_end + 1):
        assistant_mask[index] = True

    labels = build_assistant_only_labels(
        full_ids,
        assistant_mask,
        ignore_index=ignore_index,
    )

    return {
        "input_ids": full_ids,
        "labels": labels,
    }


def _validate_tokenized_record(record: dict[str, Any], *, context: str) -> None:
    sample_id = record.get("sample_id")
    input_ids = record.get("input_ids")
    labels = record.get("labels")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{context} must contain a non-empty sample_id")
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError(f"{context} input_ids must be a non-empty list")
    if not isinstance(labels, list) or len(labels) != len(input_ids):
        raise ValueError(f"{context} labels must match input_ids length")
    if not all(isinstance(token, int) for token in input_ids + labels):
        raise ValueError(f"{context} token ids must be integers")
    if not any(label != -100 for label in labels[1:]):
        raise ValueError(f"{context} has no shifted supervision tokens")


class TokenizedSFTDataset(Dataset):
    """In-memory tokenized records for tiny tests and fixtures."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("tokenized SFT dataset must not be empty")

        sample_ids: set[str] = set()
        validated = []
        for index, record in enumerate(records):
            _validate_tokenized_record(record, context=f"record {index}")
            sample_id = record["sample_id"]
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            sample_ids.add(sample_id)
            validated.append(record)

        self._records = validated

    @classmethod
    def from_jsonl(cls, path: str | Path) -> TokenizedSFTDataset:
        records = []
        artifact_path = Path(path)
        with artifact_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(
                        f"blank line in tokenized artifact at line {line_number}"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"record at line {line_number} must be an object")
                records.append(record)
        return cls(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._records[index]


class IndexedTokenizedSFTDataset(Dataset):
    """Disk-backed JSONL dataset that keeps only byte offsets in memory."""

    def __init__(self, path: str | Path, *, limit: int | None = None) -> None:
        self.path = Path(path).resolve()
        if limit is not None and limit <= 0:
            raise ValueError("dataset limit must be positive when provided")
        self._offsets: list[int] = []
        self._sample_ids: list[str] = []
        self._stream = None
        sample_ids: set[str] = set()
        with self.path.open("rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    raise ValueError(
                        f"blank line in tokenized artifact at line {line_number}"
                    )
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(
                        f"invalid JSON at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"record at line {line_number} must be an object")
                _validate_tokenized_record(record, context=f"record {line_number}")
                sample_id = record["sample_id"]
                if sample_id in sample_ids:
                    raise ValueError(f"duplicate sample_id: {sample_id}")
                sample_ids.add(sample_id)
                self._offsets.append(offset)
                self._sample_ids.append(sample_id)
                if limit is not None and len(self._offsets) >= limit:
                    break
        if not self._offsets:
            raise ValueError("tokenized SFT dataset must not be empty")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._stream is None or self._stream.closed:
            self._stream = self.path.open("rb")
        self._stream.seek(self._offsets[index])
        return json.loads(self._stream.readline())

    def sample_id_at(self, index: int) -> str:
        """Return the indexed ID without reopening and decoding the JSONL record."""
        return self._sample_ids[index]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_stream"] = None
        return state

    def __del__(self) -> None:
        if self._stream is not None and not self._stream.closed:
            self._stream.close()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StatefulRandomSampler(Sampler[int]):
    """Deterministic random sampler whose exact position can be checkpointed."""

    def __init__(self, data_source: Dataset, seed: int) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self.committed_position = 0
        self._indices = self._make_indices()

    def _make_indices(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.data_source), generator=generator).tolist()

    def __iter__(self):
        while self.position < len(self._indices):
            index = self._indices[self.position]
            self.position += 1
            yield index

    def __len__(self) -> int:
        return len(self._indices) - self.position

    def advance_epoch(self) -> None:
        if self.committed_position != len(self._indices):
            raise RuntimeError("cannot advance epoch before every yielded sample is consumed")
        self.epoch += 1
        self.position = 0
        self.committed_position = 0
        self._indices = self._make_indices()

    def commit(self, count: int) -> None:
        """Record samples actually consumed, excluding DataLoader prefetch."""
        if count <= 0:
            raise ValueError("committed sample count must be positive")
        next_position = self.committed_position + count
        if next_position > self.position:
            raise RuntimeError("cannot commit samples that the sampler has not yielded")
        if next_position > len(self._indices):
            raise RuntimeError("committed sample position exceeds the dataset")
        self.committed_position = next_position

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "position": self.committed_position,
            "seed": self.seed,
            "dataset_size": len(self.data_source),
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        if state["seed"] != self.seed:
            raise ValueError("sampler seed does not match checkpoint")
        if state["dataset_size"] != len(self.data_source):
            raise ValueError("sampler dataset size does not match checkpoint")
        epoch = state["epoch"]
        position = state["position"]
        if epoch < 0 or not 0 <= position <= len(self.data_source):
            raise ValueError("invalid sampler position in checkpoint")
        self.epoch = epoch
        self.position = position
        self.committed_position = position
        self._indices = self._make_indices()
