"""Dataset ordering and mask adapter for the pinned TRL reference run."""

from __future__ import annotations

from torch.utils.data import Dataset

from post_training_core.data import StatefulRandomSampler


class OrderedCompletionMaskDataset(Dataset):
    def __init__(self, source: Dataset, indices: list[int]) -> None:
        if not indices:
            raise ValueError("reference indices must not be empty")
        if min(indices) < 0 or max(indices) >= len(source):
            raise ValueError("reference index is outside the source dataset")
        self.source = source
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        record = self.source[self.indices[index]]
        return {
            "input_ids": record["input_ids"],
            "completion_mask": [
                int(label != -100) for label in record["labels"]
            ],
        }


def frozen_epoch_indices(source: Dataset, *, seed: int, limit: int) -> list[int]:
    if limit <= 0:
        raise ValueError("reference record limit must be positive")
    if limit > len(source):
        raise ValueError("reference record limit exceeds the dataset")
    sampler = StatefulRandomSampler(source, seed=seed)
    return list(iter(sampler))[:limit]
