"""Evaluate frozen assistant-only validation NLL without updating parameters."""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from post_training_core.config import ExperimentConfig
from post_training_core.data import (
    IndexedTokenizedSFTDataset,
    collate_sft_batch,
    sha256_file,
)
from post_training_core.engine import resolve_device
from post_training_core.experiment import (
    atomic_write_json,
    set_reproducible_seed,
    utc_now,
)
from post_training_core.nll_evaluation import evaluate_sft_nll_records


def _load_model_and_tokenizer(
    source: str,
    revision: str | None,
    config: ExperimentConfig,
):
    common = {"revision": revision} if revision is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(source, **common)
    if tokenizer.pad_token_id != config.pad_token_id:
        raise ValueError("evaluation tokenizer pad token does not match the contract")
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=(
            torch.bfloat16
            if config.parameter_dtype == "bfloat16"
            else torch.float32
        ),
        attn_implementation=config.attn_implementation,
        **common,
    )
    model.config.use_cache = False
    return model, tokenizer


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(config.seed)
    device = resolve_device(config.device)
    model, tokenizer = _load_model_and_tokenizer(
        args.model,
        args.model_revision,
        config,
    )
    model.to(device)
    dataset = IndexedTokenizedSFTDataset(
        config.validation_artifact,
        limit=config.validation_sample_limit,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(
            collate_sft_batch,
            pad_token_id=config.pad_token_id,
        ),
        num_workers=config.dataloader_num_workers,
        pin_memory=False,
    )
    aggregate, records = evaluate_sft_nll_records(
        model,
        dataloader,
        device=device,
        dtype=config.dtype,
    )
    records_path = args.output_dir / "records.jsonl"
    _atomic_write_jsonl(records_path, records)
    summary = {
        **aggregate,
        "config": str(args.config.resolve()),
        "config_hash": config.semantic_hash(),
        "created_at": utc_now(),
        "model": args.model,
        "model_revision": args.model_revision,
        "records": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "validation_artifact": str(config.validation_artifact.resolve()),
        "validation_artifact_sha256": sha256_file(config.validation_artifact),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
