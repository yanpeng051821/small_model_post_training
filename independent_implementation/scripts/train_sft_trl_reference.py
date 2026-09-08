"""Run the pinned TRL 0.18 reference over the independent pilot sample order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from post_training_core.config import ExperimentConfig
from post_training_core.data import IndexedTokenizedSFTDataset, sha256_file
from post_training_core.experiment import (
    atomic_write_json,
    collect_environment,
    set_reproducible_seed,
    utc_now,
)
from post_training_core.trl_reference import (
    OrderedCompletionMaskDataset,
    frozen_epoch_indices,
)


class SequentialSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(train_dataset or self.train_dataset)


def _sample_order_hash(sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"reference output is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig.from_yaml(args.config)
    required_records = (
        args.max_steps
        * config.per_device_train_batch_size
        * config.gradient_accumulation_steps
    )
    source = IndexedTokenizedSFTDataset(config.train_artifact)
    indices = frozen_epoch_indices(
        source,
        seed=config.seed,
        limit=required_records,
    )
    sample_ids = [source[index]["sample_id"] for index in indices]
    order_path = args.output_dir / "sample_order.jsonl"
    order_path.write_text(
        "".join(
            json.dumps({"position": position, "sample_id": sample_id}) + "\n"
            for position, sample_id in enumerate(sample_ids)
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "config": str(args.config.resolve()),
        "config_hash": config.semantic_hash(),
        "created_at": utc_now(),
        "dry_run": args.dry_run,
        "max_steps": args.max_steps,
        "required_records": required_records,
        "sample_order": str(order_path.resolve()),
        "sample_order_sha256": _sample_order_hash(sample_ids),
        "train_artifact_sha256": sha256_file(config.train_artifact),
        "trl_version": "0.18.0",
    }
    atomic_write_json(args.output_dir / "run_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    set_reproducible_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
    )
    if tokenizer.pad_token_id != config.pad_token_id:
        raise ValueError("TRL reference tokenizer pad token does not match the contract")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
        torch_dtype=torch.float32,
        attn_implementation=config.attn_implementation,
        use_cache=False,
    )
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    dataset = OrderedCompletionMaskDataset(source, indices)
    training_args = SFTConfig(
        output_dir=str(args.output_dir / "trainer"),
        max_steps=args.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.scheduler_type,
        lr_scheduler_kwargs={"min_lr_rate": config.min_lr_ratio},
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        bf16=config.dtype == "bfloat16",
        fp16=False,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=config.dataloader_num_workers,
        completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        disable_tqdm=False,
    )
    trainer = SequentialSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            pad_token_id=config.pad_token_id,
            completion_only_loss=True,
        ),
    )
    train_result = trainer.train()
    trainer.model.config.use_cache = True
    trainer.save_model(args.output_dir / "final_model")
    trainer.save_state()
    completed = {
        **manifest,
        "completed_at": utc_now(),
        "environment": collect_environment(Path(__file__).resolve().parents[1]),
        "metrics": train_result.metrics,
        "status": "completed",
    }
    atomic_write_json(args.output_dir / "run_manifest.json", completed)
    print(json.dumps(completed, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
