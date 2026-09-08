"""Load a checkpoint in a fresh process and verify deterministic generation health."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from post_training_core.experiment import atomic_write_json, utc_now
from post_training_core.generation import summarize_generation_health


def _load_prompts(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank prompt line at {line_number}")
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("id"), str)
                or not isinstance(record.get("prompt"), str)
                or not record["prompt"].strip()
            ):
                raise ValueError(f"invalid prompt record at line {line_number}")
            records.append(record)
    if not records:
        raise ValueError("prompt artifact must not be empty")
    return records


def _source_identity(source: str) -> str:
    path = Path(source)
    return str(path.resolve()) if path.exists() else source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--minimum-eos-rate", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    tokenizer_source = args.tokenizer or args.checkpoint
    tokenizer_kwargs = (
        {"revision": args.tokenizer_revision}
        if args.tokenizer_revision is not None
        else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    checkpoint_kwargs = (
        {"revision": args.checkpoint_revision}
        if args.checkpoint_revision is not None
        else {}
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        **checkpoint_kwargs,
    ).to("cuda")
    model.eval()
    model.config.use_cache = True
    eot_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(eot_token_id, int) or eot_token_id < 0:
        raise ValueError("tokenizer does not define a valid <|im_end|> token")

    records = []
    for prompt in _load_prompts(args.prompts):
        try:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=eot_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion_ids = output_ids[0, input_ids.shape[1] :].tolist()
            generated_eos = eot_token_id in completion_ids
            record = {
                **prompt,
                "completion_ids": completion_ids,
                "generated_eos": generated_eos,
                "generated_tokens": len(completion_ids),
                "output": tokenizer.decode(completion_ids, skip_special_tokens=False),
                "stop_reason": "eos" if generated_eos else "length",
                "error": None,
            }
        except Exception as error:  # Evidence must retain every failed prompt.
            record = {
                **prompt,
                "completion_ids": [],
                "generated_eos": False,
                "generated_tokens": 0,
                "output": None,
                "stop_reason": "runtime_error",
                "error": f"{type(error).__name__}: {error}",
            }
        records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "generations.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    summary = {
        **summarize_generation_health(
            records,
            minimum_eos_rate=args.minimum_eos_rate,
        ),
        "checkpoint": _source_identity(args.checkpoint),
        "created_at": utc_now(),
        "records": str(records_path.resolve()),
        "checkpoint_revision": args.checkpoint_revision,
        "tokenizer": _source_identity(tokenizer_source),
        "tokenizer_revision": args.tokenizer_revision,
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
