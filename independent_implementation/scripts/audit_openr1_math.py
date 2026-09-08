"""Materialize the frozen Gate 0B SFT train and validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform

import datasets
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

from post_training_core.audit import (
    AuditSettings,
    audit_openr1_rows,
    finalize_completed_audit_scan,
)

OPENR1_DATASET = "open-r1/OpenR1-Math-220k"
OPENR1_REVISION = "e4e141ec9dea9f8326f4d347be56105859b2bd68"
QWEN_MODEL = "Qwen/Qwen3-0.6B-Base"
QWEN_REVISION = "311c62e88814bff7206909ccd330bab0a784743b"

EVALUATION_SOURCES = {
    "math500": {
        "path": "HuggingFaceH4/MATH-500",
        "name": None,
        "split": "test",
        "field": "problem",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    },
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
        "field": "question",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
    },
    "mmlu": {
        "path": "cais/mmlu",
        "name": "all",
        "split": "test",
        "field": "question",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
    },
    "arc_challenge": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "test",
        "field": "question",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
    },
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "name": None,
        "split": "validation",
        "field": "ctx",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--max-length", type=int, default=32_768)
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Local pipeline check only; never use for a frozen artifact.",
    )
    parser.add_argument(
        "--evaluation-suite",
        action="append",
        choices=sorted(EVALUATION_SOURCES),
        dest="evaluation_suites",
        help="Limit eval sources for local smoke; default is every frozen suite.",
    )
    parser.add_argument(
        "--finalize-completed-scan",
        action="store_true",
        help=(
            "Recover a completed scan from .accepted-candidates.jsonl.tmp and "
            "rejected.jsonl without repeating tokenization."
        ),
    )
    return parser


def _load_evaluation_problems(selected_suites):
    problems = {}
    identities = {}
    for suite in selected_suites:
        source = EVALUATION_SOURCES[suite]
        dataset = load_dataset(
            source["path"],
            source["name"],
            split=source["split"],
            revision=source["revision"],
        )
        problems[suite] = dataset[source["field"]]
        identities[suite] = {
            **source,
            "fingerprint": dataset._fingerprint,
            "rows": len(dataset),
        }
    return problems, identities


def main() -> int:
    args = build_parser().parse_args()
    if args.limit_rows is not None and args.limit_rows <= 0:
        raise ValueError("limit_rows must be positive")
    selected_suites = args.evaluation_suites or list(EVALUATION_SOURCES)

    source = load_dataset(
        OPENR1_DATASET,
        "default",
        split="train",
        revision=OPENR1_REVISION,
        streaming=args.limit_rows is not None,
    )
    if args.limit_rows is not None:
        rows = list(source.take(args.limit_rows))
    else:
        rows = source
    evaluation_problems, evaluation_identities = _load_evaluation_problems(
        selected_suites
    )
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL,
        revision=QWEN_REVISION,
    )
    chat_template = tokenizer.chat_template or ""

    settings = AuditSettings(
        validation_size=args.validation_size,
        max_length=args.max_length,
    )
    source_identity = {
        "dataset": OPENR1_DATASET,
        "fingerprint": getattr(source, "_fingerprint", None),
        "frozen_candidate": (
            args.limit_rows is None
            and set(selected_suites) == set(EVALUATION_SOURCES)
        ),
        "limited_rows": args.limit_rows,
        "revision": OPENR1_REVISION,
        "rows": len(rows),
        "tokenizer": QWEN_MODEL,
        "tokenizer_revision": QWEN_REVISION,
        "tokenizer_chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "tokenizer_special_tokens": tokenizer.special_tokens_map,
        "evaluation_sources": evaluation_identities,
        "environment": {
            "datasets": datasets.__version__,
            "python": platform.python_version(),
            "transformers": transformers.__version__,
        },
    }
    if args.finalize_completed_scan:
        if args.limit_rows is not None:
            raise ValueError(
                "--finalize-completed-scan cannot be combined with --limit-rows"
            )
        result = finalize_completed_audit_scan(
            rows,
            output_dir=args.output_dir,
            settings=settings,
            source_identity=source_identity,
        )
    else:
        result = audit_openr1_rows(
            rows,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            settings=settings,
            evaluation_problems=evaluation_problems,
            source_identity=source_identity,
        )
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path.resolve()),
                "rejected": result.rejected_count,
                "train": result.train_count,
                "validation": result.validation_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
