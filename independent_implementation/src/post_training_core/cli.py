"""Command-line entry point for the independent SFT runner."""

from __future__ import annotations

import argparse
import sys

from post_training_core.config import ExperimentConfig
from post_training_core.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sft-train",
        description="Run a controlled single-device SFT experiment.",
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        default=None,
        help="Intentional preflight stop; does not change the frozen contract.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    run_dir = run_experiment(
        config,
        stop_after_steps=args.stop_after_steps,
    )
    print(run_dir)
    return 0


def entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
