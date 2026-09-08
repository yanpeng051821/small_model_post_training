"""Create a resume config whose only semantic addition is the checkpoint path."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml

from post_training_core.config import ExperimentConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {args.checkpoint}")
    config = ExperimentConfig.from_yaml(args.base_config)
    resumed = replace(config, resume_from_checkpoint=args.checkpoint.resolve())
    if resumed.semantic_hash() != config.semantic_hash():
        raise RuntimeError("resume config changed the frozen semantic contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(resumed.to_dict(), sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
