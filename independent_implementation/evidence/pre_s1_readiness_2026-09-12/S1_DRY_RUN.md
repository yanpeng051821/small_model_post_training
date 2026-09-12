# Formal S1 CPU Dry-Run

## Purpose

This is a data-and-contract preflight for the formal 16K S1 run. It reads and
validates the exact frozen training JSONL, creates the deterministic shuffled
sample order, and records the identity that a GPU launch must preserve. It does
not load Qwen weights, allocate CUDA memory, build a TRL GPU configuration, or
perform an optimizer update.

## Result

The dry-run completed on 2026-09-12 with status `dry_run`. The machine-readable
record is `s1_dry_run_manifest.json` in this directory.

| Bound field | Value |
| --- | --- |
| Source Git commit | `e063e04ea5dcedaee4408cda994091d016a2a1c9` |
| TRL backend | `0.18.0` |
| Model and tokenizer | `Qwen/Qwen3-0.6B-Base` |
| Model/tokenizer revision | `311c62e88814bff7206909ccd330bab0a784743b` |
| Retained training records | `61,224` |
| Optimizer updates | `479` |
| Warmup updates | `15` |
| Training JSONL SHA-256 | `34f919f75ef49dcb45d2fa6bead80fa93667d04f709de48653a31f0fe2de3439` |
| Validation JSONL SHA-256 | `60bbe71d35d86504527ee30860ad54ff83745de1549196c27ba86ea3678f3ff9` |
| Frozen sample-order SHA-256 | `9af3e9338d993471846cc8de3bdca760eaf2f3242c5e536840dff8cd5346c2910` |

## Engineering Findings

1. The first CPU attempt used a persistent virtual environment installed from
   an older data-disk checkout. Its editable package import masked new source
   files. The launch rule is therefore to create a fresh environment from the
   exact checkout with `uv sync --frozen`; the runbook now makes this explicit.
2. The original `--dry-run` constructed TRL `SFTConfig` before checking the
   dry-run flag. TRL correctly rejects BF16 without a GPU, but that made a
   data-only preflight impossible. The runner now computes the frozen schedule
   without `SFTConfig`, writes the manifest, and only constructs GPU-specific
   arguments for a real run. A subprocess regression test exercises this using
   a CUDA/BF16 configuration on a CPU-only process.
3. The indexed dataset now retains each validated `sample_id` alongside its
   byte offset. Writing the shuffled order no longer reopens and decodes every
   JSON line after indexing.

## Boundary

This closes only the CPU-side launch-contract gate. The host was intentionally
CPU-only and reused existing dependencies with an explicit source path for this
data preflight. It does not prove the final server environment, FlashAttention,
CUDA allocator behavior, AdamW first-step memory, checkpoint export, or the
paired evaluation route. Those remain blocking GPU gates in `S1_PRELAUNCH_PLAN.md`.
