# Gate 0B: Formal S1 Pre-Launch Plan

Status: `NO-GO` until every required item below is closed with an artifact.

This document reconciles the current state before the formal one-epoch S1
launch. It deliberately separates an engineering pilot from the formal
experiment: the 100-step checkpoint is evidence about training and recovery,
not an S1 candidate checkpoint.

## What Is Already Reusable

| Area | Evidence | Trust | Use in S1 preparation |
| --- | --- | --- | --- |
| Frozen data | `evidence/pre_s1_readiness_2026-09-12/data_manifest.server.json` and `DATA_INTEGRITY.md` | trusted | Bind S1 to the verified 16K artifact hashes |
| B0 | `evidence/gate0b-16k-b0-*.json` | trusted | Reuse as the paired pre-training baseline |
| H1 calculation tests | current CPU test suite | trusted | Do not rerun expensive model work to prove token-loss semantics |
| First-batch shadow | `evidence/gate0b-16k-first-batch-shadow.json` | trusted | Reuse as the TRL/reference numerical alignment anchor |
| 20-step smoke | `runs/gate0b-16k-pilot20-sdpa` and generation diagnosis evidence | usable with verification | Confirms a real A100 path, but does not replace the final source-tree preflight |
| 100-step TRL resume pilot | `runs/gate0b-16k-pilot100` and server evidence | usable with verification | Confirms native resume and lower held-out NLL; never resume it as S1 |

The B0 metrics to preserve are:

| Metric | B0 value |
| --- | ---: |
| Validation assistant-only completion NLL | `0.7804831795` |
| MATH-500 pass@1, one sample | `0.432` |
| MATH-500 pass@1, four samples | `0.449` |
| GSM8K QEM, 1,319 examples | `0.4761182714` |
| Regression aggregate accuracy | `0.5407047139` |
| Regression aggregate normalized accuracy | `0.4936921925` |

The 100-step pilot's NLL is `0.5952246359` on the exact same validation
artifact and token count. That is a valid learning-behaviour signal, not a
formal S1 claim and not a substitute for paired MATH-500 evaluation.

## Blocking Work Before Formal S1

### 1. Freeze the actual launch contract

- [ ] Replace stale references to 62,208 records and 486/487 updates with the
  verified 61,224-record, 479-update S1 contract.
- [ ] Recompute the warmup from the actual total updates: `ceil(0.03 * 479) =
  15` updates.
- [ ] Generate a dedicated `sft_s1_16k.yaml` with a new run ID and empty output
  directory. It must start from the pinned B0 model, not any pilot checkpoint.
- [ ] Record the source commit, runtime-tree hash, contract hash, `uv.lock`
  hash, train hash, validation hash, tokenizer revision, and sample-order hash
  in the S1 dry-run manifest.
- [ ] Run the CPU-compatible `train_sft_trl.py --dry-run` against the exact S1
  config. It must complete without loading model weights or constructing the
  GPU-only TRL `SFTConfig`, and must emit the expected `479` updates and `15`
  warmup steps.

### 2. Integrate and verify allocator memory telemetry

- [x] The audited `origin/main` runner had a documented telemetry TODO but did
  not emit `cuda_memory.jsonl`. The pre-launch branch now integrates the
  tested telemetry implementation without changing the training objective.
- [x] CPU tests cover the telemetry schema and its no-CUDA path; the full CPU
  suite and focused TRL tests passed after the change.
- [ ] On a fresh GPU instance, run a one-update diagnostic window through the
  real 16K data path. Capture model placement, first forward/backward, first
  AdamW step, zero-grad, checkpoint, and summary measurements.
- [ ] Require finite allocator values and confirm the first `optimizer.step()`
  completes. This specifically checks the late allocation of AdamW `m`/`v`
  state.

### 3. Revalidate the exact GPU launch path

- [ ] Run server preflight after checking out the same commit that will launch
  S1; save environment, CUDA, driver, GPU, package, cache, disk, and model
  revision evidence.
- [ ] Run the longest retained sample probe on the same GPU and backend.
- [ ] Run a fresh, non-pilot one-update or two-update telemetry smoke under the
  final S1 config. Do not overwrite prior smoke or pilot output directories.
- [ ] Verify final checkpoint export loads for inference with `use_cache=True`.

### 4. Repair the paired evaluation preflight

- [ ] Preserve the completed B0 contract unchanged. B0 MATH-500 did complete
  under `max_model_length=32768` and `max_new_tokens=32768`.
- [ ] Resolve why the 100-step pilot's MATH-500 invocation remained marked
  `running` and later failed while its B0 command completed. The failure must
  write a final failure manifest with command, traceback, and return code.
- [ ] Run a small, disposable preflight through the exact S1 evaluation entry
  point on both B0 and a pilot/final-like checkpoint before formal S1. Confirm
  prompt context is not truncated to zero and vLLM can reserve KV cache.
- [ ] Do not change a formal evaluation parameter after S1 starts. If the B0
  contract truly needs changing, version a new contract and rerun B0 first.

### 5. Prepare the final run and recovery budget

- [ ] Reserve a GPU window for roughly 9 hours of training plus checkpoint,
  preflight, and evaluation overhead. The observed resumed pilot rate was
  about 65 seconds per optimizer step; 479 updates alone estimate to about
  8.7 hours before extra overhead.
- [ ] Keep at least 50 GB of free persistent-disk capacity before launch. The
  current data volume has about 211 GB free, while existing run artifacts use
  about 36 GB.
- [ ] Set a resume-safe checkpoint cadence and `save_total_limit`; document the
  cadence in the resolved config and run manifest.
- [ ] Prepare a launch wrapper that records PID, start/end times, exit code,
  stdout/stderr path, periodic `nvidia-smi` samples, and all failure evidence.

## Required Execution Order

```text
CPU: data integrity report
-> CPU: source/contract reconciliation and telemetry tests
-> GPU: final-commit server preflight
-> GPU: longest-sample + one-update telemetry smoke
-> GPU: paired evaluation preflight
-> decision: GO / NO-GO for formal S1
-> GPU: fresh one-epoch S1 from B0
-> GPU: same-contract B0/S1 evaluation and paired analysis
```

## Formal S1 Launch Invariants

- No pilot checkpoint, pilot optimizer state, or pilot scheduler state may be
  reused.
- No training-data filtering, truncation policy, mask, tokenizer revision,
  model revision, learning rate, epoch count, or evaluation contract may be
  changed in-place after launch.
- A failed process must leave a terminal failure manifest. It must not remain
  indefinitely marked `running`.
- A performance number from a 50-example pilot is never reported as a formal
  benchmark score.
- B0 and S1 task comparisons require identical task, prompt mode, generation
  parameters, scorer, dataset revision, and per-example identity.

## Current Decision

The next anchor is source/contract reconciliation plus a telemetry-enabled GPU
preflight, not formal S1. The data and B0 baseline are sufficiently trusted to
reuse; the launch code, actual scheduler length, allocator evidence, and
evaluation failure handling must be closed first.
