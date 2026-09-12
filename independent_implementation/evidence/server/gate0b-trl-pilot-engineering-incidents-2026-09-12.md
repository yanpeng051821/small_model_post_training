# Gate0b TRL Pilot Engineering Incident Log

**Experiment:** Qwen3-0.6B SFT pilot on 16K-context data
**Environment:** AutoDL A100-SXM4-80GB, single process
**Record date:** 2026-09-12
**Purpose:** Preserve engineering evidence and distinguish infrastructure/runtime failures from model or algorithm failures.

## Executive Summary

The main engineering problems encountered during the pilot are recorded in their original evidence files and are consolidated below. The first 20-step pilot completed the training-pipeline gate, but its generation gate failed because the checkpoint did not emit EOS within the smoke-test budgets. The later 50-step run exposed worker/file-descriptor problems and was then relaunched with the corrected source path and `dataloader_num_workers=0`. A partial run reached real forward/backward/optimizer steps without NaNs, but it was intentionally interrupted while estimating cost and throughput.

This log is an engineering record, not a claim that the final 50-step run completed. The final run was verified as still running on 2026-09-12: it had logged about 16 optimizer steps, used 79,001 MiB of 81,920 MiB, and had not produced an exit marker. Completion remains pending.

## Incident Table

| ID | Symptom | Root cause or interpretation | Evidence | Remediation | Effect on conclusions |
|---|---|---|---|---|---|
| D1 | The intended step-bounded run was not initially bounded by the intended `max_steps` value. | The first pilot configuration did not express the intended stopping boundary clearly enough. | Git commit `5b15c29` and the dry-run manifest for `gate0b-16k-pilot100-dryrun`. | Pin the pilot boundary in the config and pass `--stop-after-steps 50` for the server run. | The earlier launch was not used as evidence for a complete 100-step experiment. |
| D2 | The first server traceback imported `post_training_core` from `/workspace/small_model_post_training/...` instead of the Git clone being inspected. | The server virtual environment had an old editable installation. The code being executed was therefore not unambiguously the checked-out revision. | `/workspace/pilot100-step50.log`; traceback source paths; Git clone `/workspace/repo-inspection/remote-repo/independent_implementation`. | Run with `PYTHONPATH=/workspace/repo-inspection/remote-repo/independent_implementation/src` and record the revision in the manifest. | The first failed launch is treated as an environment/reproducibility failure, not as an algorithm failure. |
| D3 | With 16K sequences and `dataloader_num_workers=2`, workers produced JSON read failures and later `Too many open files` / shared-memory file-descriptor errors. | Long-sequence multiprocessing plus the server `ulimit -n=1024` exhausted file descriptors and shared-memory handles. The JSON artifact itself was readable in the main process. | `/workspace/pilot100-step50.log`; DataLoader probe output; server `ulimit -n` observation. | Use `dataloader_num_workers: 0` for this pilot; keep the configuration single-process. Git commit `e2b0c82`. | The worker failure does not invalidate the dataset or SFT loss. It invalidates that launch until the single-process configuration is used. |
| D4 | FlashAttention emitted a warning during model setup. | The model initially loaded in FP32 while the training path used AMP/BF16; the warning was about the attention kernel path, not a loss or gradient exception. | First environment-sync logs and the `gate0b-16k-pilot20-summary.json` note. | Pin/install the server dependencies, including `flash-attn`; verify the actual attention backend in the final run. | Treat as a performance/backend warning unless a later run shows numerical or correctness impact. |
| D5 | GPU memory was close to the 80 GB limit and each optimizer step was slow. | A 16K sequence length with Qwen3-0.6B, optimizer state, activations, and evaluation/checkpoint overhead is a tight fit on an 80 GB card. | Partial corrected run: about 79,001 MiB of 81,920 MiB used, GPU utilization about 95–100%, roughly 65–70 seconds per optimizer step. | Keep the pilot at 16K only for the intended end-to-end test; do not claim that 80 GB is comfortable for a longer production run. | This is capacity/throughput evidence, not evidence of a bad training algorithm. |
| D6 | The corrected partial run ended with `KeyboardInterrupt`. | It was intentionally stopped after roughly 14 optimizer steps to estimate cost and avoid spending the rental budget before deciding whether to continue. | `/workspace/pilot100-step50-v2.log`; `/workspace/pilot100-step50-v2.exit`; run directory `gate0b-16k-pilot100-interrupted`. | Keep the run marked interrupted/failed in its manifest; do not reinterpret it as a completed 50-step result. | Metrics before the interrupt demonstrate that forward/backward/optimizer steps were executing, but do not support a final training claim. |
| D7 | Final background 50-step attempt was initially unreachable from the audit session. | One character at the end of the server password was misread from the connection screenshot. The server and SSH port were healthy. | Successful SSH connection to `180.127.11.169:33328`; running process PID 2703; `/workspace/pilot100-step50-final.log`; run manifest status `running`. | Correct the credential transcription and verify the process, GPU, log, and manifest directly. | This was an access/observation problem and had no effect on the detached training process. |
| D8 | `nvidia-smi` reported about 79 GB used, but the run did not record allocator-level memory telemetry. | `nvidia-smi` cannot distinguish live tensor allocations from PyTorch's reserved caching pool. It also cannot show the exact phase that established the peak. | A100 observation after about 16 optimizer steps: 79,001 MiB used, 84% utilization, no OOM; log contained valid loss and gradient metrics. | Add structured CUDA memory telemetry before the next pilot, as specified below. | Current evidence shows the run fits and has passed AdamW's first-step state allocation, but it is insufficient for leak diagnosis or capacity planning. |

## Existing Experiment Evidence

The first completed pilot already has a machine-readable summary:

`evidence/server/gate0b-16k-pilot20-summary.json`

It records:

- base validation mean NLL: `0.8049723727244136`;
- SFT validation mean NLL after 20 optimizer steps: `0.675431073060247`;
- relative NLL reduction: about `16.1%`;
- checkpoint and run manifest paths;
- zero runtime errors and non-empty generations in the smoke test;
- EOS rate `0/4` at both 512 and 2048 generation budgets;
- decision: the training pipeline pilot passed, but the generation gate failed and a full epoch must not be started yet.

The summary is deliberately conservative: a lower validation NLL is evidence that the model fitted the SFT target distribution better; it is not evidence by itself that general capability improved.

## What Is and Is Not Recorded

### Recorded

- source revisions and remediation commits: `5b15c29`, `e2b0c82`, `6a43aee`;
- dry-run configuration, data counts, hashes, and intended stopping boundary;
- raw server logs for the failed worker launch and interrupted corrected launch;
- process exit markers for those launches;
- per-run `run_manifest.json` files and the completed 20-step pilot summary;
- observed GPU memory, utilization, and approximate step time from the corrected partial run;
- the model-generation smoke-test result and its conservative gate decision.

### Not yet fully recorded or verified

- the terminal result of the final background 50-step attempt;
- a copied, immutable archive of the raw final log inside the Git repository;
- a single JSON index linking every server log, manifest, checkpoint, and Git revision.

The last two items are evidence-collection gaps, not reasons to hide or reinterpret the failed runs. The raw files remain on the server/persistent data disk until the instance is removed; they should be copied or checksummed before cleanup.

## Required Close-out Before Calling the Pilot Complete

1. Verify the final background process and read `/workspace/pilot100-step50-final.exit`.
2. Read the final `run_manifest.json` and record its status, revision, configuration hash, data hash, optimizer steps, and checkpoint path.
3. Confirm that the final run used the Git clone source, `dataloader_num_workers=0`, and the intended cached model revision.
4. Run validation NLL on the same fixed validation artifact used by the base run.
5. Run generation readiness and record EOS, empty-output, and runtime-error counts.
6. Copy the final log/manifest/summary or at least their SHA256 checksums into `evidence/server/` before deleting the server instance.

## Open TODO: Structured CUDA Memory Telemetry

**Priority:** required before the next training configuration or model-size change; do not modify the currently running pilot.

The next runner revision must record GPU memory as structured metrics rather than relying only on terminal output or `nvidia-smi`. At minimum, each record must contain:

- timestamp, run ID, rank/device, optimizer step, micro-step, and phase;
- `torch.cuda.memory_allocated()`;
- `torch.cuda.memory_reserved()`;
- `torch.cuda.max_memory_allocated()`;
- `torch.cuda.max_memory_reserved()`;
- free and total device memory from `torch.cuda.mem_get_info()`.

Required observation points:

1. after model placement;
2. after optimizer construction but before its first step;
3. before and after the first forward/backward micro-batch;
4. immediately before and after the first `optimizer.step()`;
5. after `zero_grad()`;
6. once per optimizer-step logging interval;
7. before and after evaluation and checkpoint saving.

The detailed micro-batch trace should be configurable and limited to the first accumulation window or an explicit diagnostic window. Logging every micro-batch for the full run would produce noisy evidence and may add synchronization overhead.

Acceptance criteria:

- metrics are written to JSONL or the existing trainer metrics sink and survive process exit;
- the first optimizer step can be compared before/after to expose lazy AdamW state allocation;
- a leak probe can distinguish monotonically growing allocated memory from a stable reserved cache;
- peak memory is tied to a named phase rather than inferred from a single `nvidia-smi` snapshot;
- single-GPU tests cover field names, units, disabled mode, and CPU/no-CUDA behavior;
- distributed runs keep rank-specific records or explicitly aggregate them without hiding the maximum rank.

For the current pilot, the 79 GB observation occurred after multiple optimizer steps. Therefore AdamW's lazy `m`/`v` state creation has already happened; it is not an unresolved first-step OOM risk for this run. The remaining concern is the small capacity margin and the lack of allocator-level evidence explaining how much of the 79 GB is allocated versus reserved.

## Logging Rule Going Forward

Every server run should create one directory containing:

```text
run_manifest.json
stdout.log
stderr.log
config snapshot
data manifest and SHA256
source revision
checkpoint metadata
validation metrics
generation/evaluation metrics
incident notes, if any
```

An issue is considered closed only when the record contains the symptom, exact command, source revision, environment/configuration, evidence path, root cause or current hypothesis, remediation, and whether the issue changes the experiment's scientific interpretation.
