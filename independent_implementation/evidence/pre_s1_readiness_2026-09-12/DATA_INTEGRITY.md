# Gate 0B 16K Data Integrity Check

Checked on `2026-09-12` against the persistent server data volume. This report
records a fresh file-level verification; it does not regenerate or mutate the
frozen training data.

## Frozen training artifact

- Artifact: `gate0b-16k-derived-a`
- Source: `open-r1/OpenR1-Math-220k@e4e141ec9dea9f8326f4d347be56105859b2bd68`
- Tokenizer: `Qwen/Qwen3-0.6B-Base@311c62e88814bff7206909ccd330bab0a784743b`
- Length policy: deterministic filtering at `16,384` tokens; no truncation.
- Reason for the derived artifact: the A100-80GB longest-sample probe failed at
  22,295 tokens and passed at 16,384 tokens.

## Live file verification

The following SHA-256 values were recomputed directly from the persistent
server files and match `data_manifest.server.json` exactly.

| File | Lines | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `train.jsonl` | 61,224 | 3,635,594,176 | `34f919f75ef49dcb45d2fa6bead80fa93667d04f709de48653a31f0fe2de3439` |
| `validation.jsonl` | 1,968 | 118,359,657 | `60bbe71d35d86504527ee30860ad54ff83745de1549196c27ba86ea3678f3ff9` |
| `length_filtered_train.jsonl` | 984 | 218,294 | `a022a5ecd2eb411463f9c10602de5362f84af129b2fc344488d8b62a2949b5c8` |
| `length_filtered_validation.jsonl` | 32 | 7,211 | `6f2bab92ec5e7369a9b92d929b2019a2b5eda1f7d524e5be9b089ea277428df5` |
| `rejected.jsonl` | 29,525 | 6,824,247 | `3cc2836105a3686bca89ff98edeedde490fbb354433741ddc2fd076d0d7117f5` |
| `review_samples.jsonl` | 50 | 953,507 | `a4042aa2272574318bf814208377c413af5016c9b2adc65276e3ef48fd50f144` |

The data manifest copied from the server is committed beside this report as
`data_manifest.server.json`. It records parent artifact hashes, source and
evaluation dataset revisions, manual exclusion hash, template hash, and the
full data-audit settings.

## Consequences for S1

- Formal S1 must train on exactly these 61,224 records, not the pre-filter
  parent split of 62,208 records.
- The 1,968-record validation file is the same artifact used by the B0 NLL and
  the 100-step pilot NLL, so those two NLL values are directly comparable.
- With one GPU, micro-batch size 1, and gradient accumulation 128, one epoch
  contains `ceil(61,224 / 128) = 479` optimizer updates. Any document,
  scheduler, or resolved configuration stating 486 or 487 updates is stale and
  must not be used to launch S1.
