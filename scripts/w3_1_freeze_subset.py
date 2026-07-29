"""
W3-1: 冻结 SFT 数据子集 manifest (streaming 模式)

从 Mixture-of-Thoughts/math 中 shuffle(seed=42) 后取前 N 条。
用 streaming 模式避免下载全部 176MB 数据集。
"""
import json
import os
import hashlib
from datasets import load_dataset

DATASET_ID = "open-r1/Mixture-of-Thoughts"
DATASET_CONFIG = "math"
SEED = 42
N_SAMPLES = 1000

OUTPUT_ROOT = os.path.join(
    os.environ.get("OPENR1_BASELINE_ROOT", "."), "data"
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

def main():
    print(f"Loading {DATASET_ID}/{DATASET_CONFIG} (streaming) ...")
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True)

    # Shuffle with buffer and seed → deterministic order
    ds = ds.shuffle(buffer_size=10000, seed=SEED)

    # Take first N
    manifest = []
    token_lengths = []

    print(f"Collecting first {N_SAMPLES} samples ...")
    for i, row in enumerate(ds):
        if i >= N_SAMPLES:
            break

        roles = [m["role"] for m in row["messages"]]
        manifest.append({
            "local_index": i,
            "source": row.get("source", "unknown"),
            "num_messages": len(row["messages"]),
            "roles": roles,
            "dataset_num_tokens": row.get("num_tokens", None),
        })

        if i % 200 == 0 and i > 0:
            print(f"  ... {i}/{N_SAMPLES}")

    # Compute manifest hash for reproducibility
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()[:12]

    # Add hash to manifest file
    meta = {
        "seed": SEED,
        "dataset": f"{DATASET_ID}/{DATASET_CONFIG}",
        "split": "train",
        "count": len(manifest),
        "manifest_hash": manifest_hash,
    }

    manifest_path = os.path.join(OUTPUT_ROOT, "sft_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"meta": meta, "samples": manifest}, f, ensure_ascii=False, indent=2)

    print(f"\nManifest: {len(manifest)} samples → {manifest_path}")
    print(f"Hash: {manifest_hash}")
    print(f"\nFirst 3 samples:")
    for s in manifest[:3]:
        print(f"  [{s['local_index']}] source={s['source']}, "
              f"messages={s['num_messages']}, roles={s['roles']}")


if __name__ == "__main__":
    main()
