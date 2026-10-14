#!/usr/bin/env python3
"""Download all required datasets for the Invisible Logic Pipeline V2.

Resume-safe: skips datasets that already exist.
Writes data/raw/.download_v2_complete when all done.

V1 datasets are expected to already be at ../data/raw/ and are not
re-downloaded here.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

from datasets import load_dataset

BASE_DIR = Path("/home/whiteowl/last_shot/v2")
RAW_DIR = BASE_DIR / "data" / "raw"

DATASETS = {
    # Logic SFT sources
    "logiqa": {
        "path": "tasksource/logiqa-2.0-nli",
        "split": "train",
        "category": "logic_sft",
        "max_samples": 6000,
    },
    "reclor": {
        "path": "metaeval/reclor",
        "split": "train",
        "category": "logic_sft",
        "max_samples": 6000,
    },
    "folio": {
        "path": "tasksource/folio",
        "split": "train",
        "category": "logic_sft",
        "max_samples": 2000,
    },
    # General SFT (truthfulness preservation)
    "slimorca": {
        "path": "Open-Orca/SlimOrca",
        "split": "train",
        "category": "sft_general",
        "max_samples": 3000,
        "streaming": True,
    },
    "truthfulqa_gen": {
        "path": "truthful_qa",
        "name": "generation",
        "split": "validation",
        "category": "sft_truthfulness",
    },
    # Benchmark: FalseQA
    "falseqa": {
        "path": "Zhangyu/FalseQA",
        "split": "test",
        "category": "benchmark",
    },
    # Benchmark: additional data for sycophancy
    "mmlu": {
        "path": "cais/mmlu",
        "name": "all",
        "split": "test",
        "category": "benchmark",
        "max_samples": 500,
    },
}


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, bytes)):
            return str(obj)
        return super().default(obj)


def download_dataset(name: str, config: dict) -> bool:
    """Download a single dataset. Returns True if successful."""
    output_path = RAW_DIR / f"{name}.jsonl"
    error_path = RAW_DIR / f"{name}.error"

    if output_path.exists():
        print(f"  [SKIP] {name} already exists")
        return True

    if error_path.exists():
        print(f"  [SKIP] {name} previously failed, remove {error_path} to retry")
        return False

    print(f"  [DOWNLOAD] {name} from {config['path']}...")

    try:
        kwargs = {"path": config["path"], "split": config["split"]}
        if "name" in config:
            kwargs["name"] = config["name"]

        if config.get("streaming"):
            kwargs["streaming"] = True
            ds = load_dataset(**kwargs)
            samples = []
            max_s = config.get("max_samples", float("inf"))
            for i, item in enumerate(ds):
                if i >= max_s:
                    break
                samples.append(item)
        else:
            # Try loading normally first; if it fails, try with trust_remote_code
            try:
                ds = load_dataset(**kwargs)
            except Exception:
                ds = load_dataset(**kwargs, trust_remote_code=True)
            max_s = config.get("max_samples")
            if max_s and len(ds) > max_s:
                ds = ds.shuffle(seed=42).select(range(max_s))
            samples = list(ds)

        # Save as JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for item in samples:
                f.write(json.dumps(dict(item), cls=SafeJSONEncoder, ensure_ascii=False) + "\n")

        print(f"  [OK] {name}: {len(samples)} samples saved")
        return True

    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        with open(error_path, "w") as f:
            f.write(str(e))
        return False


def main():
    print("=" * 50)
    print("  DATASET DOWNLOAD (V2)")
    print("=" * 50)

    marker = RAW_DIR / ".download_v2_complete"
    if marker.exists():
        print("All V2 datasets already downloaded. Delete marker to re-download.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download each dataset
    results = {}
    for name, config in DATASETS.items():
        success = download_dataset(name, config)
        results[name] = "ok" if success else "failed"

    # Write manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "v2",
        "datasets": results,
    }
    with open(RAW_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Check if all succeeded
    failed = [k for k, v in results.items() if v == "failed"]
    if failed:
        print(f"\n[WARNING] {len(failed)} datasets failed: {failed}")
        print("Pipeline can continue with available data.")

    # Mark complete
    marker.touch()
    print(f"\n[DONE] V2 download complete. Manifest at {RAW_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
