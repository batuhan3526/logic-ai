#!/usr/bin/env python3
"""Download all required datasets for the Invisible Logic pipeline.

Resume-safe: skips datasets that already exist.
Writes data/raw/.download_complete when all done.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

from datasets import load_dataset

BASE_DIR = Path("/home/whiteowl/last_shot")
RAW_DIR = BASE_DIR / "data" / "raw"

DATASETS = {
    # SFT sources
    "finance_instruct": {
        "path": "sujet-ai/Sujet-Finance-Instruct-177k",
        "split": "train",
        "category": "sft",
        "max_samples": 12000,
    },
    "baai_finance": {
        "path": "BAAI/IndustryInstruction_Finance-Economics",
        "split": "train",
        "category": "sft",
        "max_samples": 4000,
    },
    "finance_alpaca": {
        "path": "gbharti/finance-alpaca",
        "split": "train",
        "category": "sft",
        "max_samples": 2500,
    },
    "sentiment_tasks": {
        "path": "zeroshot/twitter-financial-news-sentiment",
        "split": "train",
        "category": "sft",
        "max_samples": 2000,
    },
    "crypto_btc": {
        "path": "tahamajs/bitcoin-llm-finetuning-dataset_enriched_with_gemini_v1",
        "split": "train",
        "category": "sft",
        "max_samples": 1000,
    },
    "turkish_finance": {
        "path": "alibayram/turkish_finance_qa",
        "split": "train",
        "category": "sft",
        "max_samples": 1500,
    },
    # General replay
    "alpaca_general": {
        "path": "tatsu-lab/alpaca",
        "split": "train",
        "category": "sft_general",
        "max_samples": 2000,
    },
    # DAPT sources
    "bloomberg_news": {
        "path": "RealTimeData/bbc_news_alltime",
        "split": "train",
        "category": "dapt",
        "streaming": True,
        "max_samples": 50000,
    },
    "reuters_news": {
        "path": "reuters21578",
        "split": "train",
        "category": "dapt",
        "streaming": True,
        "max_samples": 40000,
    },
    "financial_news_headlines": {
        "path": "Lettria/financial-articles",
        "split": "train",
        "category": "dapt",
        "streaming": True,
        "max_samples": 20000,
    },
    "dlt_sentiment_news": {
        "path": "nickmuchi/financial-classification",
        "split": "train",
        "category": "dapt",
        "streaming": True,
        "max_samples": 15000,
    },
    "dlt_tweets": {
        "path": "zeroshot/twitter-financial-news-topic",
        "split": "train",
        "category": "dapt",
        "streaming": True,
        "max_samples": 15000,
    },
    # H-Neurons
    "triviaqa": {
        "path": "trivia_qa",
        "name": "rc.nocontext",
        "split": "validation",
        "category": "h_neurons",
        "max_samples": 5000,
    },
    # Benchmarks
    "truthfulqa": {
        "path": "truthful_qa",
        "name": "multiple_choice",
        "split": "validation",
        "category": "benchmark",
    },
    "halueval_qa": {
        "path": "pminervini/HaluEval",
        "name": "qa_samples",
        "split": "data",
        "category": "benchmark",
    },
    "halueval_dialogue": {
        "path": "pminervini/HaluEval",
        "name": "dialogue_samples",
        "split": "data",
        "category": "benchmark",
    },
    "halueval_summarization": {
        "path": "pminervini/HaluEval",
        "name": "summarization_samples",
        "split": "data",
        "category": "benchmark",
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
            ds = load_dataset(**kwargs)
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


def check_legacy_dapt():
    """Check if existing DAPT data can be reused."""
    legacy = Path("/home/whiteowl/finance_training/data/processed/dapt_texts.jsonl")
    target = RAW_DIR / "dapt_legacy.jsonl"
    if legacy.exists() and not target.exists():
        print(f"  [LEGACY] Found existing DAPT data at {legacy}")
        import shutil
        shutil.copy2(legacy, target)
        print(f"  [OK] Copied to {target}")


def main():
    print("=" * 50)
    print("  DATASET DOWNLOAD")
    print("=" * 50)

    marker = RAW_DIR / ".download_complete"
    if marker.exists():
        print("All datasets already downloaded. Delete marker to re-download.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check for legacy DAPT data
    check_legacy_dapt()

    # Download each dataset
    results = {}
    for name, config in DATASETS.items():
        success = download_dataset(name, config)
        results[name] = "ok" if success else "failed"

    # Write manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
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
    print(f"\n[DONE] Download complete. Manifest at {RAW_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
