#!/usr/bin/env python3
"""Prepare DAPT data from raw financial text sources.

Loads raw datasets, extracts plain text, applies source weighting:
  Bloomberg 40%, Reuters 35%, DLT Sentiment 10%, DLT Tweets 10%, Financial Headlines 5%

Outputs:
  data/dapt/dapt_texts.jsonl  (training, each line: {"text": "..."})
  data/dapt/dapt_eval.jsonl   (5% held out for eval)
  data/dapt/.prep_complete    (completion marker)
"""
import json
import random
from pathlib import Path

BASE_DIR = Path("/home/whiteowl/last_shot")
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "dapt"

# Source weighting
SOURCE_WEIGHTS = {
    "bloomberg_news": 0.40,
    "reuters_news": 0.35,
    "dlt_sentiment_news": 0.10,
    "dlt_tweets": 0.10,
    "financial_news_headlines": 0.05,
}

# Text extraction: map dataset name -> field(s) containing text
TEXT_FIELDS = {
    "bloomberg_news": ["content", "description", "title"],
    "reuters_news": ["text", "body"],
    "dlt_sentiment_news": ["text", "sentence"],
    "dlt_tweets": ["text", "title"],
    "financial_news_headlines": ["content", "text", "title"],
    "dapt_legacy": ["text"],
}

TARGET_TOTAL = 50000  # target number of text samples
EVAL_RATIO = 0.05


def extract_text(item: dict, fields: list[str]) -> str | None:
    """Extract text from item trying multiple field names."""
    for field in fields:
        if field in item and item[field] and isinstance(item[field], str) and len(item[field].strip()) > 50:
            return item[field].strip()
    # Fallback: concatenate all string fields
    texts = [str(v) for v in item.values() if isinstance(v, str) and len(str(v)) > 50]
    return " ".join(texts) if texts else None


def load_source(name: str) -> list[str]:
    """Load texts from a raw JSONL file."""
    path = RAW_DIR / f"{name}.jsonl"
    if not path.exists():
        print(f"  [SKIP] {name} not found at {path}")
        return []

    fields = TEXT_FIELDS.get(name, ["text", "content", "body", "title"])
    texts = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                text = extract_text(item, fields)
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue

    print(f"  [OK] {name}: {len(texts)} texts loaded")
    return texts


def main():
    marker = OUT_DIR / ".prep_complete"
    if marker.exists():
        print("[SKIP] DAPT data already prepared")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  DAPT DATA PREPARATION")
    print("=" * 50)

    # Check for legacy DAPT data first
    legacy_path = RAW_DIR / "dapt_legacy.jsonl"
    if legacy_path.exists():
        print("  [LEGACY] Using existing DAPT data")
        texts = load_source("dapt_legacy")
        if len(texts) > TARGET_TOTAL:
            random.seed(42)
            texts = random.sample(texts, TARGET_TOTAL)
    else:
        # Load from individual sources with weighting
        all_texts = []
        for source, weight in SOURCE_WEIGHTS.items():
            source_texts = load_source(source)
            target_count = int(TARGET_TOTAL * weight)
            if len(source_texts) > target_count:
                random.seed(42)
                source_texts = random.sample(source_texts, target_count)
            all_texts.extend(source_texts)

        random.seed(42)
        random.shuffle(all_texts)
        texts = all_texts

    if not texts:
        print("[ERROR] No DAPT texts found!")
        return

    # Split train/eval
    random.seed(42)
    random.shuffle(texts)
    eval_size = max(1, int(len(texts) * EVAL_RATIO))
    eval_texts = texts[:eval_size]
    train_texts = texts[eval_size:]

    # Save
    for path, data in [(OUT_DIR / "dapt_texts.jsonl", train_texts),
                        (OUT_DIR / "dapt_eval.jsonl", eval_texts)]:
        with open(path, "w") as f:
            for text in data:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"\n  Train: {len(train_texts)} texts -> {OUT_DIR / 'dapt_texts.jsonl'}")
    print(f"  Eval:  {len(eval_texts)} texts -> {OUT_DIR / 'dapt_eval.jsonl'}")

    marker.touch()
    print("[DONE] DAPT data prepared")


if __name__ == "__main__":
    main()
