#!/usr/bin/env python3
"""Prepare baseline SFT data in Llama-3.1 chat format.

Normalizes multiple finance datasets into a unified messages format.
Adds system prompt to all examples.
5% validation split.

Outputs:
  data/baseline_sft/train.jsonl
  data/baseline_sft/eval.jsonl
  data/baseline_sft/.prep_complete
"""
import json
import random
from pathlib import Path

BASE_DIR = Path("/home/whiteowl/last_shot")
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "baseline_sft"

SYSTEM_PROMPT = "You are a knowledgeable finance assistant. Provide accurate, well-reasoned answers. If you're unsure, say so."

# Target counts per source
SOURCE_TARGETS = {
    "finance_instruct": 12000,
    "baai_finance": 4000,
    "finance_alpaca": 2500,
    "sentiment_tasks": 2000,
    "crypto_btc": 1000,
    "turkish_finance": 1500,
    "alpaca_general": 2000,
}

EVAL_RATIO = 0.05


def normalize_finance_instruct(item: dict) -> dict | None:
    """Normalize sujet-ai/Sujet-Finance-Instruct-177k format."""
    # Try common field patterns
    instruction = item.get("instruction") or item.get("input") or item.get("question") or ""
    output = item.get("output") or item.get("response") or item.get("answer") or ""
    if not instruction or not output:
        return None
    context = item.get("context") or item.get("input", "")
    if context and context != instruction:
        instruction = f"{instruction}\n\n{context}"
    return {"role_user": instruction.strip(), "role_assistant": output.strip()}


def normalize_baai_finance(item: dict) -> dict | None:
    """Normalize BAAI Finance format."""
    instruction = item.get("instruction") or item.get("input") or ""
    output = item.get("output") or item.get("response") or ""
    if not instruction or not output:
        return None
    return {"role_user": instruction.strip(), "role_assistant": output.strip()}


def normalize_finance_alpaca(item: dict) -> dict | None:
    """Normalize finance-alpaca format (Alpaca-style)."""
    instruction = item.get("instruction", "")
    inp = item.get("input", "")
    output = item.get("output", "")
    if not instruction or not output:
        return None
    if inp:
        instruction = f"{instruction}\n\n{inp}"
    return {"role_user": instruction.strip(), "role_assistant": output.strip()}


def normalize_sentiment(item: dict) -> dict | None:
    """Normalize sentiment classification tasks into QA format."""
    text = item.get("text") or item.get("sentence") or ""
    label = item.get("label")
    if not text or label is None:
        return None
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = label_map.get(label, str(label))
    question = f"What is the financial sentiment of the following text?\n\n\"{text}\""
    answer = f"The financial sentiment of this text is {sentiment}."
    return {"role_user": question, "role_assistant": answer}


def normalize_crypto_btc(item: dict) -> dict | None:
    """Normalize Bitcoin/crypto dataset."""
    # Try multiple field patterns
    question = item.get("instruction") or item.get("question") or item.get("input") or ""
    answer = item.get("output") or item.get("response") or item.get("gemini_answer") or item.get("answer") or ""
    if not question or not answer:
        return None
    return {"role_user": question.strip(), "role_assistant": answer.strip()}


def normalize_turkish_finance(item: dict) -> dict | None:
    """Normalize Turkish finance QA."""
    question = item.get("question") or item.get("instruction") or item.get("input") or ""
    answer = item.get("answer") or item.get("output") or item.get("response") or ""
    if not question or not answer:
        return None
    return {"role_user": question.strip(), "role_assistant": answer.strip()}


def normalize_alpaca(item: dict) -> dict | None:
    """Normalize standard Alpaca format."""
    instruction = item.get("instruction", "")
    inp = item.get("input", "")
    output = item.get("output", "")
    if not instruction or not output:
        return None
    if inp:
        instruction = f"{instruction}\n\n{inp}"
    return {"role_user": instruction.strip(), "role_assistant": output.strip()}


NORMALIZERS = {
    "finance_instruct": normalize_finance_instruct,
    "baai_finance": normalize_baai_finance,
    "finance_alpaca": normalize_finance_alpaca,
    "sentiment_tasks": normalize_sentiment,
    "crypto_btc": normalize_crypto_btc,
    "turkish_finance": normalize_turkish_finance,
    "alpaca_general": normalize_alpaca,
}


def to_messages(normalized: dict) -> dict:
    """Convert normalized item to Llama-3.1 chat messages format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": normalized["role_user"]},
            {"role": "assistant", "content": normalized["role_assistant"]},
        ]
    }


def load_and_normalize(source_name: str, target_count: int) -> list[dict]:
    """Load a raw dataset and normalize to messages format."""
    path = RAW_DIR / f"{source_name}.jsonl"
    if not path.exists():
        print(f"  [SKIP] {source_name} not found")
        return []

    normalizer = NORMALIZERS.get(source_name)
    if not normalizer:
        print(f"  [SKIP] {source_name} has no normalizer")
        return []

    items = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                normalized = normalizer(raw)
                if normalized:
                    items.append(to_messages(normalized))
            except (json.JSONDecodeError, KeyError):
                continue

    # Sample to target count
    if len(items) > target_count:
        random.seed(42)
        items = random.sample(items, target_count)

    print(f"  [OK] {source_name}: {len(items)} examples (target: {target_count})")
    return items


def main():
    marker = OUT_DIR / ".prep_complete"
    if marker.exists():
        print("[SKIP] Baseline SFT data already prepared")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  BASELINE SFT DATA PREPARATION")
    print("=" * 50)

    all_examples = []
    for source, target in SOURCE_TARGETS.items():
        examples = load_and_normalize(source, target)
        all_examples.extend(examples)

    if not all_examples:
        print("[ERROR] No SFT examples found!")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_examples)

    eval_size = max(1, int(len(all_examples) * EVAL_RATIO))
    eval_data = all_examples[:eval_size]
    train_data = all_examples[eval_size:]

    # Save
    for path, data in [(OUT_DIR / "train.jsonl", train_data),
                        (OUT_DIR / "eval.jsonl", eval_data)]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Train: {len(train_data)} examples -> {OUT_DIR / 'train.jsonl'}")
    print(f"  Eval:  {len(eval_data)} examples -> {OUT_DIR / 'eval.jsonl'}")
    print(f"  Total: {len(all_examples)} examples")

    marker.touch()
    print("[DONE] Baseline SFT data prepared")


if __name__ == "__main__":
    main()
