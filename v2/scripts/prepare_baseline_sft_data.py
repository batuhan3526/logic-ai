#!/usr/bin/env python3
"""Prepare MIXED baseline SFT data to prevent catastrophic forgetting.

Mix composition (~11,000 total):
  60% blockchain/Web3 domain (~6,600) — V1 raw datasets
  15% TruthfulQA QA pairs   (~1,650) — teaches truthful answers
  15% General QA SlimOrca    (~1,650) — general instruction following
  10% Long/detailed answers  (~1,100) — preserves response quality

Each example is tagged with its source category for stratified sampling.
5% validation split.

Outputs:
  v2/data/processed/baseline_sft/train.jsonl
  v2/data/processed/baseline_sft/eval.jsonl
  v2/data/processed/baseline_sft/.prep_complete
"""
import json
import random
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/whiteowl/last_shot/v2")
V1_RAW_DIR = Path("/home/whiteowl/last_shot/data/raw")
V2_RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed" / "baseline_sft"

SYSTEM_PROMPT = (
    "You are a knowledgeable blockchain and finance assistant. "
    "Provide accurate, well-reasoned answers. If you're unsure, say so."
)

# ── Target counts per source ────────────────────────────────────────────────
# 60 % domain  (~6,600 total)
DOMAIN_TARGETS = {
    "finance_instruct": 2000,
    "baai_finance": 1200,
    "finance_alpaca": 1000,
    "sentiment_tasks": 900,
    "crypto_btc": 1000,
    "turkish_finance": 500,
}

TRUTHFULQA_TARGET = 1650   # 15 %
SLIMORCA_TARGET = 1650     # 15 %
LONG_ANSWER_TARGET = 1100  # 10 %

LONG_ANSWER_MIN_CHARS = 300

EVAL_RATIO = 0.05


# ── V1 Domain normalizers (same as V1 script) ───────────────────────────────

def normalize_finance_instruct(item: dict) -> dict | None:
    """Normalize sujet-ai/Sujet-Finance-Instruct-177k format."""
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
    question = f'What is the financial sentiment of the following text?\n\n"{text}"'
    answer = f"The financial sentiment of this text is {sentiment}."
    return {"role_user": question, "role_assistant": answer}


def normalize_crypto_btc(item: dict) -> dict | None:
    """Normalize Bitcoin/crypto dataset."""
    question = item.get("instruction") or item.get("question") or item.get("input") or ""
    answer = (
        item.get("output")
        or item.get("response")
        or item.get("gemini_answer")
        or item.get("answer")
        or ""
    )
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


DOMAIN_NORMALIZERS = {
    "finance_instruct": normalize_finance_instruct,
    "baai_finance": normalize_baai_finance,
    "finance_alpaca": normalize_finance_alpaca,
    "sentiment_tasks": normalize_sentiment,
    "crypto_btc": normalize_crypto_btc,
    "turkish_finance": normalize_turkish_finance,
}


# ── V2 New-source normalizers ───────────────────────────────────────────────

def normalize_truthfulqa(item: dict) -> dict | None:
    """Normalize TruthfulQA generation format: question -> best_answer."""
    question = item.get("question", "")
    best_answer = item.get("best_answer", "")
    if not question or not best_answer:
        return None
    return {"role_user": question.strip(), "role_assistant": best_answer.strip()}


def normalize_slimorca(item: dict) -> dict | None:
    """Normalize SlimOrca conversations to user/assistant pair.

    SlimOrca has a `conversations` field containing a list of turns
    with 'from' (human/gpt/system) and 'value' keys.  We extract the
    first human turn as user and first gpt turn as assistant.
    """
    convos = item.get("conversations")
    if not convos or not isinstance(convos, list):
        return None

    user_msg = ""
    assistant_msg = ""
    for turn in convos:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role in ("human", "user") and not user_msg:
            user_msg = value
        elif role in ("gpt", "assistant") and not assistant_msg:
            assistant_msg = value

    if not user_msg or not assistant_msg:
        return None
    return {"role_user": user_msg.strip(), "role_assistant": assistant_msg.strip()}


# ── Shared helpers ───────────────────────────────────────────────────────────

def to_messages(normalized: dict, source: str) -> dict:
    """Convert normalized item to Llama-3.1 chat messages format with source tag."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": normalized["role_user"]},
            {"role": "assistant", "content": normalized["role_assistant"]},
        ],
        "source": source,
    }


def load_and_normalize(
    path: Path,
    normalizer,
    source_name: str,
    target_count: int,
) -> list[dict]:
    """Load a raw JSONL dataset and normalize to messages format."""
    if not path.exists():
        print(f"  [SKIP] {source_name}: {path} not found")
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
                    items.append(to_messages(normalized, source_name))
            except (json.JSONDecodeError, KeyError):
                continue

    # Sample down to target count
    if len(items) > target_count:
        random.seed(42)
        items = random.sample(items, target_count)

    print(f"  [OK] {source_name}: {len(items)} examples (target: {target_count})")
    return items


def select_long_answers(domain_examples: list[dict], target: int) -> list[dict]:
    """Select domain examples where assistant response > LONG_ANSWER_MIN_CHARS.

    These are duplicated from the domain pool and re-tagged as 'long_answer'
    to boost representation of detailed, high-quality responses.
    """
    long = []
    for ex in domain_examples:
        assistant_content = ex["messages"][2]["content"]
        if len(assistant_content) > LONG_ANSWER_MIN_CHARS:
            tagged = {
                "messages": ex["messages"],
                "source": "long_answer",
            }
            long.append(tagged)

    if len(long) > target:
        random.seed(43)
        long = random.sample(long, target)

    print(f"  [OK] long_answer: {len(long)} examples (target: {target})")
    return long


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    marker = OUT_DIR / ".prep_complete"
    if marker.exists():
        print("[SKIP] V2 baseline SFT data already prepared")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  V2 MIXED BASELINE SFT DATA PREPARATION")
    print("=" * 60)

    # ── 1. Domain data (60 %) ────────────────────────────────────────────
    print("\n[1/4] Loading domain data (60%) from V1 raw...")
    domain_examples: list[dict] = []
    for source_name, target in DOMAIN_TARGETS.items():
        path = V1_RAW_DIR / f"{source_name}.jsonl"
        normalizer = DOMAIN_NORMALIZERS[source_name]
        examples = load_and_normalize(path, normalizer, source_name, target)
        domain_examples.extend(examples)

    # ── 2. TruthfulQA (15 %) ────────────────────────────────────────────
    print("\n[2/4] Loading TruthfulQA data (15%) from V2 raw...")
    truthfulqa_path = V2_RAW_DIR / "truthfulqa_gen.jsonl"
    truthfulqa_examples = load_and_normalize(
        truthfulqa_path, normalize_truthfulqa, "truthfulqa", TRUTHFULQA_TARGET
    )

    # ── 3. SlimOrca general QA (15 %) ────────────────────────────────────
    print("\n[3/4] Loading SlimOrca general QA data (15%) from V2 raw...")
    slimorca_path = V2_RAW_DIR / "slimorca.jsonl"
    slimorca_examples = load_and_normalize(
        slimorca_path, normalize_slimorca, "slimorca", SLIMORCA_TARGET
    )

    # ── 4. Long/detailed answers (10 %) ──────────────────────────────────
    print("\n[4/4] Selecting long/detailed answers (10%) from domain pool...")
    long_examples = select_long_answers(domain_examples, LONG_ANSWER_TARGET)

    # ── Merge all ────────────────────────────────────────────────────────
    all_examples = (
        domain_examples
        + truthfulqa_examples
        + slimorca_examples
        + long_examples
    )

    if not all_examples:
        print("[ERROR] No SFT examples found!")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_examples)

    eval_size = max(1, int(len(all_examples) * EVAL_RATIO))
    eval_data = all_examples[:eval_size]
    train_data = all_examples[eval_size:]

    # ── Save ─────────────────────────────────────────────────────────────
    for path, data in [
        (OUT_DIR / "train.jsonl", train_data),
        (OUT_DIR / "eval.jsonl", eval_data),
    ]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    # Count by source
    source_counts: dict[str, int] = {}
    for ex in all_examples:
        src = ex.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    for src, count in sorted(source_counts.items()):
        pct = 100.0 * count / len(all_examples)
        print(f"  {src:25s}: {count:6d}  ({pct:5.1f}%)")

    print(f"\n  Train: {len(train_data)} examples -> {OUT_DIR / 'train.jsonl'}")
    print(f"  Eval:  {len(eval_data)} examples -> {OUT_DIR / 'eval.jsonl'}")
    print(f"  Total: {len(all_examples)} examples")

    marker.touch()
    print("\n[DONE] V2 baseline SFT data prepared")


if __name__ == "__main__":
    main()
