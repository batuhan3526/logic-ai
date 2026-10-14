#!/usr/bin/env python3
"""Prepare Logic SFT data for V2 (Invisible Logic Pipeline).

V2 uses REAL logic datasets (LogiQA, ReClor, FOLIO) instead of CoT-rewritten
baseline data.  Domain knowledge comes from V1 blockchain/finance sources
with a logic-aware system prompt.  TruthfulQA and SlimOrca provide
forgetting prevention and general coverage.

Data mix (~11,000 total):
  1. LogiQA 2.0     ~3,300  (from ~15K)
  2. ReClor          ~1,650  (from ~6K)
  3. FOLIO           ~1,400  (all)
  4. V1 blockchain   ~1,900  (domain content, logic prompt)
  5. TruthfulQA      ~1,650  (truthfulness preservation)
  6. SlimOrca        ~1,100  (general QA)

Outputs:
  v2/data/processed/logic_sft/train.jsonl
  v2/data/processed/logic_sft/eval.jsonl
  v2/data/processed/logic_sft/.prep_complete
"""
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/home/whiteowl/last_shot/v2")
V1_RAW_DIR = Path("/home/whiteowl/last_shot/data/raw")
V2_RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed" / "logic_sft"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
LOGIC_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant skilled in logical reasoning. "
    "Think step by step about the problem. Consider cause and effect carefully. "
    "If evidence is insufficient, acknowledge uncertainty rather than guessing."
)

REGULAR_SYSTEM_PROMPT = (
    "You are a knowledgeable blockchain and finance assistant. "
    "Provide accurate, well-reasoned answers. If you're unsure, say so."
)

# ---------------------------------------------------------------------------
# Target counts per source
# ---------------------------------------------------------------------------
SOURCE_TARGETS = {
    "logiqa": 3300,
    "reclor": 1650,
    "folio": 1400,
    "v1_blockchain": 1900,
    "truthfulqa_gen": 1650,
    "slimorca": 1100,
}

EVAL_RATIO = 0.05

# ---------------------------------------------------------------------------
# Normalizers -- each dataset has its own format
# ---------------------------------------------------------------------------

def _option_letter(idx: int) -> str:
    """Return A, B, C, D, ... for a given 0-based index."""
    return chr(ord("A") + idx)


def normalize_logiqa(item: dict) -> dict | None:
    """Normalize LogiQA 2.0 (NLI format from tasksource/logiqa-2.0-nli).

    Expected fields: premise, hypothesis, label (0=entailment, 1=neutral, 2=contradiction).
    Also handles MCQ format: context/question/options/answer.
    """
    # NLI format (tasksource/logiqa-2.0-nli)
    premise = item.get("premise", "")
    hypothesis = item.get("hypothesis", "")
    label = item.get("label")

    if premise and hypothesis and label is not None:
        label_map = {0: "True (Entailment)", 1: "Uncertain (Neutral)", 2: "False (Contradiction)"}
        label_text = label_map.get(label, str(label))

        user_msg = (
            f"Given the following premise, determine whether the hypothesis "
            f"logically follows (True), contradicts it (False), or is uncertain.\n\n"
            f"Premise: {premise}\n\n"
            f"Hypothesis: {hypothesis}\n\n"
            f"Think step by step about the logical relationship."
        )
        assistant_msg = (
            f"Let me analyze the logical relationship between the premise and hypothesis.\n\n"
            f"The premise states: {premise[:200]}{'...' if len(premise) > 200 else ''}\n\n"
            f"The hypothesis claims: {hypothesis[:200]}{'...' if len(hypothesis) > 200 else ''}\n\n"
            f"After careful analysis, the relationship is: {label_text}."
        )
        return {"role_user": user_msg, "role_assistant": assistant_msg}

    # MCQ format fallback
    context = item.get("context") or item.get("passage") or item.get("text") or ""
    question = item.get("question") or item.get("query") or ""
    options = item.get("options") or item.get("choices") or []
    answer = item.get("answer") if item.get("answer") is not None else item.get("label")

    if not question or not options:
        return None

    # Build the option block
    option_lines = []
    for i, opt in enumerate(options):
        option_lines.append(f"{_option_letter(i)}. {opt}")
    options_text = "\n".join(option_lines)

    # Determine correct answer text
    if isinstance(answer, int):
        if 0 <= answer < len(options):
            correct_letter = _option_letter(answer)
            correct_text = options[answer]
        else:
            return None
    elif isinstance(answer, str):
        if len(answer) == 1 and answer.upper() in "ABCDEFGH":
            idx = ord(answer.upper()) - ord("A")
            if 0 <= idx < len(options):
                correct_letter = answer.upper()
                correct_text = options[idx]
            else:
                return None
        else:
            correct_letter = "?"
            correct_text = answer
            for i, opt in enumerate(options):
                if opt.strip().lower() == answer.strip().lower():
                    correct_letter = _option_letter(i)
                    correct_text = opt
                    break
    else:
        return None

    # Build user message
    user_parts = []
    if context:
        user_parts.append(f"Context:\n{context.strip()}")
    user_parts.append(f"Question:\n{question.strip()}")
    user_parts.append(f"Options:\n{options_text}")
    user_msg = "\n\n".join(user_parts)

    # Build assistant message with reasoning
    assistant_msg = (
        f"Let me work through this step by step.\n\n"
        f"The question asks: {question.strip()}\n\n"
        f"Analyzing the options against the given information, "
        f"the correct answer is {correct_letter}. {correct_text}\n\n"
        f"This follows because the context and logical structure of the "
        f"question support option {correct_letter} as the most valid conclusion."
    )

    return {"role_user": user_msg, "role_assistant": assistant_msg}


def normalize_reclor(item: dict) -> dict | None:
    """Normalize ReClor (reading comprehension + logical reasoning).

    Expected fields: context, question, answers (list[str]), label (int).
    """
    context = item.get("context") or item.get("passage") or ""
    question = item.get("question") or ""
    answers = item.get("answers") or item.get("options") or item.get("choices") or []
    label = item.get("label") if item.get("label") is not None else item.get("answer")

    if not context or not question or not answers:
        return None
    if label is None:
        return None

    # Build option block
    option_lines = []
    for i, ans in enumerate(answers):
        option_lines.append(f"{_option_letter(i)}. {ans}")
    options_text = "\n".join(option_lines)

    # Determine correct answer
    if isinstance(label, int) and 0 <= label < len(answers):
        correct_letter = _option_letter(label)
        correct_text = answers[label]
    elif isinstance(label, str) and len(label) == 1 and label.upper() in "ABCDEFGH":
        idx = ord(label.upper()) - ord("A")
        if 0 <= idx < len(answers):
            correct_letter = label.upper()
            correct_text = answers[idx]
        else:
            return None
    else:
        return None

    user_msg = (
        f"Read the following passage and answer the question.\n\n"
        f"Passage:\n{context.strip()}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Options:\n{options_text}"
    )

    assistant_msg = (
        f"Let me reason through this carefully.\n\n"
        f"The passage states key information that is relevant to the question. "
        f"Considering the logical implications:\n\n"
        f"The correct answer is {correct_letter}. {correct_text}\n\n"
        f"This answer is supported by the reasoning presented in the passage, "
        f"while the other options either contradict the passage, introduce "
        f"unsupported assumptions, or fail to address the question logically."
    )

    return {"role_user": user_msg, "role_assistant": assistant_msg}


def normalize_folio(item: dict) -> dict | None:
    """Normalize FOLIO (first-order logic).

    Expected fields: premises (str or list), conclusion (str), label (str).
    Label is typically 'True', 'False', or 'Unknown'/'Uncertain'.
    """
    premises = item.get("premises") or item.get("premise") or ""
    conclusion = item.get("conclusion") or ""
    label = item.get("label") or ""

    if not premises or not conclusion or not label:
        return None

    # Premises can be a list or a single string
    if isinstance(premises, list):
        premises_text = "\n".join(f"- {p.strip()}" for p in premises if p.strip())
    else:
        # Single string -- may have multiple sentences
        premises_text = premises.strip()

    # Normalize label
    label_str = str(label).strip().lower()
    if label_str in ("true", "1", "yes", "entailment"):
        verdict = "True"
        reasoning_hint = "the premises logically entail the conclusion"
    elif label_str in ("false", "0", "no", "contradiction"):
        verdict = "False"
        reasoning_hint = "the premises contradict the conclusion"
    elif label_str in ("unknown", "uncertain", "2", "neutral"):
        verdict = "Unknown"
        reasoning_hint = (
            "the premises neither confirm nor contradict the conclusion -- "
            "there is insufficient information to determine its truth value"
        )
    else:
        # Accept the label as-is if it looks reasonable
        verdict = str(label).strip()
        reasoning_hint = "the logical relationship supports this conclusion"

    user_msg = (
        f"Given the following premises, determine whether the conclusion "
        f"logically follows. Answer True, False, or Unknown.\n\n"
        f"Premises:\n{premises_text}\n\n"
        f"Conclusion:\n{conclusion.strip()}"
    )

    assistant_msg = (
        f"Let me evaluate this step by step.\n\n"
        f"First, I'll identify the key facts from the premises and then check "
        f"whether the conclusion can be derived from them.\n\n"
        f"After careful analysis, the answer is: {verdict}\n\n"
        f"This is because {reasoning_hint}."
    )

    return {"role_user": user_msg, "role_assistant": assistant_msg}


def normalize_v1_blockchain(item: dict) -> dict | None:
    """Normalize V1 blockchain/finance data (multiple source formats).

    V1 raw files use several formats:
      - Alpaca-style: instruction, input, output
      - BAAI conversations: conversations[{from, value}]
      - Sujet-Finance: inputs, answer (or instruction/output)
      - Crypto BTC: instruction, output (or question/answer)
    """
    # -- Try Alpaca / simple instruction-output first --
    instruction = item.get("instruction") or ""
    inp = item.get("input") or ""
    output = item.get("output") or item.get("response") or ""

    if instruction and output:
        user_text = instruction.strip()
        if inp and inp.strip() and inp.strip() != instruction.strip():
            user_text = f"{instruction.strip()}\n\n{inp.strip()}"
        return {"role_user": user_text, "role_assistant": output.strip()}

    # -- Try BAAI conversations format --
    conversations = item.get("conversations")
    if conversations and isinstance(conversations, list) and len(conversations) >= 2:
        human_msg = ""
        assistant_msg = ""
        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "")
            if role == "human" and not human_msg:
                human_msg = value.strip()
            elif role in ("gpt", "assistant") and not assistant_msg:
                assistant_msg = value.strip()
        if human_msg and assistant_msg:
            return {"role_user": human_msg, "role_assistant": assistant_msg}

    # -- Try Sujet-Finance inputs/answer format --
    inputs_text = item.get("inputs") or ""
    answer_text = item.get("answer") or ""
    if inputs_text and answer_text:
        return {"role_user": inputs_text.strip(), "role_assistant": answer_text.strip()}

    # -- Try question/answer fallback --
    question = item.get("question") or item.get("query") or ""
    answer = item.get("answer") or item.get("gemini_answer") or ""
    if question and answer:
        return {"role_user": question.strip(), "role_assistant": answer.strip()}

    return None


def normalize_truthfulqa(item: dict) -> dict | None:
    """Normalize TruthfulQA generation split.

    V2 raw file (truthfulqa_gen.jsonl) expected fields:
      question, best_answer (str), correct_answers (list), incorrect_answers (list)
    V1 raw file (truthfulqa.jsonl) expected fields:
      question, mc1_targets / mc2_targets
    """
    question = item.get("question") or ""
    if not question:
        return None

    # Try V2 format first (generation split)
    best_answer = item.get("best_answer") or ""
    if best_answer:
        return {"role_user": question.strip(), "role_assistant": best_answer.strip()}

    # Try V2 correct_answers list
    correct_answers = item.get("correct_answers")
    if correct_answers and isinstance(correct_answers, list) and correct_answers:
        return {"role_user": question.strip(), "role_assistant": correct_answers[0].strip()}

    # Fall back to V1 mc1_targets format
    mc1 = item.get("mc1_targets")
    if mc1 and isinstance(mc1, dict):
        choices = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        for choice, lbl in zip(choices, labels):
            if lbl == 1:
                return {"role_user": question.strip(), "role_assistant": choice.strip()}

    mc2 = item.get("mc2_targets")
    if mc2 and isinstance(mc2, dict):
        choices = mc2.get("choices", [])
        labels = mc2.get("labels", [])
        for choice, lbl in zip(choices, labels):
            if lbl == 1:
                return {"role_user": question.strip(), "role_assistant": choice.strip()}

    return None


def normalize_slimorca(item: dict) -> dict | None:
    """Normalize SlimOrca format.

    Expected fields: conversations (list of {from, value}).
    """
    conversations = item.get("conversations")
    if not conversations or not isinstance(conversations, list):
        return None

    human_msg = ""
    assistant_msg = ""
    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "human" and not human_msg:
            human_msg = value.strip()
        elif role in ("gpt", "assistant") and not assistant_msg:
            assistant_msg = value.strip()

    if human_msg and assistant_msg:
        return {"role_user": human_msg, "role_assistant": assistant_msg}
    return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning empty list on failure."""
    items = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def load_and_normalize(
    name: str,
    path: Path,
    normalizer,
    target_count: int,
    system_prompt: str,
) -> list[dict]:
    """Load a raw dataset, normalize, sample, and wrap in messages format."""
    if not path.exists():
        print(f"  [WARN] {name}: file not found at {path} -- skipping")
        return []

    raw_items = _load_jsonl(path)
    if not raw_items:
        print(f"  [WARN] {name}: file is empty or unparseable -- skipping")
        return []

    # Normalize
    normalized = []
    for raw in raw_items:
        result = normalizer(raw)
        if result is not None:
            normalized.append(result)

    if not normalized:
        print(f"  [WARN] {name}: 0 items survived normalization -- skipping")
        return []

    # Sample down to target (or keep all if fewer)
    if len(normalized) > target_count:
        normalized = random.sample(normalized, target_count)

    # Wrap in messages format
    examples = []
    for item in normalized:
        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["role_user"]},
                {"role": "assistant", "content": item["role_assistant"]},
            ]
        })

    print(f"  [OK] {name}: {len(examples)} examples "
          f"(from {len(raw_items)} raw, {len(normalized)} valid, target {target_count})")
    return examples


def load_v1_blockchain(target_count: int) -> list[dict]:
    """Load V1 blockchain/finance domain data from multiple source files.

    We look at V1 raw sources that are finance/blockchain-relevant,
    combine them, sample to target_count, and apply the logic system prompt.
    """
    v1_sources = [
        "finance_instruct",
        "baai_finance",
        "finance_alpaca",
        "crypto_btc",
        "sentiment_tasks",
    ]

    all_items: list[dict] = []
    for src in v1_sources:
        path = V1_RAW_DIR / f"{src}.jsonl"
        if not path.exists():
            print(f"  [WARN] v1_blockchain/{src}: not found -- skipping")
            continue

        raw_items = _load_jsonl(path)
        count_before = len(all_items)
        for raw in raw_items:
            result = normalize_v1_blockchain(raw)
            if result is not None:
                all_items.append(result)

        added = len(all_items) - count_before
        print(f"    v1/{src}: {added} usable items")

    if not all_items:
        print(f"  [WARN] v1_blockchain: no items loaded from any source -- skipping")
        return []

    # Sample to target
    if len(all_items) > target_count:
        all_items = random.sample(all_items, target_count)

    # Wrap with logic system prompt (domain data gets logic-aware prompt)
    examples = []
    for item in all_items:
        examples.append({
            "messages": [
                {"role": "system", "content": LOGIC_SYSTEM_PROMPT},
                {"role": "user", "content": item["role_user"]},
                {"role": "assistant", "content": item["role_assistant"]},
            ]
        })

    print(f"  [OK] v1_blockchain: {len(examples)} examples (target {target_count})")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    marker = OUT_DIR / ".prep_complete"
    if marker.exists():
        print("[SKIP] V2 Logic SFT data already prepared")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    print("=" * 60)
    print("  V2 LOGIC SFT DATA PREPARATION")
    print("=" * 60)

    all_examples: list[dict] = []

    # ---- 1. LogiQA 2.0 ----
    print("\n[1/6] LogiQA 2.0")
    logiqa_examples = load_and_normalize(
        name="logiqa",
        path=V2_RAW_DIR / "logiqa.jsonl",
        normalizer=normalize_logiqa,
        target_count=SOURCE_TARGETS["logiqa"],
        system_prompt=LOGIC_SYSTEM_PROMPT,
    )
    all_examples.extend(logiqa_examples)

    # ---- 2. ReClor ----
    print("\n[2/6] ReClor")
    reclor_examples = load_and_normalize(
        name="reclor",
        path=V2_RAW_DIR / "reclor.jsonl",
        normalizer=normalize_reclor,
        target_count=SOURCE_TARGETS["reclor"],
        system_prompt=LOGIC_SYSTEM_PROMPT,
    )
    all_examples.extend(reclor_examples)

    # ---- 3. FOLIO ----
    print("\n[3/6] FOLIO")
    folio_examples = load_and_normalize(
        name="folio",
        path=V2_RAW_DIR / "folio.jsonl",
        normalizer=normalize_folio,
        target_count=SOURCE_TARGETS["folio"],
        system_prompt=LOGIC_SYSTEM_PROMPT,
    )
    all_examples.extend(folio_examples)

    # ---- 4. V1 blockchain domain (logic format) ----
    print("\n[4/6] V1 blockchain domain")
    blockchain_examples = load_v1_blockchain(
        target_count=SOURCE_TARGETS["v1_blockchain"],
    )
    all_examples.extend(blockchain_examples)

    # ---- 5. TruthfulQA (forgetting prevention) ----
    print("\n[5/6] TruthfulQA")
    # Try V2 raw first, fall back to V1 raw
    truthfulqa_path = V2_RAW_DIR / "truthfulqa_gen.jsonl"
    if not truthfulqa_path.exists():
        truthfulqa_path = V1_RAW_DIR / "truthfulqa.jsonl"
    truthfulqa_examples = load_and_normalize(
        name="truthfulqa",
        path=truthfulqa_path,
        normalizer=normalize_truthfulqa,
        target_count=SOURCE_TARGETS["truthfulqa_gen"],
        system_prompt=REGULAR_SYSTEM_PROMPT,
    )
    all_examples.extend(truthfulqa_examples)

    # ---- 6. General QA (SlimOrca) ----
    print("\n[6/6] SlimOrca (general QA)")
    slimorca_examples = load_and_normalize(
        name="slimorca",
        path=V2_RAW_DIR / "slimorca.jsonl",
        normalizer=normalize_slimorca,
        target_count=SOURCE_TARGETS["slimorca"],
        system_prompt=REGULAR_SYSTEM_PROMPT,
    )
    all_examples.extend(slimorca_examples)

    # ---- Summary before split ----
    print(f"\n{'=' * 60}")
    print(f"  Total examples collected: {len(all_examples)}")

    if not all_examples:
        print("[ERROR] No examples loaded from any source! Cannot proceed.")
        return

    # ---- Shuffle ----
    random.seed(42)
    random.shuffle(all_examples)

    # ---- Split train / eval (5%) ----
    eval_size = max(1, int(len(all_examples) * EVAL_RATIO))
    eval_data = all_examples[:eval_size]
    train_data = all_examples[eval_size:]

    # ---- Save ----
    for path, data in [
        (OUT_DIR / "train.jsonl", train_data),
        (OUT_DIR / "eval.jsonl", eval_data),
    ]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Train: {len(train_data):,} examples -> {OUT_DIR / 'train.jsonl'}")
    print(f"  Eval:  {len(eval_data):,} examples -> {OUT_DIR / 'eval.jsonl'}")
    print(f"  Total: {len(all_examples):,} examples")

    # ---- Source breakdown ----
    print(f"\n  Source breakdown:")
    print(f"    LogiQA:       {len(logiqa_examples):>5}")
    print(f"    ReClor:       {len(reclor_examples):>5}")
    print(f"    FOLIO:        {len(folio_examples):>5}")
    print(f"    V1 Blockchain:{len(blockchain_examples):>5}")
    print(f"    TruthfulQA:   {len(truthfulqa_examples):>5}")
    print(f"    SlimOrca:     {len(slimorca_examples):>5}")

    # ---- Marker ----
    marker.touch()
    print(f"\n[DONE] V2 Logic SFT data prepared")


if __name__ == "__main__":
    main()
