#!/usr/bin/env python3
"""Prepare DPO preference pairs from the merged SFT model's own outputs.

V2 approach: Generate candidate responses from the SFT model itself,
score them on factual quality, truthfulness, and response quality,
then select best/worst as chosen/rejected pairs.

Usage:
    python v2/scripts/prepare_dpo_data.py --pipeline baseline --model-path v2/output/baseline_sft/merged_model
    python v2/scripts/prepare_dpo_data.py --pipeline logic --model-path v2/output/logic_sft/merged_model

Outputs:
    v2/data/processed/{pipeline}_dpo/train.jsonl
    v2/data/processed/{pipeline}_dpo/eval.jsonl
    v2/data/processed/{pipeline}_dpo/.prep_complete
"""
import argparse
import gc
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/home/whiteowl/last_shot/v2")
V1_RAW_DIR = Path("/home/whiteowl/last_shot/data/raw")
V2_RAW_DIR = BASE_DIR / "data" / "raw"

SPLIT_RATIO = 0.10
TARGET_PAIRS_MIN = 3000
TARGET_PAIRS_MAX = 4000
NUM_CANDIDATES = 3
CHECKPOINT_INTERVAL = 500
MIN_SCORE_DIFF = 0.1

# Generation parameters
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_MAX_NEW_TOKENS = 256
DEFAULT_BATCH_SIZE = 8

# Hallucination patterns: absolute claims without hedging
HALLUCINATION_PATTERNS = [
    r"\bdefinitely\b",
    r"\bguaranteed\b",
    r"\balways\b",
    r"\bnever\b",
    r"\bimpossible\b",
    r"\bcertainly\b",
    r"\bwithout a doubt\b",
    r"\b100%\b",
    r"\babsolutely certain\b",
    r"\bno question\b",
    r"\bwill definitely\b",
    r"\bis guaranteed to\b",
    r"\bwill always\b",
    r"\bwill never\b",
]

# Hedging phrases that mitigate absolute claims
HEDGING_PATTERNS = [
    r"\bgenerally\b",
    r"\btypically\b",
    r"\busually\b",
    r"\boften\b",
    r"\btend(s)? to\b",
    r"\bin most cases\b",
    r"\bit('s| is) (likely|possible|probable)\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bprobably\b",
    r"\bapproximately\b",
    r"\baround\b",
    r"\babout\b",
    r"\bI('m| am) not (sure|certain)\b",
    r"\bI don't have (access|real-time)\b",
    r"\bas of my (last|latest)\b",
    r"\bhowever\b",
    r"\bthat said\b",
    r"\bon the other hand\b",
]

# Pattern for specific numbers without source attribution
UNSOURCED_NUMBER_PATTERN = re.compile(
    r"\b\d+(\.\d+)?(%| percent| billion| million| trillion| thousand)\b"
)
SOURCE_ATTRIBUTION_PATTERN = re.compile(
    r"(according to|source|reported|data from|as of|per |based on|study|survey|research)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Prompt loading helpers
# ---------------------------------------------------------------------------

def load_blockchain_web3_prompts() -> list[dict]:
    """Load blockchain/Web3 domain questions from V1 raw data.

    Scans finance_instruct, crypto_btc, finance_alpaca, dlt_tweets,
    dlt_sentiment_news, and baai_finance for relevant prompts.

    Returns list of {"prompt": str, "reference": str_or_None}.
    """
    prompts = []

    # Keywords to identify blockchain/Web3/crypto content
    crypto_keywords = [
        "blockchain", "crypto", "bitcoin", "ethereum", "defi",
        "smart contract", "web3", "token", "nft", "mining",
        "staking", "wallet", "decentralized", "consensus",
        "proof of", "solidity", "dao", "dapp", "ledger",
        "yield farming", "liquidity pool", "gas fee",
        "altcoin", "ico", "airdrop", "layer 2", "sidechain",
        "cross-chain", "oracle", "amm", "dex", "cefi",
        "metaverse", "play-to-earn", "validator", "node",
        "hash rate", "block reward", "halving", "fork",
    ]
    keyword_pattern = re.compile(
        "|".join(re.escape(k) for k in crypto_keywords), re.IGNORECASE
    )

    # -- finance_instruct.jsonl --
    fi_path = V1_RAW_DIR / "finance_instruct.jsonl"
    if fi_path.exists():
        with open(fi_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Use user_prompt field if available, else inputs
                user_q = item.get("user_prompt", "") or item.get("inputs", "")
                answer = item.get("answer")
                if user_q and keyword_pattern.search(user_q):
                    # Clean up the user prompt (strip "Question:\n" prefix if present)
                    user_q = re.sub(r"^Question:\s*", "", user_q.strip())
                    if len(user_q) > 20:
                        prompts.append({"prompt": user_q, "reference": answer})

    # -- finance_alpaca.jsonl --
    fa_path = V1_RAW_DIR / "finance_alpaca.jsonl"
    if fa_path.exists():
        with open(fa_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instruction = item.get("instruction", "")
                inp = item.get("input", "")
                output = item.get("output")
                prompt_text = instruction
                if inp:
                    prompt_text = f"{instruction}\n{inp}"
                if prompt_text and keyword_pattern.search(prompt_text):
                    if len(prompt_text) > 20:
                        prompts.append({"prompt": prompt_text.strip(), "reference": output})

    # -- baai_finance.jsonl (English entries with crypto content) --
    baai_path = V1_RAW_DIR / "baai_finance.jsonl"
    if baai_path.exists():
        with open(baai_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("lang") != "en":
                    continue
                convs = item.get("conversations", [])
                if not convs:
                    continue
                human_msg = next(
                    (c["value"] for c in convs if c.get("from") == "human"), None
                )
                gpt_msg = next(
                    (c["value"] for c in convs if c.get("from") == "gpt"), None
                )
                if human_msg and keyword_pattern.search(human_msg):
                    if len(human_msg) > 20:
                        prompts.append({"prompt": human_msg.strip(), "reference": gpt_msg})

    # -- Also pick general finance questions that are useful for DPO --
    # (investment, market, trading questions even without crypto keywords)
    finance_keywords = [
        "invest", "stock", "bond", "portfolio", "market cap",
        "interest rate", "inflation", "recession", "bull market",
        "bear market", "hedge", "risk", "dividend", "ipo",
        "mutual fund", "etf", "forex", "commodity", "financial",
    ]
    finance_pattern = re.compile(
        "|".join(re.escape(k) for k in finance_keywords), re.IGNORECASE
    )

    if fa_path.exists():
        with open(fa_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instruction = item.get("instruction", "")
                inp = item.get("input", "")
                output = item.get("output")
                prompt_text = instruction
                if inp:
                    prompt_text = f"{instruction}\n{inp}"
                if prompt_text and finance_pattern.search(prompt_text):
                    if len(prompt_text) > 20 and not keyword_pattern.search(prompt_text):
                        prompts.append({"prompt": prompt_text.strip(), "reference": output})

    if fi_path.exists():
        with open(fi_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user_q = item.get("user_prompt", "") or item.get("inputs", "")
                answer = item.get("answer")
                task_type = item.get("task_type", "")
                # Only QA tasks for general finance
                if task_type == "qa" and user_q and finance_pattern.search(user_q):
                    user_q = re.sub(r"^Question:\s*", "", user_q.strip())
                    if len(user_q) > 20 and not keyword_pattern.search(user_q):
                        prompts.append({"prompt": user_q, "reference": answer})

    print(f"  [DATA] Loaded {len(prompts)} blockchain/Web3/finance prompts from V1 raw data")
    return prompts


def load_truthfulqa_prompts() -> list[dict]:
    """Load TruthfulQA questions.

    Returns list of {"prompt": str, "reference": str_or_None}.
    """
    prompts = []

    # Try V2 raw dir first, then V1
    tqa_path = V2_RAW_DIR / "truthfulqa_gen.jsonl"
    if not tqa_path.exists():
        tqa_path = V1_RAW_DIR / "truthfulqa.jsonl"

    if not tqa_path.exists():
        print(f"  [WARN] TruthfulQA data not found at {tqa_path}")
        return prompts

    with open(tqa_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = item.get("question", "")
            # Extract reference answer from mc1_targets if available
            reference = None
            mc1 = item.get("mc1_targets")
            if mc1 and isinstance(mc1, dict):
                choices = mc1.get("choices", [])
                labels = mc1.get("labels", [])
                for choice, label in zip(choices, labels):
                    if label == 1:
                        reference = choice
                        break
            # Also check for best_answer field
            if not reference:
                reference = item.get("best_answer")

            if question and len(question) > 10:
                prompts.append({"prompt": question.strip(), "reference": reference})

    print(f"  [DATA] Loaded {len(prompts)} TruthfulQA prompts")
    return prompts


def load_slimorca_prompts() -> list[dict]:
    """Load SlimOrca general knowledge questions.

    Returns list of {"prompt": str, "reference": str_or_None}.
    """
    prompts = []

    # Try V2 raw dir first, then V1 alpaca_general as fallback
    so_path = V2_RAW_DIR / "slimorca.jsonl"
    fallback_path = V1_RAW_DIR / "alpaca_general.jsonl"

    target_path = so_path if so_path.exists() else fallback_path

    if not target_path.exists():
        print(f"  [WARN] SlimOrca/general data not found")
        return prompts

    with open(target_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # SlimOrca format: conversations list
            if "conversations" in item:
                convs = item["conversations"]
                human_msg = None
                gpt_msg = None
                for c in convs:
                    role = c.get("from", c.get("role", ""))
                    value = c.get("value", c.get("content", ""))
                    if role in ("human", "user"):
                        human_msg = value
                    elif role in ("gpt", "assistant"):
                        gpt_msg = value
                if human_msg and len(human_msg) > 20:
                    prompts.append({"prompt": human_msg.strip(), "reference": gpt_msg})

            # Alpaca format: instruction + input
            elif "instruction" in item:
                instruction = item.get("instruction", "")
                inp = item.get("input", "")
                output = item.get("output")
                prompt_text = instruction
                if inp:
                    prompt_text = f"{instruction}\n{inp}"
                if prompt_text and len(prompt_text) > 20:
                    prompts.append({"prompt": prompt_text.strip(), "reference": output})

    print(f"  [DATA] Loaded {len(prompts)} general knowledge prompts")
    return prompts


def build_prompt_pool(target_total: int) -> list[dict]:
    """Build a prompt pool with the specified distribution:
    60% blockchain/Web3, 20% TruthfulQA, 20% general knowledge.

    Each item: {"prompt": str, "reference": str_or_None, "source": str}
    """
    blockchain_prompts = load_blockchain_web3_prompts()
    truthfulqa_prompts = load_truthfulqa_prompts()
    general_prompts = load_slimorca_prompts()

    # Calculate target counts per source
    n_blockchain = int(target_total * 0.60)
    n_truthfulqa = int(target_total * 0.20)
    n_general = int(target_total * 0.20)

    # Deduplicate prompts within each source
    seen_prompts = set()

    def dedup(items):
        deduped = []
        for item in items:
            key = item["prompt"].strip().lower()[:200]
            if key not in seen_prompts:
                seen_prompts.add(key)
                deduped.append(item)
        return deduped

    blockchain_prompts = dedup(blockchain_prompts)
    truthfulqa_prompts = dedup(truthfulqa_prompts)
    general_prompts = dedup(general_prompts)

    # Sample or use all available
    random.shuffle(blockchain_prompts)
    random.shuffle(truthfulqa_prompts)
    random.shuffle(general_prompts)

    selected_blockchain = blockchain_prompts[:n_blockchain]
    selected_truthfulqa = truthfulqa_prompts[:n_truthfulqa]
    selected_general = general_prompts[:n_general]

    # Tag sources
    for item in selected_blockchain:
        item["source"] = "blockchain"
    for item in selected_truthfulqa:
        item["source"] = "truthfulqa"
    for item in selected_general:
        item["source"] = "general"

    pool = selected_blockchain + selected_truthfulqa + selected_general
    random.shuffle(pool)

    print(f"\n  [POOL] Built prompt pool: {len(pool)} total")
    print(f"         Blockchain/Web3:  {len(selected_blockchain)} ({len(selected_blockchain)/max(len(pool),1)*100:.1f}%)")
    print(f"         TruthfulQA:       {len(selected_truthfulqa)} ({len(selected_truthfulqa)/max(len(pool),1)*100:.1f}%)")
    print(f"         General:          {len(selected_general)} ({len(selected_general)/max(len(pool),1)*100:.1f}%)")

    if len(pool) < target_total:
        print(f"  [WARN] Only {len(pool)} prompts available (target: {target_total})")
        print(f"         Will use all available prompts.")

    return pool


# ---------------------------------------------------------------------------
# Model loading and generation
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str):
    """Load the merged SFT model and tokenizer."""
    print(f"\n  [MODEL] Loading model from {model_path}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    elapsed = time.time() - start
    print(f"  [MODEL] Loaded in {elapsed:.1f}s")
    return model, tokenizer


def format_prompt_for_generation(prompt: str, tokenizer) -> str:
    """Format a prompt for generation using the model's chat template if available."""
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted
    except Exception:
        # Fallback: simple prompt format
        return f"### User:\n{prompt}\n\n### Assistant:\n"


def generate_candidates_batch(
    prompts: list[str],
    model,
    tokenizer,
    num_candidates: int = NUM_CANDIDATES,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[str]]:
    """Generate candidate responses for a batch of prompts.

    Returns: list of lists, where each inner list contains num_candidates responses.
    Handles OOM by reducing batch size and retrying.
    """
    all_results = [[] for _ in range(len(prompts))]
    current_batch_size = min(batch_size, len(prompts))

    # Format prompts for generation
    formatted_prompts = [
        format_prompt_for_generation(p, tokenizer) for p in prompts
    ]

    for candidate_idx in range(num_candidates):
        i = 0
        while i < len(formatted_prompts):
            batch_end = min(i + current_batch_size, len(formatted_prompts))
            batch_prompts = formatted_prompts[i:batch_end]

            inputs = None
            try:
                # Tokenize with left-padding for batch generation
                tokenizer.padding_side = "left"
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=768,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=GEN_MAX_NEW_TOKENS,
                        temperature=GEN_TEMPERATURE,
                        top_p=GEN_TOP_P,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                # Decode only the generated tokens (skip input)
                for j, (output, input_ids) in enumerate(
                    zip(outputs, inputs["input_ids"])
                ):
                    input_len = input_ids.shape[0]
                    generated_ids = output[input_len:]
                    response = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()
                    all_results[i + j].append(response)

                # Free GPU memory
                del inputs, outputs
                inputs = None
                torch.cuda.empty_cache()

                i = batch_end

            except torch.cuda.OutOfMemoryError:
                # OOM: reduce batch size and retry this batch
                if inputs is not None:
                    del inputs
                    inputs = None
                torch.cuda.empty_cache()
                gc.collect()

                current_batch_size = max(1, current_batch_size // 2)
                print(
                    f"  [OOM] Reducing batch size to {current_batch_size}, retrying..."
                )
                # Don't advance i; retry with smaller batch

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if inputs is not None:
                        del inputs
                        inputs = None
                    torch.cuda.empty_cache()
                    gc.collect()
                    current_batch_size = max(1, current_batch_size // 2)
                    print(
                        f"  [OOM] Reducing batch size to {current_batch_size}, retrying..."
                    )
                else:
                    # For other runtime errors, skip this batch
                    print(f"  [ERROR] Generation error: {e}")
                    for j in range(i, batch_end):
                        if len(all_results[j]) <= candidate_idx:
                            all_results[j].append("")
                    i = batch_end

    return all_results


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def compute_token_f1(candidate: str, reference: str) -> float:
    """Compute token-level F1 overlap between candidate and reference."""
    if not reference or not candidate:
        return 0.0

    # Tokenize by splitting on whitespace and punctuation
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return 0.0

    cand_counter = Counter(cand_tokens)
    ref_counter = Counter(ref_tokens)

    # Count overlapping tokens
    overlap = 0
    for token, count in cand_counter.items():
        overlap += min(count, ref_counter.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return min(f1, 1.0)


def score_factual_quality(candidate: str, reference: Optional[str]) -> float:
    """Score factual quality (0-1).

    If reference answer exists, compute token F1 overlap.
    Otherwise use length/coherence heuristic.
    """
    if reference and len(reference.strip()) > 10:
        return compute_token_f1(candidate, reference)

    # Heuristic: length and coherence
    if not candidate or len(candidate.strip()) < 10:
        return 0.1

    text = candidate.strip()

    # Check for coherent sentence structure
    has_sentences = bool(re.search(r"[.!?]", text))
    has_words = len(text.split()) >= 5

    score = 0.3  # base score
    if has_sentences:
        score += 0.2
    if has_words:
        score += 0.1

    # Reward moderate length
    word_count = len(text.split())
    if 20 <= word_count <= 100:
        score += 0.2
    elif 10 <= word_count < 20:
        score += 0.1
    elif word_count > 100:
        score += 0.15

    # Check for repetitive content (sign of low quality)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > 2:
        unique_ratio = len(set(s.lower() for s in sentences)) / len(sentences)
        if unique_ratio < 0.5:
            score -= 0.2

    return max(0.0, min(1.0, score))


def score_truthfulness(candidate: str) -> float:
    """Score truthfulness (0-1).

    Penalize known hallucination patterns:
    - Absolute claims without hedging
    - Specific numbers without source attribution
    """
    if not candidate or len(candidate.strip()) < 5:
        return 0.5

    text = candidate.strip()
    score = 1.0

    # Count hallucination pattern matches
    hallucination_count = 0
    for pattern in HALLUCINATION_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        hallucination_count += len(matches)

    # Count hedging pattern matches
    hedging_count = 0
    for pattern in HEDGING_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        hedging_count += len(matches)

    # Penalize hallucination patterns, mitigated by hedging
    if hallucination_count > 0:
        # If there's hedging present, the penalty is reduced
        effective_hallucinations = max(0, hallucination_count - hedging_count)
        penalty = effective_hallucinations * 0.15
        score -= penalty

    # Check for specific numbers without source attribution
    number_matches = UNSOURCED_NUMBER_PATTERN.findall(text)
    if len(number_matches) > 0:
        has_source = bool(SOURCE_ATTRIBUTION_PATTERN.search(text))
        if not has_source:
            # Penalize unsourced specific numbers
            score -= min(0.3, len(number_matches) * 0.1)

    return max(0.0, min(1.0, score))


def score_response_quality(candidate: str) -> float:
    """Score response quality (0-1).

    - Very short (<50 chars): 0.2
    - Short-to-moderate (50-100 chars): linear 0.2 to 0.5
    - Good range (100-500 chars): linear 0.5 to 1.0
    - Long (500-1000 chars): 1.0 (plateau)
    - Very long (>1000 chars): 0.7 (penalize verbosity)
    """
    if not candidate:
        return 0.0

    length = len(candidate.strip())

    if length < 50:
        return 0.2
    elif length < 100:
        # Linear interpolation from 0.2 to 0.5
        return 0.2 + 0.3 * (length - 50) / 50
    elif length <= 500:
        # Linear interpolation from 0.5 to 1.0
        return 0.5 + 0.5 * (length - 100) / 400
    elif length <= 1000:
        return 1.0
    else:
        return 0.7


def score_candidate(candidate: str, reference: Optional[str]) -> dict:
    """Score a candidate response on all dimensions.

    Returns dict with individual scores and combined score.
    """
    factual = score_factual_quality(candidate, reference)
    truthfulness = score_truthfulness(candidate)
    quality = score_response_quality(candidate)

    combined = 0.5 * factual + 0.25 * truthfulness + 0.25 * quality

    return {
        "factual": factual,
        "truthfulness": truthfulness,
        "quality": quality,
        "combined": combined,
    }


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    pairs: list[dict],
    processed_prompts: set,
    checkpoint_path: Path,
):
    """Save intermediate checkpoint."""
    checkpoint_data = {
        "pairs": pairs,
        "processed_prompts": list(processed_prompts),
    }
    # Write to temp file first, then rename for atomicity
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False)
    tmp_path.rename(checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> tuple[list[dict], set]:
    """Load checkpoint if it exists.

    Returns (pairs, processed_prompts).
    """
    if not checkpoint_path.exists():
        return [], set()

    try:
        with open(checkpoint_path) as f:
            data = json.load(f)
        pairs = data.get("pairs", [])
        processed = set(data.get("processed_prompts", []))
        return pairs, processed
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [WARN] Corrupt checkpoint, starting fresh: {e}")
        return [], set()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_dpo_pairs(
    prompt_pool: list[dict],
    model,
    tokenizer,
    checkpoint_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Generate DPO preference pairs from the prompt pool.

    For each prompt:
    1. Generate NUM_CANDIDATES responses
    2. Score each candidate
    3. Select best as chosen, worst as rejected
    4. Skip if score difference < MIN_SCORE_DIFF
    """
    # Load checkpoint
    pairs, processed_prompts = load_checkpoint(checkpoint_path)
    if pairs:
        print(f"  [RESUME] Loaded checkpoint: {len(pairs)} pairs, {len(processed_prompts)} prompts processed")

    # Filter out already-processed prompts
    remaining = [
        item for item in prompt_pool
        if item["prompt"][:200] not in processed_prompts
    ]
    print(f"  [GEN] {len(remaining)} prompts to process ({len(processed_prompts)} already done)")

    total_generated = len(pairs)
    total_skipped_similar = 0
    total_skipped_empty = 0

    # Process in batches
    for batch_start in range(0, len(remaining), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining))
        batch_items = remaining[batch_start:batch_end]
        batch_prompts = [item["prompt"] for item in batch_items]

        # Generate candidates for the batch
        candidates_batch = generate_candidates_batch(
            batch_prompts, model, tokenizer,
            num_candidates=NUM_CANDIDATES, batch_size=batch_size,
        )

        # Score and select pairs
        for item, candidates in zip(batch_items, candidates_batch):
            prompt_key = item["prompt"][:200]
            processed_prompts.add(prompt_key)

            # Filter out empty candidates
            valid_candidates = [c for c in candidates if c and len(c.strip()) > 0]
            if len(valid_candidates) < 2:
                total_skipped_empty += 1
                continue

            # Score each candidate
            scored = []
            for cand in valid_candidates:
                scores = score_candidate(cand, item.get("reference"))
                scored.append({"text": cand, "scores": scores})

            # Sort by combined score
            scored.sort(key=lambda x: x["scores"]["combined"], reverse=True)

            best = scored[0]
            worst = scored[-1]

            # Skip if too similar
            score_diff = best["scores"]["combined"] - worst["scores"]["combined"]
            if score_diff < MIN_SCORE_DIFF:
                total_skipped_similar += 1
                continue

            pair = {
                "prompt": item["prompt"],
                "chosen": best["text"],
                "rejected": worst["text"],
            }
            pairs.append(pair)
            total_generated += 1

        # Progress report
        prompts_done = batch_start + len(batch_items)
        if prompts_done % 100 < batch_size or batch_end == len(remaining):
            print(
                f"  [PROGRESS] {prompts_done}/{len(remaining)} prompts | "
                f"{total_generated} pairs | "
                f"{total_skipped_similar} skipped (similar) | "
                f"{total_skipped_empty} skipped (empty)"
            )

        # Checkpoint every CHECKPOINT_INTERVAL prompts
        total_processed = len(processed_prompts)
        if total_processed % CHECKPOINT_INTERVAL < batch_size:
            save_checkpoint(pairs, processed_prompts, checkpoint_path)
            print(f"  [CHECKPOINT] Saved at {total_processed} prompts processed")

    # Final checkpoint
    save_checkpoint(pairs, processed_prompts, checkpoint_path)

    print(f"\n  [DONE] Generated {len(pairs)} DPO pairs")
    print(f"         Skipped (too similar): {total_skipped_similar}")
    print(f"         Skipped (empty):       {total_skipped_empty}")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs from merged SFT model outputs."
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=["baseline", "logic"],
        help="Pipeline name (baseline or logic).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the merged SFT model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for generation (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=None,
        help=f"Target number of pairs (default: {TARGET_PAIRS_MIN}-{TARGET_PAIRS_MAX}).",
    )
    args = parser.parse_args()

    # Paths
    out_dir = BASE_DIR / "data" / "processed" / f"{args.pipeline}_dpo"
    marker = out_dir / ".prep_complete"
    train_path = out_dir / "train.jsonl"
    holdout_path = out_dir / "eval.jsonl"
    checkpoint_path = out_dir / "checkpoint.json"

    # Check marker
    if marker.exists():
        print(f"[SKIP] {args.pipeline} DPO data already prepared ({marker})")
        return

    # Resolve model path (relative to BASE_DIR if not absolute)
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = str(BASE_DIR / model_path)

    print("=" * 60)
    print(f"  V2 DPO DATA PREPARATION: {args.pipeline.upper()}")
    print("=" * 60)
    print(f"  Model:       {model_path}")
    print(f"  Output:      {out_dir}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Seed:        {args.seed}")

    random.seed(args.seed)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine target prompt count
    # We need more prompts than target pairs because some will be skipped
    target_pairs = args.target_pairs or TARGET_PAIRS_MAX
    # Overshoot by ~40% to account for skipped pairs
    target_prompts = int(target_pairs * 1.4)

    # Step 1: Build prompt pool
    print("\n--- Step 1: Loading prompt sources ---")
    prompt_pool = build_prompt_pool(target_prompts)

    if len(prompt_pool) == 0:
        print("[ERROR] No prompts found! Check data paths.")
        print(f"  V1 raw dir:  {V1_RAW_DIR} (exists: {V1_RAW_DIR.exists()})")
        print(f"  V2 raw dir:  {V2_RAW_DIR} (exists: {V2_RAW_DIR.exists()})")
        sys.exit(1)

    # Step 2: Load model
    print("\n--- Step 2: Loading SFT model ---")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(model_path)

    # Step 3: Generate DPO pairs
    print("\n--- Step 3: Generating DPO pairs ---")
    start_time = time.time()
    pairs = generate_dpo_pairs(
        prompt_pool, model, tokenizer,
        checkpoint_path=checkpoint_path,
        batch_size=args.batch_size,
    )
    gen_elapsed = time.time() - start_time

    # Free model memory
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    if len(pairs) == 0:
        print("[ERROR] No pairs generated! Check model and prompts.")
        sys.exit(1)

    # Step 4: Split into train and held-out set
    print("\n--- Step 4: Train/held-out split ---")
    random.shuffle(pairs)
    holdout_size = max(1, int(len(pairs) * SPLIT_RATIO))
    holdout_data = pairs[:holdout_size]
    train_data = pairs[holdout_size:]

    # Step 5: Save
    print("\n--- Step 5: Saving data ---")
    for path, data in [(train_path, train_data), (holdout_path, holdout_data)]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved {len(data)} pairs to {path}")

    # Mark complete
    marker.touch()

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  Cleaned up checkpoint file")

    # Summary
    print("\n" + "=" * 60)
    print(f"  DPO DATA PREPARATION COMPLETE: {args.pipeline.upper()}")
    print("=" * 60)
    print(f"  Total pairs:  {len(pairs)}")
    print(f"  Train:        {len(train_data)}")
    print(f"  Held-out:     {len(holdout_data)}")
    print(f"  Time:         {gen_elapsed:.1f}s ({gen_elapsed/60:.1f}m)")
    print(f"  Train path:   {train_path}")
    print(f"  Holdout path: {holdout_path}")
    print(f"  Marker:       {marker}")


if __name__ == "__main__":
    main()
