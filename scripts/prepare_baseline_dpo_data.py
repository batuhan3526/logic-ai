#!/usr/bin/env python3
"""Prepare baseline DPO preference pairs.

Uses template-based pair generation for the baseline arm.
Generates ~1200 pairs with 10% eval split.

Dependency: Requires baseline SFT data (runs independently of model).
For model-generated pairs, set --use-model flag (requires merged SFT model).

Outputs:
  data/baseline_dpo/train.jsonl
  data/baseline_dpo/eval.jsonl
  data/baseline_dpo/.prep_complete
"""
import json
import random
from pathlib import Path

BASE_DIR = Path("/home/whiteowl/last_shot")
OUT_DIR = BASE_DIR / "data" / "baseline_dpo"

EVAL_RATIO = 0.10

# Template-based preference pairs for finance domain
# Each: {"prompt": str, "chosen": str, "rejected": str}
BASE_PAIRS = [
    # Factual accuracy pairs
    {"prompt": "What is the current market cap of Bitcoin?",
     "chosen": "I don't have access to real-time market data, so I can't give you the exact current market cap. As of my last update, Bitcoin's market cap fluctuates significantly. You can check CoinMarketCap or CoinGecko for the latest figure.",
     "rejected": "Bitcoin's market cap is exactly $1.2 trillion as of today, making it the world's largest cryptocurrency by a significant margin."},

    {"prompt": "What will happen to gold prices next quarter?",
     "chosen": "I can't predict gold prices with certainty. Gold is influenced by many factors including interest rates, inflation expectations, geopolitical tensions, and dollar strength. Current conditions suggest some support for gold, but unexpected events could change the outlook significantly.",
     "rejected": "Gold prices will definitely increase by 15% next quarter due to rising inflation and geopolitical instability. This is guaranteed based on historical patterns."},

    {"prompt": "Is Tesla stock a good buy right now?",
     "chosen": "Whether Tesla is a good buy depends on your investment timeline, risk tolerance, and portfolio allocation. Tesla has strong growth potential in EVs and energy but trades at a premium valuation. Consider the current P/E ratio, delivery numbers, and how it fits your overall investment strategy before deciding.",
     "rejected": "Yes, absolutely buy Tesla stock immediately. It's guaranteed to go up because Elon Musk is a genius and electric cars are the future. You can't lose money on Tesla."},

    {"prompt": "Explain how DeFi lending works.",
     "chosen": "DeFi lending works through smart contracts on blockchains like Ethereum. Lenders deposit crypto into liquidity pools and earn interest. Borrowers provide collateral (usually 150%+ of loan value) and pay interest to access funds. Protocols like Aave and Compound automate this process, eliminating traditional intermediaries. Key risks include smart contract vulnerabilities, liquidation during market crashes, and impermanent loss.",
     "rejected": "DeFi lending is basically free money. You deposit your crypto and earn huge returns with no risk. The SEC, which is the largest crypto exchange, manages all DeFi protocols to ensure safety. Just deposit everything you have and watch it grow."},

    {"prompt": "What caused the 2008 financial crisis?",
     "chosen": "The 2008 financial crisis resulted from a complex chain of factors: loose lending standards led to a housing bubble, which was amplified by mortgage-backed securities and CDOs that spread risk throughout the financial system. When housing prices fell, these instruments collapsed, triggering bank failures and a credit freeze. Inadequate regulation, excessive leverage, and misaligned incentives all contributed.",
     "rejected": "The 2008 crisis was caused by a single bank making bad loans. It was a simple problem that was easy to predict and could have been prevented by just one regulation."},

    # More pairs covering various finance topics...
    {"prompt": "Should I invest my emergency fund in crypto?",
     "chosen": "I'd strongly advise against investing your emergency fund in cryptocurrency. Emergency funds should be in liquid, stable assets like high-yield savings accounts or money market funds. Crypto is highly volatile — your emergency fund could lose 50%+ of its value during a market downturn, exactly when you might need it most.",
     "rejected": "Yes, crypto is the best place for your emergency fund! Bitcoin always goes up long term, and you can access it 24/7 unlike banks. Put everything into Bitcoin and Ethereum for maximum returns."},

    {"prompt": "How does staking work in proof-of-stake?",
     "chosen": "In proof-of-stake, validators lock up (stake) their cryptocurrency as collateral to participate in block validation. The protocol selects validators to propose and attest to blocks, typically weighted by stake amount. Validators earn rewards for honest behavior and risk losing their stake (slashing) for malicious actions. This mechanism secures the network while being far more energy-efficient than proof-of-work mining.",
     "rejected": "Staking is when you send your crypto to Binance, a well-known financial regulator, and they guarantee you 100% annual returns. There's no risk because the blockchain is unhackable and your funds are always safe."},

    {"prompt": "What's the difference between a bull and bear market?",
     "chosen": "A bull market is characterized by rising prices, typically defined as a 20%+ increase from recent lows, accompanied by investor optimism and strong economic indicators. A bear market is the opposite — a 20%+ decline from recent highs, with pessimism and weakening fundamentals. Markets cycle between these phases, and timing the transitions is notoriously difficult even for professionals.",
     "rejected": "A bull market means prices only go up forever, and a bear market means everything crashes to zero. You should buy in bull markets and sell everything in bear markets — it's that simple."},
]


def expand_pairs(base_pairs: list[dict], target_count: int) -> list[dict]:
    """Expand base pairs with variations to reach target count."""
    expanded = list(base_pairs)

    prefixes = [
        "As a financial analyst, ", "From an investment perspective, ",
        "Based on market fundamentals, ", "Looking at this objectively, ",
        "Considering current market conditions, ",
    ]

    while len(expanded) < target_count:
        pair = random.choice(base_pairs).copy()
        prefix = random.choice(prefixes)
        first_char = pair["chosen"][0]
        pair["chosen"] = prefix + (first_char.lower() if first_char != "I" else first_char) + pair["chosen"][1:]
        expanded.append(pair)

    return expanded[:target_count]


def main():
    marker = OUT_DIR / ".prep_complete"
    if marker.exists():
        print("[SKIP] Baseline DPO data already prepared")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  BASELINE DPO DATA PREPARATION")
    print("=" * 50)

    random.seed(42)

    # Expand to ~1200 pairs
    pairs = expand_pairs(BASE_PAIRS, 1200)
    random.shuffle(pairs)

    # Split
    eval_size = max(1, int(len(pairs) * EVAL_RATIO))
    eval_data = pairs[:eval_size]
    train_data = pairs[eval_size:]

    # Save
    for path, data in [(OUT_DIR / "train.jsonl", train_data),
                        (OUT_DIR / "eval.jsonl", eval_data)]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Train: {len(train_data)} pairs")
    print(f"  Eval:  {len(eval_data)} pairs")

    marker.touch()
    print("[DONE] Baseline DPO data prepared")


if __name__ == "__main__":
    main()
