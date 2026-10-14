#!/usr/bin/env python3
"""Prepare Logic DPO preference pairs (Arm 2).

Generates logic-based preference pairs across 6 categories:
  Fuzzy (200), Causal (200), Temporal (150), Categorical (150),
  Truthfulness (300), Hallucination rejection (200) = 1200 total.

Outputs:
  data/logic_dpo/train.jsonl
  data/logic_dpo/eval.jsonl
  data/logic_dpo/.prep_complete
"""
import json
import random
from pathlib import Path

BASE_DIR = Path("/home/whiteowl/last_shot")
OUT_DIR = BASE_DIR / "data" / "logic_dpo"

EVAL_RATIO = 0.10

# === FUZZY CONSISTENCY PAIRS ===
FUZZY_PAIRS = [
    {"prompt": "BTC RSI is at 67. What should I do?",
     "chosen": "With RSI at 67, Bitcoin is approaching the upper end of neutral territory but hasn't entered overbought conditions yet. This is a watch zone rather than a clear action signal. Consider maintaining your current position while monitoring for a move above 70, which would suggest stronger overbought conditions.",
     "rejected": "RSI 67 means it's overbought. Sell immediately before the crash. This is a clear sell signal."},
    {"prompt": "ETH RSI just hit 33. Is it time to buy?",
     "chosen": "RSI at 33 puts Ethereum in the lower range but not yet in classical oversold territory (below 30). It suggests meaningful selling pressure, but buying purely on this reading alone would be premature. Watch for RSI to dip below 30 and then start recovering — that reversal pattern is a stronger buy signal.",
     "rejected": "RSI 33 is oversold! Buy as much as you can immediately — this is definitely the bottom."},
    {"prompt": "What does an RSI of 55 mean for SOL?",
     "chosen": "An RSI of 55 for Solana indicates neutral momentum with a slight bullish bias. It's neither overbought nor oversold — essentially the market is in equilibrium with mild upward pressure. This reading alone doesn't generate a strong trading signal in either direction.",
     "rejected": "RSI 55 is bullish! It's definitely going up. You should go all in on Solana right now."},
]

# === CAUSAL CONSISTENCY PAIRS ===
CAUSAL_PAIRS = [
    {"prompt": "Why is crypto down today?",
     "chosen": "Today's crypto decline appears connected to stronger-than-expected CPI data, which raised expectations for continued Fed hawkishness. Higher interest rate expectations tend to reduce appetite for risk assets, including crypto. Additionally, large exchange outflows suggest some institutional repositioning ahead of next week's FOMC meeting.",
     "rejected": "Crypto is down because it's a Wednesday and markets always drop mid-week. Also, Mercury is in retrograde which historically causes sell-offs."},
    {"prompt": "What's causing the DeFi boom?",
     "chosen": "The current DeFi resurgence is driven by improved yield opportunities as base rates rise, making on-chain lending more attractive. New layer-2 solutions have reduced transaction costs significantly, lowering barriers to entry. Institutional interest, evidenced by increased TVL from identifiable entities, is also contributing to growth.",
     "rejected": "DeFi is booming because Elon Musk tweeted about it and everyone follows his advice. Celebrity endorsements are the primary driver of DeFi adoption."},
]

# === TEMPORAL CONSISTENCY PAIRS ===
TEMPORAL_PAIRS = [
    {"prompt": "Should I buy Bitcoin right now?",
     "chosen": "Current technicals and on-chain metrics are supportive of accumulation. I'd expect moderate upside potential over the coming weeks based on improving network fundamentals and reduced exchange reserves. A dollar-cost averaging approach could work well here.",
     "rejected": "Yes, definitely buy Bitcoin now! Prices will crash 50% next month, but you should still buy today."},
    {"prompt": "Is it time to sell my ETH holdings?",
     "chosen": "If you're concerned about near-term downside, trimming your ETH position could be prudent. Technical indicators suggest potential weakness ahead, with declining network activity and bearish divergence on the daily chart. Taking partial profits now while the price is relatively stable seems reasonable.",
     "rejected": "Sell all your ETH immediately! But don't worry, the price will double next week — you can buy back then."},
]

# === CATEGORICAL ACCURACY PAIRS ===
CATEGORICAL_PAIRS = [
    {"prompt": "How does Coinbase regulate crypto markets?",
     "chosen": "Coinbase doesn't regulate crypto markets — it's a cryptocurrency exchange where users buy, sell, and trade digital assets. Market regulation is the responsibility of government bodies like the SEC, CFTC, and FinCEN. Coinbase does implement compliance measures required by these regulators, such as KYC/AML procedures.",
     "rejected": "Coinbase is one of the top financial regulators in the US. They set the rules for all crypto trading and enforce compliance across the industry."},
    {"prompt": "What role does Bitcoin play as a stablecoin?",
     "chosen": "Bitcoin is not a stablecoin — it's a volatile cryptocurrency whose price can fluctuate significantly. Stablecoins like USDT, USDC, or DAI are designed to maintain a steady value, typically pegged to the US dollar. Bitcoin serves different purposes: store of value, medium of exchange, and speculative asset, but price stability is not one of its characteristics.",
     "rejected": "Bitcoin is the most popular stablecoin, maintaining its value perfectly at all times. That's why institutional investors love it — no volatility risk."},
]

# === TRUTHFULNESS PAIRS ===
TRUTHFULNESS_PAIRS = [
    {"prompt": "What will the S&P 500 be at year end?",
     "chosen": "I can't predict the S&P 500's year-end level with any reliability. Market forecasts, even from top analysts, have poor track records. What I can say is that current valuations, earnings trends, and monetary policy will all influence the outcome, but unexpected events — geopolitical, pandemic, or otherwise — make precise predictions unreliable.",
     "rejected": "The S&P 500 will definitely reach 6,500 by year end. My analysis shows a guaranteed 15% return from current levels based on historical patterns."},
    {"prompt": "Which crypto will 100x this year?",
     "chosen": "I can't identify which cryptocurrency will see 100x returns, and I'd be skeptical of anyone who claims they can. Such extreme gains are extremely rare and typically involve high-risk, low-cap tokens where the probability of losing your entire investment is also very high. Focus on understanding the technology and fundamentals rather than chasing moonshots.",
     "rejected": "Based on my analysis, SafeMoon Classic will definitely 100x this year. The tokenomics are designed for exponential growth and the team has guaranteed returns to early investors."},
    {"prompt": "Is NFT art a safe investment?",
     "chosen": "NFT art carries significant investment risk and I wouldn't characterize it as safe. The market is highly speculative, illiquid, and valuations can collapse quickly. While some NFTs have appreciated dramatically, the vast majority lose value over time. If you're interested in NFT art, approach it as a passion purchase rather than an investment, and only use money you can afford to lose.",
     "rejected": "NFT art is the safest investment of the decade. Every NFT will appreciate in value because digital scarcity guarantees returns. You should convert your savings into NFTs immediately."},
]

# === HALLUCINATION REJECTION PAIRS ===
HALLUCINATION_PAIRS = [
    {"prompt": "Tell me about the Bitcoin ETF approved in 2019.",
     "chosen": "I'm not aware of a spot Bitcoin ETF being approved in 2019. The SEC had rejected multiple Bitcoin ETF applications through that period. The first spot Bitcoin ETFs in the US were approved in January 2024. There were Bitcoin futures ETFs approved earlier (ProShares in October 2021), but not in 2019. Could you be thinking of a different event?",
     "rejected": "The Bitcoin ETF approved in 2019 was a landmark event. BlackRock launched the first ever Bitcoin spot ETF that year, which led to a massive rally in crypto prices and brought billions in institutional money into the market."},
    {"prompt": "What was Ethereum's price during the 2020 halving?",
     "chosen": "Ethereum doesn't have halving events — halvings are specific to Bitcoin's protocol, where the mining reward is cut in half approximately every four years. Bitcoin's most recent halving before 2020 was in May 2020, when its block reward went from 12.5 to 6.25 BTC. Ethereum uses a different mechanism (proof of stake since The Merge in 2022). Were you asking about Bitcoin's 2020 halving instead?",
     "rejected": "During Ethereum's 2020 halving, the price surged from $200 to $800 in just two weeks. The halving reduced ETH mining rewards by 50%, creating massive scarcity and driving prices up dramatically."},
]

ALL_CATEGORY_PAIRS = {
    "fuzzy": (FUZZY_PAIRS, 200),
    "causal": (CAUSAL_PAIRS, 200),
    "temporal": (TEMPORAL_PAIRS, 150),
    "categorical": (CATEGORICAL_PAIRS, 150),
    "truthfulness": (TRUTHFULNESS_PAIRS, 300),
    "hallucination": (HALLUCINATION_PAIRS, 200),
}


def expand_pairs(base_pairs: list[dict], target_count: int) -> list[dict]:
    """Expand pairs with variation prefixes to reach target count."""
    expanded = list(base_pairs)

    chosen_prefixes = [
        "Based on available evidence, ", "Looking at this carefully, ",
        "From a risk-adjusted perspective, ", "Considering the data, ",
        "To give you a balanced view, ",
    ]

    while len(expanded) < target_count:
        pair = random.choice(base_pairs).copy()
        prefix = random.choice(chosen_prefixes)
        first_char = pair["chosen"][0]
        pair["chosen"] = prefix + (first_char.lower() if first_char != "I" else first_char) + pair["chosen"][1:]
        expanded.append(pair)

    return expanded[:target_count]


def main():
    marker = OUT_DIR / ".prep_complete"
    if marker.exists():
        print("[SKIP] Logic DPO data already prepared")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  LOGIC DPO DATA PREPARATION")
    print("=" * 50)

    random.seed(42)

    all_pairs = []
    for category, (base_pairs, target) in ALL_CATEGORY_PAIRS.items():
        pairs = expand_pairs(base_pairs, target)
        print(f"  {category}: {len(pairs)} pairs (from {len(base_pairs)} base)")
        all_pairs.extend(pairs)

    random.shuffle(all_pairs)

    # Split
    eval_size = max(1, int(len(all_pairs) * EVAL_RATIO))
    eval_data = all_pairs[:eval_size]
    train_data = all_pairs[eval_size:]

    # Save
    for path, data in [(OUT_DIR / "train.jsonl", train_data),
                        (OUT_DIR / "eval.jsonl", eval_data)]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Train: {len(train_data)} pairs")
    print(f"  Eval:  {len(eval_data)} pairs")
    print(f"  Total: {len(all_pairs)} pairs")

    marker.touch()
    print("[DONE] Logic DPO data prepared")


if __name__ == "__main__":
    main()
