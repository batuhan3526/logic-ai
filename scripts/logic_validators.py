"""
5 Logic Form Validators for the Invisible Logic project.

These validators check if a finance answer is logic-consistent.
They are used during data preparation to verify/filter training data.
They are fast (keyword/regex based, no LLM calls).
"""

import re

ENTITY_CATEGORIES = {
    "regulator": ["sec", "cftc", "fed", "federal reserve", "ecb", "fca", "finra", "occ"],
    "exchange": ["binance", "coinbase", "kraken", "bybit", "okx", "ftx", "gemini", "bitfinex"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "cardano", "ada"],
    "tradfi": ["s&p", "nasdaq", "dow jones", "goldman sachs", "jpmorgan", "blackrock"],
    "technology": ["blockchain", "smart contract", "mining", "defi", "nft", "layer 2"],
    "macro": ["inflation", "gdp", "interest rate", "unemployment", "cpi", "ppi"],
}


def validate_fuzzy(text: str) -> dict:
    """
    Check if numerical indicators (RSI, etc.) are interpreted with appropriate uncertainty.

    RSI in the range 60-80 or 20-40 should NOT use binary/definitive language like
    "definitely overbought" or "sell immediately".
    Extreme values (RSI > 80 or RSI < 20) are allowed to use binary language.
    """
    lower = text.lower()

    # Find RSI values mentioned in text
    rsi_pattern = re.compile(r'rsi\s+(?:is\s+)?(?:at\s+)?(?:of\s+)?(\d+(?:\.\d+)?)', re.IGNORECASE)
    rsi_matches = rsi_pattern.findall(text)

    if not rsi_matches:
        # No RSI value found; no fuzzy concern
        return {"pass": True, "reason": "No RSI indicator found; fuzzy check not applicable."}

    # Binary/overconfident language patterns
    binary_patterns = [
        r'\bdefinitely\b',
        r'\bclearly\s+overbought\b',
        r'\bclearly\s+oversold\b',
        r'\bsell\s+immediately\b',
        r'\bbuy\s+immediately\b',
        r'\bwithout\s+(?:a\s+)?doubt\b',
        r'\bcertain(?:ly)?\b',
        r'\babsolutely\b',
        r'\bis\s+overbought\b',
        r'\bis\s+oversold\b',
    ]

    has_binary_language = any(re.search(p, lower) for p in binary_patterns)

    if not has_binary_language:
        return {"pass": True, "reason": "RSI interpreted with appropriate uncertainty."}

    # Check each RSI value: if it's in the moderate range (20-40 or 60-80), binary language is a problem
    for val_str in rsi_matches:
        rsi_val = float(val_str)
        in_moderate_range = (60 <= rsi_val <= 80) or (20 <= rsi_val <= 40)
        if in_moderate_range:
            return {
                "pass": False,
                "reason": (
                    f"RSI value {rsi_val} is in moderate range but text uses binary/definitive language. "
                    "Moderate RSI values require nuanced interpretation."
                ),
            }

    # RSI is at extreme values (>80 or <20); binary language is acceptable
    return {"pass": True, "reason": "RSI is at extreme value; binary language acceptable."}


def validate_causal(text: str) -> dict:
    """
    Check if cause-effect claims are valid.

    Detect spurious causes like "because it was Tuesday", "because of the moon",
    "because of luck". Pass if no causal markers are present or if causes are valid.
    """
    lower = text.lower()

    # Spurious cause patterns
    spurious_patterns = [
        r'because\s+it\s+was\s+(?:a\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
        r'because\s+(?:of\s+)?(?:the\s+)?moon',
        r'because\s+(?:of\s+)?luck',
        r'because\s+(?:of\s+)?(?:good|bad)\s+luck',
        r'because\s+(?:of\s+)?karma',
        r'because\s+(?:of\s+)?(?:a\s+)?coincidence',
        r'because\s+(?:of\s+)?(?:the\s+)?stars',
        r'because\s+(?:of\s+)?astrology',
        r'because\s+(?:of\s+)?(?:the\s+)?weather',
        r'due\s+to\s+(?:the\s+)?(?:day\s+of\s+the\s+week|moon|luck|karma)',
    ]

    for pattern in spurious_patterns:
        if re.search(pattern, lower):
            return {
                "pass": False,
                "reason": f"Spurious causal claim detected matching pattern: '{pattern}'.",
            }

    return {"pass": True, "reason": "No spurious causal claims detected."}


def validate_temporal(text: str) -> dict:
    """
    Check if action recommendations match forecasts.

    BUY + only downward forecast = fail.
    SELL + only upward forecast = fail.
    Mixed signals are OK.
    """
    lower = text.lower()

    # Action signals
    buy_patterns = [
        r'\b(?:recommend|suggesting?|advise?)\s+(?:a\s+)?buying?\b',
        r'\b(?:recommend|suggesting?|advise?)\s+(?:to\s+)?buy\b',
        r'\bbuy(?:ing)?\s+(?:is\s+)?(?:recommended|advised|suggested)\b',
        r'\bi\s+(?:recommend|suggest|advise)\s+buying?\b',
        r'\bi\s+(?:recommend|suggest|advise)\s+(?:a\s+)?buy\b',
        r'\bgo(?:ing)?\s+long\b',
        r'\blong\s+position\b',
        r'\baccumulate\b',
    ]
    sell_patterns = [
        r'\b(?:recommend|suggesting?|advise?)\s+(?:a\s+)?selling?\b',
        r'\b(?:recommend|suggesting?|advise?)\s+(?:to\s+)?sell\b',
        r'\bsell(?:ing)?\s+(?:is\s+)?(?:recommended|advised|suggested)\b',
        r'\bi\s+(?:recommend|suggest|advise)\s+selling?\b',
        r'\bi\s+(?:recommend|suggest|advise)\s+(?:a\s+)?sell\b',
        r'\bgo(?:ing)?\s+short\b',
        r'\bshort\s+position\b',
        r'\bdistribute\b',
    ]

    # Forecast direction signals
    upward_patterns = [
        r'\b(?:prices?|value|market)\s+(?:will|may|could|should|is\s+likely\s+to)\s+(?:rise|increase|go\s+up|climb|surge|rally|appreciate)\b',
        r'\b(?:expect|anticipate|forecast|project)\s+(?:a\s+)?(?:rise|increase|growth|rally|upward|bullish)\b',
        r'\bbullish\s+(?:outlook|forecast|trend|momentum)\b',
        r'\bexpect\s+(?:prices?\s+to\s+)?rise\b',
        r'\bexpect\s+(?:prices?\s+to\s+)?(?:go\s+up|increase|climb)\b',
        r'\b(?:prices?|value)\s+(?:to\s+)?rise\b',
        r'\bupward\s+(?:trend|momentum|movement)\b',
    ]
    downward_patterns = [
        r'\b(?:prices?|value|market)\s+(?:will|may|could|should|is\s+likely\s+to)\s+(?:fall|decrease|go\s+down|drop|decline|sink|depreciate)\b',
        r'\b(?:expect|anticipate|forecast|project)\s+(?:a\s+)?(?:fall|decrease|decline|drop|downward|bearish)\b',
        r'\bbearish\s+(?:outlook|forecast|trend|momentum)\b',
        r'\bexpect\s+(?:prices?\s+to\s+)?(?:fall|decline|drop|decrease)\b',
        r'\b(?:prices?|value)\s+(?:will\s+)?(?:likely\s+)?decline\b',
        r'\bdownward\s+(?:trend|momentum|movement)\b',
        r'\bprices\s+will\s+likely\s+decline\b',
    ]

    has_buy = any(re.search(p, lower) for p in buy_patterns)
    has_sell = any(re.search(p, lower) for p in sell_patterns)
    has_upward = any(re.search(p, lower) for p in upward_patterns)
    has_downward = any(re.search(p, lower) for p in downward_patterns)

    # No action or no forecast: no temporal concern
    if not (has_buy or has_sell):
        return {"pass": True, "reason": "No action recommendation found; temporal check not applicable."}
    if not (has_upward or has_downward):
        return {"pass": True, "reason": "No directional forecast found; temporal check not applicable."}

    # BUY with only downward forecast
    if has_buy and has_downward and not has_upward:
        return {
            "pass": False,
            "reason": "BUY recommendation contradicts a purely downward forecast.",
        }

    # SELL with only upward forecast
    if has_sell and has_upward and not has_downward:
        return {
            "pass": False,
            "reason": "SELL recommendation contradicts a purely upward forecast.",
        }

    return {"pass": True, "reason": "Action recommendation is consistent with forecast direction."}


def validate_categorical(text: str) -> dict:
    """
    Check if entities are correctly classified.

    Detect:
    - Exchanges (Binance, Coinbase, etc.) misclassified as regulators
    - Regulators (SEC, CFTC, Fed) misclassified as exchanges
    - Crypto (Bitcoin, ETH) misclassified as stocks/bonds
    """
    lower = text.lower()

    # Misclassification patterns: entity type + wrong label context
    # Exchange misclassified as regulator
    exchange_as_regulator_patterns = []
    for exchange in ENTITY_CATEGORIES["exchange"]:
        exchange_as_regulator_patterns.append(
            re.compile(
                r'\b' + re.escape(exchange) + r'\b'
                r'.{0,60}'
                r'(?:regulator|regulatory\s+(?:body|authority)|watchdog|oversight\s+agency|government\s+agency)',
                re.IGNORECASE,
            )
        )
        exchange_as_regulator_patterns.append(
            re.compile(
                r'(?:regulator|regulatory\s+(?:body|authority)|watchdog|oversight\s+agency|government\s+agency)'
                r'.{0,60}'
                r'\b' + re.escape(exchange) + r'\b',
                re.IGNORECASE,
            )
        )

    # Regulator misclassified as exchange
    regulator_as_exchange_patterns = []
    for regulator in ENTITY_CATEGORIES["regulator"]:
        regulator_as_exchange_patterns.append(
            re.compile(
                r'\b' + re.escape(regulator) + r'\b'
                r'.{0,60}'
                r'(?:exchange|trading\s+platform|marketplace|market\s+place)',
                re.IGNORECASE,
            )
        )
        regulator_as_exchange_patterns.append(
            re.compile(
                r'(?:exchange|trading\s+platform|marketplace)'
                r'.{0,60}'
                r'\b' + re.escape(regulator) + r'\b',
                re.IGNORECASE,
            )
        )

    # Crypto misclassified as stock or bond
    crypto_as_stock_patterns = []
    for crypto in ENTITY_CATEGORIES["crypto"]:
        crypto_as_stock_patterns.append(
            re.compile(
                r'\b' + re.escape(crypto) + r'\b'
                r'.{0,60}'
                r'(?:stock|equity|share|bond|fixed[\s-]income|security|securities)',
                re.IGNORECASE,
            )
        )
        crypto_as_stock_patterns.append(
            re.compile(
                r'(?:stock|equity|share|bond|fixed[\s-]income)'
                r'.{0,60}'
                r'\b' + re.escape(crypto) + r'\b',
                re.IGNORECASE,
            )
        )

    # Check exchange-as-regulator
    for pattern in exchange_as_regulator_patterns:
        if pattern.search(text):
            return {
                "pass": False,
                "reason": "A cryptocurrency exchange appears to be misclassified as a regulator.",
            }

    # Check regulator-as-exchange
    for pattern in regulator_as_exchange_patterns:
        if pattern.search(text):
            return {
                "pass": False,
                "reason": "A financial regulator appears to be misclassified as an exchange.",
            }

    # Check crypto-as-stock/bond
    for pattern in crypto_as_stock_patterns:
        if pattern.search(text):
            return {
                "pass": False,
                "reason": "A cryptocurrency appears to be misclassified as a stock or bond.",
            }

    return {"pass": True, "reason": "Entity classifications appear correct."}


def validate_fol(text: str) -> dict:
    """
    Check formal trading rules using First-Order Logic style checks.

    - RSI > 70 + BUY must acknowledge risk
    - RSI < 30 + SELL must acknowledge oversold
    - Overconfident claims ("guaranteed", "100% certain") = fail
    """
    lower = text.lower()

    # Overconfidence patterns — always fail
    overconfident_patterns = [
        r'\bguaranteed?\b',
        r'\b100\s*%\s*(?:certain|sure|confident|accurate|reliable)\b',
        r'\bno\s+(?:risk|doubt|uncertainty)\b',
        r'\bcannot\s+(?:fail|lose|go\s+wrong)\b',
        r'\bwill\s+definitely\s+(?:rise|fall|increase|decrease|go\s+up|go\s+down)\b',
        r'\bwith\s+(?:absolute|complete|total)\s+certainty\b',
        r'\bhigh\s+confidence\b',
    ]

    for pattern in overconfident_patterns:
        if re.search(pattern, lower):
            return {
                "pass": False,
                "reason": f"Overconfident claim detected matching pattern: '{pattern}'.",
            }

    # Find RSI values
    rsi_pattern = re.compile(r'rsi\s+(?:is\s+)?(?:at\s+)?(?:of\s+)?(\d+(?:\.\d+)?)', re.IGNORECASE)
    rsi_matches = rsi_pattern.findall(text)

    if not rsi_matches:
        return {"pass": True, "reason": "No RSI-based trading rules detected; FOL check passes."}

    # Risk acknowledgment keywords
    risk_keywords = [
        r'\brisk\b',
        r'\bcaution(?:ary)?\b',
        r'\bcareful\b',
        r'\bmonitor\b',
        r'\bwatch\s+(?:closely|carefully)\b',
        r'\bpotential\s+(?:downside|reversal|correction)\b',
        r'\belevated\s+risk\b',
        r'\bvolatile?\b',
        r'\buncertain(?:ty)?\b',
        r'\boverextended\b',
        r'\bapproaching\s+overbought\b',
        r'\bnear\s+overbought\b',
    ]
    # Oversold acknowledgment keywords
    oversold_keywords = [
        r'\boversold\b',
        r'\bbounce\b',
        r'\brecovery\b',
        r'\brebound\b',
        r'\bpotential\s+reversal\b',
    ]

    # Buy signal patterns
    buy_patterns = [
        r'\b(?:buy|long|accumulate|purchase)\b',
        r'\brecommend\s+(?:buying?|a\s+buy)\b',
        r'\bgreat\s+time\s+to\s+buy\b',
    ]
    sell_patterns = [
        r'\b(?:sell|short|distribute|exit)\b',
        r'\brecommend\s+(?:selling?|a\s+sell)\b',
    ]

    has_buy = any(re.search(p, lower) for p in buy_patterns)
    has_sell = any(re.search(p, lower) for p in sell_patterns)
    has_risk_ack = any(re.search(p, lower) for p in risk_keywords)
    has_oversold_ack = any(re.search(p, lower) for p in oversold_keywords)

    for val_str in rsi_matches:
        rsi_val = float(val_str)

        # RSI > 70 + BUY must acknowledge risk
        if rsi_val > 70 and has_buy and not has_risk_ack:
            return {
                "pass": False,
                "reason": (
                    f"RSI value {rsi_val} is above 70 (overbought territory) and a BUY signal is present "
                    "but no risk acknowledgment found."
                ),
            }

        # RSI < 30 + SELL must acknowledge oversold
        if rsi_val < 30 and has_sell and not has_oversold_ack and not has_risk_ack:
            return {
                "pass": False,
                "reason": (
                    f"RSI value {rsi_val} is below 30 (oversold territory) and a SELL signal is present "
                    "but no oversold/risk acknowledgment found."
                ),
            }

    return {"pass": True, "reason": "FOL trading rules are consistently applied."}


def validate_all(text: str) -> dict:
    """
    Run all 5 validators and return a combined result.

    Returns:
        {
            "total_pass": int,
            "total_checks": 5,
            "all_pass": bool,
            "checks": {
                "fuzzy": {"pass": bool, "reason": str},
                "causal": {"pass": bool, "reason": str},
                "temporal": {"pass": bool, "reason": str},
                "categorical": {"pass": bool, "reason": str},
                "fol": {"pass": bool, "reason": str},
            }
        }
    """
    checks = {
        "fuzzy": validate_fuzzy(text),
        "causal": validate_causal(text),
        "temporal": validate_temporal(text),
        "categorical": validate_categorical(text),
        "fol": validate_fol(text),
    }

    total_pass = sum(1 for result in checks.values() if result["pass"])
    total_checks = 5

    return {
        "total_pass": total_pass,
        "total_checks": total_checks,
        "all_pass": total_pass == total_checks,
        "checks": checks,
    }
