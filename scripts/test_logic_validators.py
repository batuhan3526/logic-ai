"""Tests for the 5 logic form validators."""
import pytest
from scripts.logic_validators import (
    validate_fuzzy, validate_causal, validate_temporal,
    validate_categorical, validate_fol, validate_all
)

# Fuzzy Logic tests
def test_fuzzy_passes_nuanced_rsi():
    text = "RSI is at 68, approaching overbought territory but not definitive."
    assert validate_fuzzy(text)["pass"] is True

def test_fuzzy_fails_binary_rsi():
    text = "RSI is 68 so it's overbought, sell immediately."
    assert validate_fuzzy(text)["pass"] is False

# Causal Logic tests
def test_causal_passes_valid_chain():
    text = "The Fed raised rates, which increased borrowing costs and reduced demand."
    assert validate_causal(text)["pass"] is True

def test_causal_fails_spurious():
    text = "Bitcoin rose because it was Tuesday."
    assert validate_causal(text)["pass"] is False

# Temporal Logic tests
def test_temporal_passes_consistent():
    text = "I recommend buying BTC as I expect prices to rise in the coming weeks."
    assert validate_temporal(text)["pass"] is True

def test_temporal_fails_contradictory():
    text = "I recommend buying BTC. Prices will likely decline significantly."
    assert validate_temporal(text)["pass"] is False

# Categorical Logic tests
def test_categorical_passes_correct():
    text = "Binance, one of the largest cryptocurrency exchanges, reported..."
    assert validate_categorical(text)["pass"] is True

def test_categorical_fails_wrong():
    text = "Binance, a financial regulator, imposed new rules..."
    assert validate_categorical(text)["pass"] is False

# FOL tests
def test_fol_passes_consistent_risk():
    text = "With RSI at 75, buying now carries elevated risk."
    assert validate_fol(text)["pass"] is True

def test_fol_fails_ignores_risk():
    text = "RSI is 82, this is a great time to buy with high confidence."
    assert validate_fol(text)["pass"] is False

# Combined
def test_validate_all_returns_scores():
    text = "RSI is at 68, near overbought. The Fed's rate hike increased costs. I suggest holding as prices may be stable."
    result = validate_all(text)
    assert "total_pass" in result
    assert "checks" in result
    assert len(result["checks"]) == 5

def test_validate_all_no_text():
    result = validate_all("Hello world")
    assert result["total_pass"] == 5  # all pass when no indicators found
