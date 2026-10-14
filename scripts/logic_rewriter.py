#!/usr/bin/env python3
"""Rewrite logic-inconsistent answers using Claude API."""
import os

try:
    import anthropic
    client = anthropic.Anthropic()
except Exception:
    client = None

REWRITE_PROMPT = """Rewrite this finance answer to be factually accurate and logically consistent.
Keep the language natural — no formal notation, no logic symbols, no <think> blocks.
Keep it concise and similar in length to the original.

Original question: {question}
Original answer: {answer}
Failed logic checks: {failed_checks}

Rewrite the answer to fix the logic issues while keeping it natural and concise."""


def rewrite_answer(question: str, answer: str, failed_checks: list[str]) -> str | None:
    """Rewrite an answer to be logic-consistent. Returns None if API unavailable."""
    if client is None:
        return None
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(
                question=question, answer=answer,
                failed_checks="\n".join(failed_checks)
            )}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"  [REWRITE ERROR] {e}")
        return None
