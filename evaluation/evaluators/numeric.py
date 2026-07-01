"""Metric 4: Numerical Correctness — fraction of required facts within tolerance."""

from __future__ import annotations

from typing import Any


def score(
    trace: dict[str, Any],
    gold: dict[str, Any],
    default_tolerance: float = 0.05,
) -> float:
    """Return fraction of required_facts whose predicted value is within tolerance.

    Args:
        trace: Agent run trace (has 'answer' → {'claims': [...]}).
        gold: Gold query dict with 'gold_answer' and 'required_facts' keys.
        default_tolerance: Relative tolerance override (default 5%).

    Returns:
        Score in [0, 1].
    """
    gold_answer: dict[str, float] = gold.get("gold_answer", {})
    required_facts: list[str] = gold.get("required_facts", list(gold_answer.keys()))
    tolerance: float = gold.get("tolerance", default_tolerance)

    if not required_facts:
        return 0.0

    claims = trace.get("answer", {}).get("claims", [])

    hits = 0
    for key in required_facts:
        gold_val = gold_answer.get(key)
        if gold_val is None:
            continue
        # Find a matching claim by keyword presence in statement
        match = next(
            (c for c in claims if key.lower() in c.get("statement", "").lower()),
            None,
        )
        if match is not None:
            pred_val = match.get("value")
            if pred_val is not None and abs(gold_val) > 0:
                if abs(float(pred_val) - float(gold_val)) / abs(float(gold_val)) <= tolerance:
                    hits += 1

    return hits / len(required_facts)
