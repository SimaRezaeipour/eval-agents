"""Metric 2: Tool Selection F1 — multiset F1 between predicted and gold tool sequences."""

from __future__ import annotations

from collections import Counter
from typing import Any


def score(trace: dict[str, Any], gold: dict[str, Any]) -> float:
    """Compute F1 between the tool multisets of the predicted plan and the gold plan.

    Args:
        trace: Agent run trace (has 'plan' → {'steps': [...]}).
        gold: Gold query dict (has 'gold_plan' → [{...}]).

    Returns:
        F1 score in [0, 1].
    """
    pred_steps = trace.get("plan", {})
    if isinstance(pred_steps, dict):
        pred_steps = pred_steps.get("steps", [])

    gold_steps = gold.get("gold_plan", [])

    pred_tools = Counter(s.get("tool", "") for s in pred_steps if isinstance(s, dict))
    gold_tools = Counter(s.get("tool", "") for s in gold_steps if isinstance(s, dict))

    tp = sum((pred_tools & gold_tools).values())
    if tp == 0:
        return 0.0

    precision = tp / sum(pred_tools.values())
    recall = tp / sum(gold_tools.values())
    return 2 * precision * recall / (precision + recall)
