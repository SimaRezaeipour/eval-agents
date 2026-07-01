"""Metric 3: Argument Correctness — per-step arg accuracy averaged across the plan."""

from __future__ import annotations

from typing import Any


def _args_match(pred_val: Any, gold_val: Any) -> bool:
    """Return True if pred_val is equivalent to gold_val."""
    if pred_val is None or gold_val is None:
        return pred_val is gold_val
    # Case-insensitive string comparison (handles tickers)
    if isinstance(pred_val, str) and isinstance(gold_val, str):
        return pred_val.strip().upper() == gold_val.strip().upper()
    # Numeric: allow 1% tolerance
    try:
        pf, gf = float(pred_val), float(gold_val)
        if gf == 0:
            return pf == 0
        return abs(pf - gf) / abs(gf) <= 0.01
    except (TypeError, ValueError):
        pass
    return pred_val == gold_val


def _step_score(pred_args: dict[str, Any], gold_args: dict[str, Any]) -> float:
    """Score a single step's args as correct / (correct + wrong + missing)."""
    all_keys = set(gold_args.keys())
    if not all_keys:
        return 1.0  # no expected args → vacuously correct

    correct = sum(
        1 for k in all_keys
        if k in pred_args and _args_match(pred_args[k], gold_args[k])
    )
    missing = sum(1 for k in all_keys if k not in pred_args)
    wrong = len(all_keys) - correct - missing
    return correct / (correct + wrong + missing)


def score(trace: dict[str, Any], gold: dict[str, Any]) -> float:
    """Return mean per-step argument correctness score in [0, 1]."""
    pred_steps = trace.get("plan", {})
    if isinstance(pred_steps, dict):
        pred_steps = pred_steps.get("steps", [])

    gold_steps = gold.get("gold_plan", [])
    if not gold_steps:
        return 0.0

    # Match predicted steps to gold steps by position (same tool, same order)
    step_scores: list[float] = []
    for i, gold_step in enumerate(gold_steps):
        if i < len(pred_steps):
            pred_step = pred_steps[i]
            if isinstance(pred_step, dict) and pred_step.get("tool") == gold_step.get("tool"):
                step_scores.append(
                    _step_score(
                        pred_step.get("args", {}),
                        gold_step.get("args", {}),
                    )
                )
            else:
                step_scores.append(0.0)
        else:
            step_scores.append(0.0)

    return sum(step_scores) / len(step_scores)
