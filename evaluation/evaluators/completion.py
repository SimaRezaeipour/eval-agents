"""Metric 1: Task Completion — did the agent produce a usable FinalAnswer?"""

from __future__ import annotations

from typing import Any


def score(trace: dict[str, Any]) -> float:
    """Return 1.0 if the answer is a dict with ≥1 claim, else 0.0."""
    answer = trace.get("answer", {})
    if not isinstance(answer, dict):
        return 0.0
    claims = answer.get("claims", [])
    summary = answer.get("summary", "")
    if isinstance(claims, list) and len(claims) >= 1 and summary:
        return 1.0
    return 0.0
