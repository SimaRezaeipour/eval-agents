"""Metric 6: Plan-Goal Alignment — LLM judge scores how well the plan answers the query."""

from __future__ import annotations

import json
import logging
from typing import Any

from evaluation.judges.prompts import ALIGNMENT_SYSTEM, ALIGNMENT_USER
from src.llm.client import complete

logger = logging.getLogger(__name__)


def score(trace: dict[str, Any], gold: dict[str, Any]) -> float:
    """Return plan-goal alignment score in [0, 1] (rescaled from 1–5 rubric)."""
    plan = trace.get("plan", {})
    query = trace.get("query", gold.get("query", ""))

    if not plan or not query:
        return 0.0

    user_message = ALIGNMENT_USER.format(
        query=query,
        plan_json=json.dumps(plan, indent=2, default=str),
    )

    raw = complete(system=ALIGNMENT_SYSTEM, user=user_message)
    try:
        data = json.loads(raw)
        raw_score = int(data.get("score", 1))
        return (raw_score - 1) / 4
    except Exception as exc:
        logger.warning("Alignment judge parse error: %s — raw: %s", exc, raw[:200])
        return 0.0
