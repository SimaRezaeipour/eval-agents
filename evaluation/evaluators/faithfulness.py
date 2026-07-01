"""Metric 5: Faithfulness — LLM judge checks if claims are grounded in the scratchpad."""

from __future__ import annotations

import json
import logging
from typing import Any

from evaluation.judges.prompts import FAITHFULNESS_SYSTEM, FAITHFULNESS_USER
from src.llm.client import complete

logger = logging.getLogger(__name__)


def score(trace: dict[str, Any]) -> float:
    """Return fraction of claims supported by the scratchpad (0.0 if no claims)."""
    claims = trace.get("answer", {}).get("claims", [])
    if not claims:
        return 0.0

    indexed_claims = [{"claim_id": i, **c} for i, c in enumerate(claims)]
    state = trace.get("state", {})

    user_message = FAITHFULNESS_USER.format(
        claims_json=json.dumps(indexed_claims, indent=2),
        state_json=json.dumps(state, indent=2, default=str),
    )

    raw = complete(system=FAITHFULNESS_SYSTEM, user=user_message)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = next(iter(data.values()))
        supported = sum(1 for r in data if r.get("supported") is True)
        return supported / len(claims)
    except Exception as exc:
        logger.warning("Faithfulness judge parse error: %s — raw: %s", exc, raw[:200])
        return 0.0
