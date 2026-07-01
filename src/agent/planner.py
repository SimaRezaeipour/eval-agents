"""Planner: LLM call → structured Plan."""

from __future__ import annotations

import json
import logging

from src.agent.schemas import Plan
from src.data.loader import list_universe
from src.llm.client import complete

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a financial analysis planner. Given a user question, produce a JSON plan
using ONLY these tools: load_prices, compute_returns, compute_volatility,
moving_average, simple_forecast.

Rules:
- Use real ticker symbols (uppercase).
- Dates must be ISO 8601 (YYYY-MM-DD).
- Dataset coverage: 1980-01-01 to 2020-04-01.
- Every step that consumes prior data must list its upstream step IDs in depends_on.
- Do NOT invent new tools or perform calculations inside args.
- Args that reference a previous step's output must use the step's id string exactly.
- Output MUST be valid JSON matching the Plan schema:
  {"steps": [{"id": "s1", "tool": "...", "args": {...}, "depends_on": [], "rationale": "..."}, ...]}
"""


def make_plan(query: str) -> Plan:
    """Call the LLM with structured output and return a validated Plan."""
    universe = list_universe()

    user_message = (
        f"{query}\n\n"
        f"Available universe ({len(universe)} tickers): {', '.join(universe)}\n"
        "Data coverage: 1980-01-01 to 2020-04-01"
    )

    raw = complete(system=SYSTEM_PROMPT, user=user_message)
    logger.debug("Planner raw response: %s", raw[:500])

    data = json.loads(raw)
    if isinstance(data, list):
        data = {"steps": data}

    return Plan.model_validate(data)
