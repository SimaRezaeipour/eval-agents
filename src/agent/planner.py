"""Planner: LLM call → structured Plan."""

from __future__ import annotations

import json
import logging

from src.agent.schemas import Plan
from src.data.loader import list_universe
from src.llm.client import complete

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a financial analysis planner. Given a user question, produce the shortest
valid JSON plan that can answer it accurately using ONLY these tools:
load_prices, compute_returns, compute_volatility, moving_average, simple_forecast.

Priority order:
1. Correctness: choose the minimal tool chain that actually answers the question.
2. Reliability: prefer simple plans with clear data flow.
3. Efficiency: avoid unnecessary steps.

Rules:
- Use real ticker symbols (uppercase).
- Dates must be ISO 8601 (YYYY-MM-DD).
- Dataset coverage: 1980-01-01 to 2020-04-01.
- Every step that consumes prior data must list its upstream step IDs in depends_on.
- Do NOT invent new tools or perform calculations inside args.
- Use the exact tool argument names expected by the tools:
  - load_prices: ticker, start, end
  - compute_returns: prices_ref, kind
  - compute_volatility: returns_ref, window, annualize
- Args that reference a previous step's output must use the step's id string exactly.
- For numeric questions, include the smallest plan that reaches the requested metric;
  if the requested metric is volatility, choose a simple volatility computation and
  avoid unnecessary intermediate forecasting steps.
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
