"""Executor: deterministic loop over a Plan — no LLM calls."""

from __future__ import annotations

import logging
from typing import Any

from src.agent.schemas import Plan

logger = logging.getLogger(__name__)


def _resolve_refs(args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Replace step-ID strings in args with their actual values from state.

    The planner emits args like {"prices_ref": "s1"} instead of copying raw data.
    This function swaps those ID strings for the real objects.
    """
    resolved: dict[str, Any] = {}
    for key, val in args.items():
        if isinstance(val, str) and val in state:
            resolved[key] = state[val]
        else:
            resolved[key] = val
    return resolved


def execute(plan: Plan, tools: dict[str, Any]) -> dict[str, Any]:
    """Run each step in the plan deterministically.

    Returns:
        {"state": {step_id: observation}, "errors": [{step, error}]}
    """
    state: dict[str, Any] = {}
    errors: list[dict[str, str]] = []

    for step in plan.steps:
        tool_fn = tools.get(step.tool)
        if tool_fn is None:
            msg = f"Unknown tool: {step.tool!r}"
            logger.error("Step %s — %s", step.id, msg)
            errors.append({"step": step.id, "error": msg})
            state[step.id] = None
            continue

        try:
            resolved_args = _resolve_refs(step.args, state)
            result = tool_fn(**resolved_args)
            state[step.id] = result
            logger.debug("Step %s (%s) OK", step.id, step.tool)
        except Exception as exc:
            logger.warning("Step %s (%s) FAILED: %s", step.id, step.tool, exc)
            errors.append({"step": step.id, "error": str(exc)})
            state[step.id] = None

    return {"state": state, "errors": errors}
