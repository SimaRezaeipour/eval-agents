"""Executor: deterministic loop over a Plan — no LLM calls."""

from __future__ import annotations

import logging
from typing import Any

from src.agent.schemas import Plan

logger = logging.getLogger(__name__)


def _resolve_refs(args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Normalize planner args to the tool signatures expected by the executor.

    The planner may emit aliases such as ``start_date``/``end_date`` or
    ``prices``/``returns``. This function swaps step-ID strings for actual
    values from state and maps common aliases to the canonical parameter names
    used by the tool implementations.
    """
    resolved: dict[str, Any] = {}
    for key, val in args.items():
        if isinstance(val, str) and val in state:
            resolved[key] = state[val]
        else:
            resolved[key] = val

    alias_map = {
        "start": ("start_date",),
        "end": ("end_date",),
        "prices_ref": ("prices", "prices_ref"),
        "returns_ref": ("returns", "returns_ref"),
    }

    for canonical_name, aliases in alias_map.items():
        if canonical_name in resolved:
            continue
        for alias in aliases:
            if alias in resolved:
                resolved[canonical_name] = resolved.pop(alias)
                break

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
