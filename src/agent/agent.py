"""Agent orchestrator: plan → execute → synthesize."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.agent.executor import execute
from src.agent.planner import make_plan
from src.agent.synthesizer import synthesize
from src.agent.tools import TOOLS
from src.tracing.logger import log_run

logger = logging.getLogger(__name__)


def run(query: str) -> dict[str, Any]:
    """Run the Plan-and-Execute agent on a natural-language query.

    Returns a trace dict with keys:
        query, plan, state, errors, answer, latency_s
    """
    t0 = time.time()

    logger.info("Planner: generating plan for query=%r", query[:80])
    plan = make_plan(query)

    logger.info("Executor: running %d steps", len(plan.steps))
    exec_result = execute(plan, TOOLS)

    logger.info("Synthesizer: generating answer")
    answer = synthesize(query, exec_result["state"])

    trace: dict[str, Any] = {
        "query": query,
        "plan": plan.model_dump(),
        "state": {k: repr(v)[:500] for k, v in exec_result["state"].items()},
        "errors": exec_result["errors"],
        "answer": answer.model_dump(),
        "latency_s": round(time.time() - t0, 3),
    }

    log_run(trace)
    return trace
