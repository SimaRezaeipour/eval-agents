"""Evaluation runner — 30 queries × 2 systems = 60 rows → results/full.csv."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from baselines.singleshot import run as run_baseline
from evaluation.evaluators import alignment, args, completion, faithfulness, numeric, tool_f1
from src.agent.agent import run as run_agent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path("results/full.csv")
BENCHMARK_PATH = Path("benchmark/queries.jsonl")

SYSTEMS: dict[str, Any] = {
    "agent": run_agent,
    "baseline": run_baseline,
}

METRIC_COLS = ["completion", "tool_f1", "args", "numeric", "faithfulness", "alignment"]
OVERALL_METRIC_COL = "overall"


def evaluate_one(system_name: str, system_fn: Any, query: dict[str, Any]) -> dict[str, Any]:
    """Run one system on one query and return a scored row dict."""
    try:
        trace = system_fn(query["query"])
    except Exception as exc:
        logger.error("System %s failed on qid=%s: %s", system_name, query["id"], exc)
        trace = {"query": query["query"], "plan": [], "state": {}, "errors": [str(exc)], "answer": {}, "latency_s": 0}

    row = {
        "system": system_name,
        "qid": query["id"],
        "tier": query["tier"],
        "completion": completion.score(trace),
        "tool_f1": tool_f1.score(trace, query),
        "args": args.score(trace, query),
        "numeric": numeric.score(trace, query),
        "faithfulness": faithfulness.score(trace),
        "alignment": alignment.score(trace, query),
        "latency_s": trace.get("latency_s", 0),
    }
    row[OVERALL_METRIC_COL] = sum(row[m] for m in METRIC_COLS) / len(METRIC_COLS)
    return row


def main() -> None:
    queries = [json.loads(line) for line in BENCHMARK_PATH.read_text().splitlines() if line.strip()]
    logger.info("Loaded %d queries from %s", len(queries), BENCHMARK_PATH)

    rows: list[dict[str, Any]] = []
    for system_name, system_fn in SYSTEMS.items():
        logger.info("Evaluating system: %s", system_name)
        for q in queries:
            if q.get("adversarial"):
                continue  # exclude adversarial probes from main metrics
            row = evaluate_one(system_name, system_fn, q)
            rows.append(row)
            logger.info(
                "  qid=%s tier=%s %s=%.3f",
                row["qid"],
                row["tier"],
                OVERALL_METRIC_COL,
                row[OVERALL_METRIC_COL],
            )

    df = pd.DataFrame(rows)
    df["ARS"] = df[OVERALL_METRIC_COL]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)
    logger.info("Results written to %s (%d rows)", RESULTS_PATH, len(df))

    summary = df.groupby("system")[["ARS", OVERALL_METRIC_COL] + METRIC_COLS].mean().round(4)
    print("\n=== Results Summary ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
