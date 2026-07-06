from __future__ import annotations

from unittest.mock import patch

from evaluation.run_eval import evaluate_one


def _gold_query() -> dict:
    return {
        "id": "q001",
        "tier": "T1",
        "query": "What was AAPL vol in 2019?",
        "gold_plan": [],
        "gold_answer": {},
        "required_facts": [],
        "tolerance": 0.10,
    }


def _trace() -> dict:
    return {
        "query": "What was AAPL vol in 2019?",
        "plan": {"steps": []},
        "state": {},
        "errors": [],
        "answer": {},
        "latency_s": 1.0,
    }


def test_evaluate_one_includes_combined_overall_metric() -> None:
    system_fn = lambda q: _trace()  # noqa: E731
    with patch("evaluation.evaluators.faithfulness.complete", return_value='[{"supported": true}]'), patch(
        "evaluation.evaluators.alignment.complete", return_value='{"score": 5}'
    ):
        row = evaluate_one("agent", system_fn, _gold_query())

    assert "overall" in row
    assert row["overall"] == (row["completion"] + row["tool_f1"] + row["args"] + row["numeric"] + row["faithfulness"] + row["alignment"]) / 6
