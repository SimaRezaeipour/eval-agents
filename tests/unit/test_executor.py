from __future__ import annotations

from src.agent.executor import execute
from src.agent.schemas import Plan, PlanStep


def test_execute_maps_alias_args_to_tool_parameters() -> None:
    plan = Plan(
        steps=[
            PlanStep(
                id="s1",
                tool="load_prices",
                args={"ticker": "AAPL", "start_date": "2020-01-01", "end_date": "2020-02-01"},
                depends_on=[],
                rationale="Load prices",
            )
        ]
    )

    def fake_load_prices(ticker: str, start: str, end: str) -> dict[str, str]:
        return {"ticker": ticker, "start": start, "end": end}

    result = execute(plan, {"load_prices": fake_load_prices})

    assert result["errors"] == []
    assert result["state"]["s1"] == {"ticker": "AAPL", "start": "2020-01-01", "end": "2020-02-01"}
