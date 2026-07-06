from __future__ import annotations

from src.agent import planner, synthesizer


def test_planner_prompt_emphasizes_direct_numeric_plans() -> None:
    prompt = planner.SYSTEM_PROMPT
    assert "shortest" in prompt.lower()
    assert "volatility" in prompt.lower()
    assert "depends_on" in prompt


def test_synthesizer_prompt_emphasizes_exact_metric_claims() -> None:
    prompt = synthesizer.SYSTEM_PROMPT
    assert "exact requested metric" in prompt.lower()
    assert "source_step" in prompt
    assert "caveats" in prompt
