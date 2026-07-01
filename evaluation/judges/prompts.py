"""Prompt templates for LLM-as-judge evaluators."""

FAITHFULNESS_SYSTEM = """\
You are an evaluation judge. Your task is to check whether each numeric claim in an
agent's answer is supported by the tool observations in the scratchpad.

For each claim, output a JSON object:
  {"claim_id": <int>, "supported": true/false, "evidence_step": "<step_id or null>",
   "reason": "<1-sentence explanation>"}

A claim is supported ONLY if its numeric value appears within 1% of the value
in the cited source_step's observation. If the source_step is missing or null,
the claim is unsupported.

Return a JSON array of these objects — one per claim. No other text.
"""

FAITHFULNESS_USER = """\
Claims:
{claims_json}

Scratchpad (tool observations):
{state_json}
"""

ALIGNMENT_SYSTEM = """\
You are an evaluation judge. Rate how well the agent's plan addresses the user's query.

Scoring rubric:
  5 — Fully addresses: plan covers every part of the query, correct tickers/dates, no extraneous steps.
  4 — Mostly addresses: covers the main question; minor missing element.
  3 — Partially addresses: answers part of a multi-part question, or right approach but wrong scope.
  2 — Tangentially related: touches the topic but misses the actual question.
  1 — Irrelevant: plan does not address the query at all.

Output JSON: {"score": <1-5>, "reason": "<1-2 sentence explanation>"}
No other text.
"""

ALIGNMENT_USER = """\
Query: {query}

Plan:
{plan_json}
"""
