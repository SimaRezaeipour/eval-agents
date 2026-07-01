"""Synthesizer: LLM call that reads the scratchpad → FinalAnswer."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.schemas import FinalAnswer
from src.llm.client import complete

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a financial analyst. Using ONLY the observations in <scratchpad>, produce a
FinalAnswer JSON. Every number in `claims` MUST come from a specific step (cite
source_step). If data is missing or a step failed (value is null), say so in `caveats`.
Do NOT fabricate values.

Output MUST be valid JSON matching this schema:
{
  "summary": "<narrative answer>",
  "claims": [{"statement": "...", "value": <float>, "source_step": "<step_id>"}, ...],
  "caveats": ["..."]
}
"""


def _serialize_state(state: dict[str, Any]) -> str:
    serializable: dict[str, str] = {}
    for k, v in state.items():
        serializable[k] = "null (step failed)" if v is None else repr(v)[:800]
    return json.dumps(serializable, indent=2)


def synthesize(query: str, state: dict[str, Any]) -> FinalAnswer:
    """Call the LLM with the scratchpad and return a validated FinalAnswer."""
    scratchpad = _serialize_state(state)
    user_message = f"Query: {query}\n\n<scratchpad>\n{scratchpad}\n</scratchpad>"

    raw = complete(system=SYSTEM_PROMPT, user=user_message)
    logger.debug("Synthesizer raw response: %s", raw[:500])

    data = json.loads(raw)
    return FinalAnswer.model_validate(data)
