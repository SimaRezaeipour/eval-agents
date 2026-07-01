"""Single-shot LLM baseline — same trace schema as the agent, no tools."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.data.loader import load_prices
from src.llm.client import complete
from src.tracing.logger import log_run

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a financial analyst. Answer the user's question using the CSV data provided.
Produce a JSON response matching this schema exactly:
{
  "summary": "<narrative answer>",
  "claims": [{"statement": "...", "value": <float>, "source_step": "raw_data"}],
  "caveats": ["..."]
}
Be precise with numbers. If you cannot compute a value from the data, say so in caveats.
"""

_KNOWN_TICKERS = ["AAPL", "MSFT", "FB", "BABA", "BAC", "AMD", "INTC", "QCOM",
                  "TWTR", "SNAP", "ROKU", "GE", "C", "KO", "VZ"]


def _extract_tickers(query: str) -> list[str]:
    return [t for t in _KNOWN_TICKERS if t in query.upper()]


def _build_csv_snippet(tickers: list[str], n_rows: int = 200) -> str:
    parts: list[str] = []
    for ticker in tickers:
        try:
            df = load_prices(ticker, "2018-01-01", "2020-04-01")
            tail = df.tail(n_rows)[["date", "close", "adj_close", "volume"]]
            parts.append(f"# {ticker}\n{tail.to_csv(index=False)}")
        except Exception as exc:
            parts.append(f"# {ticker} — data unavailable: {exc}")
    return "\n".join(parts) if parts else "No relevant ticker data found."


def run(query: str) -> dict[str, Any]:
    """Run the single-shot baseline. Returns a trace dict with the same shape as the agent."""
    t0 = time.time()
    tickers = _extract_tickers(query)
    csv_snippet = _build_csv_snippet(tickers)

    user_message = f"Question: {query}\n\nData:\n{csv_snippet}"
    raw = complete(system=SYSTEM_PROMPT, user=user_message)

    try:
        answer = json.loads(raw)
    except json.JSONDecodeError:
        answer = {"summary": raw, "claims": [], "caveats": ["Response was not valid JSON"]}

    trace: dict[str, Any] = {
        "query": query,
        "plan": [],
        "state": {},
        "errors": [],
        "answer": answer,
        "latency_s": round(time.time() - t0, 3),
    }
    log_run(trace)
    return trace
