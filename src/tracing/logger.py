"""JSONL trace writer — one record per agent run."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRACE_FILE = Path("data/traces/traces.jsonl")


def log_run(trace: dict[str, Any]) -> None:
    """Append *trace* as a single JSONL line to the trace file."""
    TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **trace}
    with TRACE_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    logger.debug("Trace written to %s", TRACE_FILE)
