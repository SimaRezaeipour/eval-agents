"""Tool: load_prices — wraps the DuckDB loader."""

from __future__ import annotations

import pandas as pd

from src.data.loader import load_prices as _load_prices


def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted OHLCV data for *ticker* from *start* to *end* (ISO dates).

    Returns a DataFrame with columns: date, close, adj_close, volume.
    Raises ValueError if ticker/date range yields no data.
    """
    return _load_prices(ticker=ticker, start=start, end=end)
