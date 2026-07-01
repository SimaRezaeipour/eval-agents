"""DuckDB-based price loader — no Python loops over raw data."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import duckdb
import pandas as pd

_PARQUET_GLOB = "data/processed/prices.parquet/**/*.parquet"


@lru_cache(maxsize=1)
def _get_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        f"CREATE OR REPLACE VIEW prices AS SELECT * FROM read_parquet('{_PARQUET_GLOB}')"
    )
    return con


def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Return OHLCV rows for *ticker* between *start* and *end* (inclusive, ISO dates).

    Raises ValueError if the ticker is not in the universe or no rows are found.
    """
    con = _get_connection()
    df: pd.DataFrame = con.execute(
        """
        SELECT date, close, adj_close, volume
        FROM prices
        WHERE ticker = ?
          AND date BETWEEN ? AND ?
        ORDER BY date
        """,
        [ticker.upper(), start, end],
    ).df()

    if df.empty:
        raise ValueError(
            f"No price data found for ticker={ticker!r} between {start} and {end}. "
            "Check that the ticker is in the top-50 universe and the date range is valid "
            "(dataset covers up to 2020-04-01)."
        )
    return df


def list_universe() -> list[str]:
    """Return all tickers present in the processed parquet."""
    universe_csv = Path("data/processed/universe.csv")
    if universe_csv.exists():
        return pd.read_csv(universe_csv)["ticker"].tolist()
    con = _get_connection()
    return con.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").df()["ticker"].tolist()
