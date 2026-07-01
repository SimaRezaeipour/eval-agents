"""Unit tests for the DuckDB price loader (requires processed parquet)."""

from __future__ import annotations

import pytest

# Skip entire module if processed parquet doesn't exist yet
from pathlib import Path
pytestmark = pytest.mark.skipif(
    not Path("data/processed/prices.parquet").exists(),
    reason="Processed parquet not found — run src.data.prepare first",
)

from src.data.loader import list_universe, load_prices  # noqa: E402


class TestLoadPrices:
    def test_aapl_2019_row_count(self) -> None:
        df = load_prices("AAPL", "2019-01-01", "2019-12-31")
        assert len(df) >= 250

    def test_columns_present(self) -> None:
        df = load_prices("AAPL", "2019-01-01", "2019-12-31")
        assert set(df.columns) >= {"date", "close", "adj_close", "volume"}

    def test_dates_in_range(self) -> None:
        df = load_prices("MSFT", "2018-06-01", "2018-09-30")
        assert df["date"].min().year == 2018
        assert df["date"].max().year == 2018

    def test_ticker_case_insensitive(self) -> None:
        df_upper = load_prices("AAPL", "2019-01-01", "2019-03-31")
        df_lower = load_prices("aapl", "2019-01-01", "2019-03-31")
        assert len(df_upper) == len(df_lower)

    def test_unknown_ticker_raises(self) -> None:
        with pytest.raises(ValueError, match="No price data"):
            load_prices("ZZZZ", "2019-01-01", "2019-12-31")

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="No price data"):
            load_prices("AAPL", "2025-01-01", "2025-12-31")


class TestListUniverse:
    def test_returns_50_tickers(self) -> None:
        universe = list_universe()
        assert len(universe) == 50

    def test_aapl_in_universe(self) -> None:
        assert "AAPL" in list_universe()

    def test_msft_in_universe(self) -> None:
        assert "MSFT" in list_universe()
