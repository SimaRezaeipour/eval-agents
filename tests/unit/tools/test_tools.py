"""Unit tests for all 5 tools (no network, no LLM calls)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.agent.tools.forecast import simple_forecast
from src.agent.tools.moving_average import moving_average
from src.agent.tools.returns import compute_returns
from src.agent.tools.volatility import compute_volatility


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_prices() -> pd.DataFrame:
    """100 rows of synthetic price data with known properties."""
    rng = np.random.default_rng(42)
    n = 100
    returns = rng.normal(0.0005, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "date": pd.date_range("2019-01-01", periods=n, freq="B"),
        "close": close,
        "adj_close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n),
    })


@pytest.fixture()
def sample_returns(sample_prices: pd.DataFrame) -> pd.Series:
    return compute_returns(sample_prices)


# ---------------------------------------------------------------------------
# compute_returns
# ---------------------------------------------------------------------------

class TestComputeReturns:
    def test_log_returns_length(self, sample_prices: pd.DataFrame) -> None:
        r = compute_returns(sample_prices, kind="log")
        assert len(r) == len(sample_prices) - 1

    def test_simple_returns_length(self, sample_prices: pd.DataFrame) -> None:
        r = compute_returns(sample_prices, kind="simple")
        assert len(r) == len(sample_prices) - 1

    def test_log_returns_finite(self, sample_prices: pd.DataFrame) -> None:
        r = compute_returns(sample_prices)
        assert r.notna().all()

    def test_missing_adj_close_raises(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="adj_close"):
            compute_returns(df)


# ---------------------------------------------------------------------------
# compute_volatility
# ---------------------------------------------------------------------------

class TestComputeVolatility:
    def test_returns_float(self, sample_returns: pd.Series) -> None:
        vol = compute_volatility(sample_returns, window=30)
        assert isinstance(vol, float)

    def test_annualized_range(self, sample_returns: pd.Series) -> None:
        # Synthetic data has σ≈1.5%/day → annualized ≈ 24%
        vol = compute_volatility(sample_returns, window=30, annualize=True)
        assert 0.05 < vol < 2.0

    def test_not_annualized_smaller(self, sample_returns: pd.Series) -> None:
        vol_ann = compute_volatility(sample_returns, window=30, annualize=True)
        vol_raw = compute_volatility(sample_returns, window=30, annualize=False)
        assert vol_ann == pytest.approx(vol_raw * math.sqrt(252), rel=1e-6)

    def test_short_series_falls_back_to_available_history(self) -> None:
        short = pd.Series([0.01, -0.01, 0.02])
        vol = compute_volatility(short, window=30)
        assert isinstance(vol, float)

    def test_empty_series_raises(self) -> None:
        with pytest.raises(ValueError, match="no returns available"):
            compute_volatility(pd.Series([], dtype=float), window=30)

    def test_non_series_raises(self) -> None:
        with pytest.raises(TypeError):
            compute_volatility([0.01, 0.02], window=5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# moving_average
# ---------------------------------------------------------------------------

class TestMovingAverage:
    def test_length_preserved(self, sample_prices: pd.DataFrame) -> None:
        ma = moving_average(sample_prices, window=10)
        assert len(ma) == len(sample_prices)

    def test_first_values_nan(self, sample_prices: pd.DataFrame) -> None:
        ma = moving_average(sample_prices, window=10)
        assert ma.iloc[:9].isna().all()

    def test_last_value_finite(self, sample_prices: pd.DataFrame) -> None:
        ma = moving_average(sample_prices, window=10)
        assert math.isfinite(ma.iloc[-1])

    def test_missing_adj_close_raises(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="adj_close"):
            moving_average(df, window=2)

    def test_zero_window_raises(self, sample_prices: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            moving_average(sample_prices, window=0)


# ---------------------------------------------------------------------------
# simple_forecast
# ---------------------------------------------------------------------------

class TestSimpleForecast:
    def test_mean_method_keys(self, sample_returns: pd.Series) -> None:
        result = simple_forecast(sample_returns, horizon=21, method="mean")
        assert set(result.keys()) == {"point", "lower", "upper"}

    def test_ets_method_keys(self, sample_returns: pd.Series) -> None:
        result = simple_forecast(sample_returns, horizon=21, method="ets")
        assert set(result.keys()) == {"point", "lower", "upper"}

    def test_lower_le_point_le_upper(self, sample_returns: pd.Series) -> None:
        result = simple_forecast(sample_returns, horizon=21)
        assert result["lower"] <= result["point"] <= result["upper"]

    def test_linear_method(self, sample_returns: pd.Series) -> None:
        result = simple_forecast(sample_returns, horizon=5, method="linear")
        assert result["lower"] <= result["point"] <= result["upper"]

    def test_qlib_fallback(self, sample_returns: pd.Series) -> None:
        """qlib is not installed — should fall back to ETS without raising."""
        result = simple_forecast(sample_returns, horizon=5, method="qlib")
        assert "point" in result

    def test_insufficient_data_raises(self) -> None:
        short = pd.Series([0.01] * 10)
        with pytest.raises(ValueError, match="30"):
            simple_forecast(short, horizon=5)
