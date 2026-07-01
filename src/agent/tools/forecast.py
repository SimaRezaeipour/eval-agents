"""Tool: simple_forecast — point + interval forecast from a returns Series."""

from __future__ import annotations

import logging
import warnings
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ForecastResult(TypedDict):
    point: float
    lower: float
    upper: float


def simple_forecast(
    returns_ref: pd.Series,
    horizon: int = 21,
    method: Literal["mean", "ets", "linear", "qlib"] = "ets",
) -> ForecastResult:
    """Forecast cumulative return over *horizon* trading days.

    Args:
        returns_ref: Series of daily log-returns from compute_returns.
        horizon: Forecast horizon in trading days (default 21 ≈ 1 month).
        method: 'ets' (default), 'mean', 'linear', or 'qlib' (falls back to ets).

    Returns:
        dict with keys 'point', 'lower', 'upper' (all floats, log-return scale).
    """
    if not isinstance(returns_ref, pd.Series):
        raise TypeError("returns_ref must be a pandas Series.")
    if len(returns_ref) < 30:
        raise ValueError(f"Need at least 30 observations, got {len(returns_ref)}.")

    if method == "qlib":
        try:
            import qlib  # noqa: F401
        except ImportError:
            logger.warning("pyqlib not installed — falling back to ETS.")
            method = "ets"

    if method == "ets":
        return _ets_forecast(returns_ref, horizon)
    elif method == "mean":
        return _mean_forecast(returns_ref, horizon)
    elif method == "linear":
        return _linear_forecast(returns_ref, horizon)
    else:
        raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _mean_forecast(returns: pd.Series, horizon: int) -> ForecastResult:
    mu = float(returns.mean())
    sigma = float(returns.std())
    point = mu * horizon
    z = 1.96  # 95% interval
    margin = z * sigma * (horizon ** 0.5)
    return ForecastResult(point=point, lower=point - margin, upper=point + margin)


def _linear_forecast(returns: pd.Series, horizon: int) -> ForecastResult:
    x = np.arange(len(returns))
    y = returns.values
    coeffs = np.polyfit(x, y, deg=1)
    next_vals = np.polyval(coeffs, np.arange(len(returns), len(returns) + horizon))
    point = float(next_vals.sum())
    sigma = float(returns.std())
    z = 1.96
    margin = z * sigma * (horizon ** 0.5)
    return ForecastResult(point=point, lower=point - margin, upper=point + margin)


def _ets_forecast(returns: pd.Series, horizon: int) -> ForecastResult:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SimpleExpSmoothing(returns.values, initialization_method="estimated")
        fit = model.fit(optimized=True)

    forecast_vals = fit.forecast(horizon)
    point = float(forecast_vals.sum())
    sigma = float(returns.std())
    z = 1.96
    margin = z * sigma * (horizon ** 0.5)
    return ForecastResult(point=point, lower=point - margin, upper=point + margin)
