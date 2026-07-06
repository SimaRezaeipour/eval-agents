"""Tool: compute_volatility — annualized rolling volatility from a returns Series."""

from __future__ import annotations

import math

import pandas as pd


def compute_volatility(
    returns_ref: pd.Series,
    window: int = 30,
    annualize: bool = True,
) -> float:
    """Compute annualized (or raw) volatility as the rolling std of returns.

    Args:
        returns_ref: Series of daily returns from compute_returns.
        window: Rolling window in trading days (default 30).
        annualize: If True, multiply by sqrt(252).

    Returns:
        Scalar float — the volatility at the last available date.

    Falls back to the largest available window when the requested window is
    longer than the available history so short-series calculations still work.
    """
    if not isinstance(returns_ref, pd.Series):
        raise TypeError("returns_ref must be a pandas Series.")

    if len(returns_ref) == 0:
        raise ValueError("Not enough data: no returns available.")

    effective_window = min(window, len(returns_ref))
    if effective_window < 2:
        raise ValueError("Not enough data: need at least 2 observations.")

    rolling_std = returns_ref.rolling(effective_window).std().dropna()
    if rolling_std.empty:
        raise ValueError("Not enough data to compute volatility.")

    vol = float(rolling_std.iloc[-1])

    if annualize:
        vol *= math.sqrt(252)
    return vol
