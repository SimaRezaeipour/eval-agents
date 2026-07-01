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

    Raises:
        ValueError: If fewer than *window* observations are available.
    """
    if not isinstance(returns_ref, pd.Series):
        raise TypeError("returns_ref must be a pandas Series.")
    if len(returns_ref) < window:
        raise ValueError(
            f"Not enough data: have {len(returns_ref)} rows, need at least {window}."
        )

    rolling_std = returns_ref.rolling(window).std().dropna()
    vol = float(rolling_std.iloc[-1])

    if annualize:
        vol *= math.sqrt(252)
    return vol
