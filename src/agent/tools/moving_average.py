"""Tool: moving_average — simple rolling mean of closing prices."""

from __future__ import annotations

import pandas as pd


def moving_average(
    prices_ref: pd.DataFrame,
    window: int,
) -> pd.Series:
    """Compute simple moving average on adj_close prices.

    Args:
        prices_ref: DataFrame from load_prices (must have 'adj_close').
        window: Rolling window in trading days.

    Returns:
        pd.Series of rolling means, NaN values included at the start.
    """
    if "adj_close" not in prices_ref.columns:
        raise ValueError("prices_ref must have an 'adj_close' column.")
    if window < 1:
        raise ValueError(f"window must be ≥ 1, got {window}.")

    return prices_ref["adj_close"].astype(float).rolling(window).mean().rename(f"MA_{window}")
