"""Tool: compute_returns — simple or log returns from a prices DataFrame."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def compute_returns(
    prices_ref: pd.DataFrame,
    kind: Literal["simple", "log"] = "log",
) -> pd.Series:
    """Compute daily returns from a prices DataFrame.

    Args:
        prices_ref: DataFrame returned by load_prices (must have 'adj_close' column).
        kind: 'log' (default) or 'simple'.

    Returns:
        pd.Series of daily returns, NaN-dropped.
    """
    if "adj_close" not in prices_ref.columns:
        raise ValueError("prices_ref must have an 'adj_close' column.")

    closes = prices_ref["adj_close"].astype(float)
    if kind == "log":
        returns = np.log(closes / closes.shift(1))
    else:
        returns = closes.pct_change()

    return returns.dropna().rename("returns")
