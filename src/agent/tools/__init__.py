from .prices import load_prices
from .returns import compute_returns
from .volatility import compute_volatility
from .moving_average import moving_average
from .forecast import simple_forecast

TOOLS: dict = {
    "load_prices": load_prices,
    "compute_returns": compute_returns,
    "compute_volatility": compute_volatility,
    "moving_average": moving_average,
    "simple_forecast": simple_forecast,
}

__all__ = ["TOOLS"]
