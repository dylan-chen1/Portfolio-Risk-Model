"""
data_loader.py

Pulls historical price data for the portfolio, and converts
raw adjusted prices into daily returns used by the volatility,
Monte Carlo, and backtest code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_price_data(tickers: Iterable[str], start: str,
    end: str, cache: bool = True,) -> pd.DataFrame:
    """
    Download adjusted close prices for `tickers` between `start` and `end`
    (YYYY-MM-DD strings). Caches each ticker to a CSV under data/raw/ so
    repeat runs don't re-hit the API.

    Returns a DataFrame indexed by date, one column per ticker.
    """
    tickers = list(tickers)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = {}

    for ticker in tickers:
        cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.csv"
        if cache and cache_path.exists():
            series = pd.read_csv(cache_path, index_col=0, parse_dates=True)["Adj Close"]
        else:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, multi_level_index=False)
            if data.empty:
                raise ValueError(f"No data returned for {ticker} — check the symbol and date range.")
            series = data["Adj Close"]
            if cache:
                data.to_csv(cache_path)
        frames[ticker] = series

    prices = pd.DataFrame(frames).dropna(how="all")
    return prices


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Convert a price DataFrame to daily returns.
    Using log method, it is what GARCH math uses
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    return returns.dropna(how="all")


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Combine per-asset returns into a single portfolio return series using
    fixed weights
    """
    w = pd.Series(weights)
    missing = set(w.index) - set(returns.columns)
    if missing:
        raise ValueError(f"Weights reference tickers not in returns: {missing}")
    aligned = returns[w.index]
    return (aligned * w).sum(axis=1)


if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "AGG", "GLD"]
    prices = load_price_data(tickers, start="2006-01-01", end="2023-01-01")
    rets = compute_returns(prices)
    weights = {"SPY": 0.40, "QQQ": 0.20, "AGG": 0.30, "GLD": 0.10}
    port_rets = portfolio_returns(rets, weights)
    print(port_rets.describe())
