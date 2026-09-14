"""
volatility_models.py

Two ways to estimate the volatility that feeds the Monte Carlo VaR:

1. rolling_historical_vol — simple, backward-looking rolling std. Limitation is
   that it  treats every period as equally volatile and each year is a cutoff
2. fit_garch — GARCH(1,1). Lets today's volatility depend on recent
   shocks. It is more accurate to the behavior real
   crashes show
"""

from __future__ import annotations

import pandas as pd
from arch import arch_model


def rolling_historical_vol(returns: pd.Series, window: int = 252) -> pd.Series:
    """
    Annualized rolling volatility
    For every day, takes the standard deviation of the past `window`
    daily returns, then annualizes it (multiply by sqrt(252), since
    volatility scales with the square root of time).
    """
    daily_vol = returns.rolling(window=window).std()
    return daily_vol * (252 ** 0.5)


def fit_garch(returns: pd.Series, distribution: str="normal", p: int = 1, q: int = 1):
    """
    Fit a GARCH(p, q) model to daily returns and return the fitted result.
    Shocks decrease gradually. If t distribution, then this is where
    nu is calculated
    """
    
    scaled_returns = returns * 100 # percentages

    # mean="Zero": we're not trying to predict direction, only swing
    # size, which is what VaR actually needs.
    # dist="normal": plain bell curve
    # dist="t": fat-tailed distribution 
    model = arch_model(y = scaled_returns, p=p,q=q, mean="ZERO", vol="GARCH", dist = distribution)
    result = model.fit(disp="off")
    return result


if __name__ == "__main__":
    from data_loader import load_price_data, compute_returns, portfolio_returns
    import matplotlib.pyplot as plt

    tickers = ["SPY", "QQQ", "AGG", "GLD"]
    weights = {"SPY": 0.40, "QQQ": 0.20, "AGG": 0.30, "GLD": 0.10}

    prices = load_price_data(tickers, start="2006-01-01", end="2023-01-01")
    rets = compute_returns(prices)

    port_rets = portfolio_returns(rets, weights)

    rolling_vol = rolling_historical_vol(port_rets, window=252)

    garch_result = fit_garch(port_rets)
    # conditional_volatility comes back in percent-scale (since we scaled
    # returns by 100 before fitting) and as a daily number, so undo the
    # scaling and annualize it the same way as rolling_vol.
    garch_annualized_vol = (garch_result.conditional_volatility / 100) * (252 ** 0.5)

    plt.figure(figsize=(10, 5))
    plt.plot(rolling_vol.index, rolling_vol.values, label="Rolling historical (1yr)")
    plt.plot(garch_annualized_vol.index, garch_annualized_vol.values, label="GARCH(1,1)", alpha=0.8)
    plt.title("Portfolio Volatility: Rolling vs. GARCH")
    plt.ylabel("Annualized volatility")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig("volatility_comparison.png")
    print("Saved plot to volatility_comparison.png")
    print(garch_result.summary())

