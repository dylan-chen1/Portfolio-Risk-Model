"""
monte_carlo.py

Monte Carlo VaR engine: simulate many possible portfolio return paths
using a volatility estimate (from volatility_models.py), then read the
VaR/CVaR off the simulated distribution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def simulate_returns(mu: float, sigma: float, n_sims: int,
    horizon: int = 1, seed: Optional[int] = None,
    distribution: str = "normal", nu: Optional[float] = None) returns np.ndarray:
    """
    Simulate `n_sims` draws of `horizon`-day cumulative return, using the
    daily mean (mu) and daily volatility (sigma). Draws from either a t
    or normal distribution, whichever one is requested
    """
    rng = np.random.default_rng(seed)
    if distribution == "t" and nu is None:
        raise ValueError("nu is required when distribution = 't'")
    if distribution == "t":
        daily_shocks = rng.standard_t(nu, size=(n_sims, horizon))
        daily_shocks /= np.sqrt(nu/(nu-2))
        daily_shocks = (daily_shocks*sigma)+mu
    else:
        daily_shocks = rng.normal(loc=mu, scale=sigma, size=(n_sims, horizon))
    # log returns are additive across days, so cumulative horizon return
    # is just the sum of the simulated daily returns
    return daily_shocks.sum(axis=1)


def value_at_risk(simulated_returns: np.ndarray, confidence: float = 0.99) -> float:
    """
    VaR at `confidence`: the cutoff where 1% of
    simulated outcomes are worse than it. Returned as a positive number
    representing the loss.
    """
    cutoff = np.percentile(simulated_returns, (1 - confidence) * 100)
    return -cutoff


def conditional_var(simulated_returns: np.ndarray, confidence: float = 0.99) -> float:
    """
    CVaR is the average loss among simulated outcomes
    worse than the VaR cutoff. It's  how bad the bad days get on average,
    not just where the cutoff sits.
    """
    cutoff = np.percentile(simulated_returns, (1 - confidence) * 100)
    tail = simulated_returns[simulated_returns <= cutoff]
    return -tail.mean()


if __name__ == "__main__":
    from data_loader import load_price_data, compute_returns, portfolio_returns
    from volatility_models import fit_garch

    tickers = ["SPY", "QQQ", "AGG", "GLD"]
    weights = {"SPY": 0.40, "QQQ": 0.20, "AGG": 0.30, "GLD": 0.10}

    prices = load_price_data(tickers, start="2006-01-01", end="2023-01-01")
    rets = compute_returns(prices)
    port_rets = portfolio_returns(rets, weights)

    garch_result = fit_garch(port_rets)
    # most recent day's volatility estimate, undoing the earlier x100
    # scaling to get back to a plain daily decimal (NOT annualized --
    # simulation works one day at a time)
    today_vol = garch_result.conditional_volatility.iloc[-1] / 100

    sims = simulate_returns(mu=0.0, sigma=today_vol, n_sims=10000, horizon=1, seed=42)
    var_95 = value_at_risk(sims, confidence=0.95)
    cvar_95 = conditional_var(sims, confidence=0.95)

    print(f"Latest GARCH daily volatility estimate: {today_vol:.4%}")
    print(f"1-day 95% VaR:  {var_95:.4%}")
    print(f"1-day 95% CVaR: {cvar_95:.4%}")

    portfolio_value = 10_000
    print(f"\nOn a ${portfolio_value:,} portfolio:")
    print(f"  95% VaR  -> {var_95:.4%} loss = ${var_95 * portfolio_value:,.2f}")
    print(f"  95% CVaR -> {cvar_95:.4%} loss = ${cvar_95 * portfolio_value:,.2f}")
