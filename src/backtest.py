
"""
backtest.py

Backtests the VaR model against real crash windows:
2008 financial crisis (Sep 2008 - Mar 2009)
2020 COVID crash (Feb 2020 - Apr 2020)
2022 Bear Market (Jan 2022 - Nov 2022

Runs a Monte Carlo VaR for each day in the window using each day's
GARCH volatility estimate, checking for if the actual losses for
each day exceeded the VaR. A well-calibrated 99% VaR should be
exceeded about 1% of the time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from monte_carlo import simulate_returns, value_at_risk, conditional_var

CRASH_WINDOWS = {
    "2008_financial_crisis": ("2008-09-01", "2009-03-31"),
    "2020_covid_crash": ("2020-02-15", "2020-04-30"),
    "2022_bear_market": ("2022-01-01", "2022-11-01")
}


def count_var_exceptions(actual_returns: pd.Series, predicted_var: pd.Series):
    """
    Compares actual daily losses to predicted VaR
    returns (num_exceptions, total_days, exception_rate).
    """
    losses = -actual_returns
    exceptions = losses > predicted_var # goes through every day's losses and checks if it is greater than the predicted_var, storing a series of True/False.
    return int(exceptions.sum()), len(actual_returns), float(exceptions.mean())


def build_cvar(actual_returns: pd.Series, predicted_var: pd.Series, predicted_cvar: pd.Series):
    """
    Gets the series of exceptions and retrieves the CVaR for each exception
    Finds the real loss that occurred on the exception days.
    Averages the CVaRs and real losses.
    """
    losses = -actual_returns
    exceptions = losses > predicted_var
    exceptions_predicted_cvar = predicted_cvar[exceptions] # keep only the CVaR for days where there is an exception
    exception_losses = losses[exceptions] # keep only the real losses for days where there is an exception
    return exception_losses.mean(), exceptions_predicted_cvar.mean()




def build_var_series(vol_series: pd.Series, dates: pd.DatetimeIndex,confidence: float = 0.99,
    n_sims: int = 10000, seed: int = 42, distribution: str = "normal", nu: Optional[float] = None) -> pd.Series:
    """
    Runs a Monte Carlo VaR for each date in `dates`, using that day's
    volatility from `vol_series`.
    """
    var_by_day = {}
    cvar_by_day = {}
    for i, date in enumerate(dates):
        sigma = vol_series.loc[date] # pulls volatility for each day as date gets iterated with the for loop
        sims = simulate_returns(mu=0.0, sigma=sigma, n_sims=n_sims, horizon=1, seed=seed + i, distribution=distribution, nu=nu) # monte carlo simulation every day with the new sigma
                                                                                                                                # and seed, returning an array of 10,000 values
        var_by_day[date] = value_at_risk(sims, confidence) # a dictionary that calls value_at_risk to get the VaR for the day through passing in sims
        cvar_by_day[date] = conditional_var(sims, confidence) # same thing as above
    return pd.Series(var_by_day), pd.Series(cvar_by_day) # returns a tuple. pd.Series turns these dictionaries into series, kine of like Excel Sheets with two columns


def run_backtest(port_rets: pd.Series, vol_series: pd.Series, confidence: float = 0.99, distribution: str = "normal", nu: Optional[float] = None):
    """
    Runs the exception-count backtest across every window in
    CRASH_WINDOWS. Returns a dictionary of results for each window.
    """
    results = {}
    for name, (start, end) in CRASH_WINDOWS.items():
        window_returns = port_rets.loc[start:end] # gets the real returns for the time period
        window_vol = vol_series.loc[window_returns.index] # garch volatility was already calculated, this is pulling the garch volatility for the specific date window
        var_preds, cvar_preds = build_var_series(window_vol, window_returns.index, confidence=confidence,distribution=distribution, nu=nu) # runs monte carlo simulation for each day with GARCH,
                                                                                                                               # returns the VaR and CVaR for each day in a tuple
        n_exc, n_days, rate = count_var_exceptions(window_returns, var_preds) # stores the number of exceptions, days, and the rate from the function count_var_exceptions
        average_losses, cvar = build_cvar(window_returns, var_preds, cvar_preds) # stores the average losses and CVaR from the function build_cvar
        results[name] = {"exceptions": n_exc, "days": n_days, "rate": rate, "average losses": average_losses, "CVaR": cvar} # puts the exceptions, days, and rates into a labeled dictionary
    return results


if __name__ == "__main__":
    from data_loader import load_price_data, compute_returns, portfolio_returns
    from volatility_models import fit_garch

    tickers = ["SPY", "QQQ", "AGG", "GLD"]
    weights = {"SPY": 0.40, "QQQ": 0.20, "AGG": 0.30, "GLD": 0.10}

    prices = load_price_data(tickers, start="2006-01-01", end="2023-01-01")
    rets = compute_returns(prices)
    port_rets = portfolio_returns(rets, weights)

    garch_result = fit_garch(port_rets, "t")
    nu_result = garch_result.params["nu"]
    vol_series = garch_result.conditional_volatility / 100

    results = run_backtest(port_rets, vol_series, confidence=0.99, nu=nu_result, distribution = "t")

    print(f"{'Window':<24}{'Exceptions':<13}{'Total Days':<13}{'Rate':<8}{'Target':<9}{'Average Real Losses':<22}{'CVaR':<8}")
    for name, r in results.items():
        print(f"{name:<24}{r['exceptions']:<13}{r['days']:<13}{r['rate']:<8.1%}{'1%':<9}{r['average losses']:<22.1%}{r['CVaR']:<8.1%}")
