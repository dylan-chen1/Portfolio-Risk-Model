# Portfolio Risk Model — Monte Carlo VaR
Portfolio Risk Model, tracking how much one might lose in the event of economic crisis. Using a GARCH model, we are able to get a portfolio volatility estimate for each day. The volatility is calculated each day using a recursive GARCH formula, taking into account the real portfolio change in the last day along with yesterday's volatility estimate. 
I tested the model against the 2008 financial crisis, 2020 covid crash, and 2022 bear market, running a Monte Carlo Simulation of 10,000 trials per trading day, with each day factoring the volatility estimate of the day into the distribution. The VaR (value at risk) is calculated for each day by finding the cutoff of the largest 1% of losses among the 10,000 trials. The CVaR (conditional value at risk) is calculated for each day by finding the average loss among the largest 1% of losses, showing how much of a loss you can expect in the case of a major market drop.
Once we have the VaR and CVaR for every trading day in the time window being observed, we check the real portfolio return that happened for each day. If the actual loss exceeds the predicted VaR for the day, that counts as an exception which is recorded. The goal is for the model to predict VaR's accurate enough to where exceptions happen 1% of the time per time window.
The CVaR is also recorded for each exception day, and the entire window's CVaR is calculated through averaging all the CVaR's from the window. We then find the real portfolio losses that happened for each exception day, averaging all the actual losses from exception days from the window. The average CVaR is compared against the actual average loss. The goal is for the CVaR to be as close as possible to the real average losses.

## Findings
Final Results using a t-distribution:
2008 Financial Crisis: 

## Why GARCH
Using a GARCH model, we are able to get a portfolio volatility estimate for each day. The volatility is calculated each day using a recursive GARCH formula, taking into account the real portfolio change in the last day along with yesterday's volatility estimate. 
Rolling historical volatility treats every period the same. GARCH(1,1)
lets today's volatility depend on recent shocks — "volatility
clustering" — which is exactly the pattern real crashes show. That
should make for a more honest VaR estimate, and a more interesting
backtest, than a flat historical-vol assumption.

## Structure
- `src/data_loader.py` — pulls & caches price data, computes returns (done)
- `src/volatility_models.py` — rolling historical vol + GARCH(1,1) (next)
- `src/monte_carlo.py` — Monte Carlo simulation + VaR/CVaR (next)
- `src/backtest.py` — crash-window backtest, exception counting (next)

## Default portfolio (change these tickers/weights anytime)
| Ticker | Weight | Role |
|---|---|---|
| SPY | 40% | US equity |
| QQQ | 20% | Growth/tech tilt |
| AGG | 30% | Fixed income |
| GLD | 10% | Hedge / alternative |

## Setup
```
pip install -r requirements.txt
python src/data_loader.py   # smoke test: pulls data, prints return stats
```
