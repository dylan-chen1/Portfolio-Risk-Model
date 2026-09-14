# Portfolio Risk Model — Monte Carlo VaR

Value at Risk for a hypothetical portfolio, estimated via Monte Carlo
simulation on top of GARCH(1,1) volatility, backtested against the 2008
financial crisis and the 2020 COVID crash.

## Why GARCH
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
