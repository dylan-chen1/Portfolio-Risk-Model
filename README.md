# Portfolio Risk Model with Monte Carlo VaR: How It Works
Portfolio Risk Model, tracking how much one might lose in the event of economic crisis. Using a GARCH model, we are able to get a portfolio volatility estimate for each day. The volatility is calculated each day using a recursive GARCH formula, taking into account the real portfolio change in the last day along with yesterday's volatility estimate. 

I tested the model against the 2008 financial crisis, 2020 covid crash, and 2022 bear market, running a Monte Carlo Simulation of 10,000 trials per trading day, with each day factoring the volatility estimate of the day into the distribution. The VaR (value at risk) is calculated for each day by finding the cutoff of the largest 1% of losses among the 10,000 trials. The CVaR (conditional value at risk) is calculated for each day by finding the average loss among the largest 1% of losses, showing how much of a loss you can expect in the case of a major market drop.

Once we have the VaR and CVaR for every trading day in the time window being observed, we check the real portfolio return that happened for  each day. If the actual loss exceeds the predicted VaR for the day, that counts as an exception which is recorded, along with that day's CVaR. The goal is for the model to predict VaRs accurate enough to where exceptions happen 1% of the time per time window, and the CVaR is as close as possible to the actual average losses that occurred.


## Findings and Process
Final Results using a t-distribution at 99% VaR:
| Window | Exceptions | Total Days | Rate | Target | Average Real Losses | CVaR |
| --- | --- | --- | --- | --- | --- | --- |
| 2008 Financial Crisis | 2 | 146 | 1.4% | 1% | 3.3% | 3.2% | 
| 2020 Covid Crash | 4 | 52 | 7.7% | 1% | 4.3% | 4.3% |
| 2022 Bear Market | 4 | 210 | 1.9% | 1% | 2.6% | 3.0% |

The test was first run with a 95% VaR with a normal distribution, so I was aiming for a 5% rate. At this point with a 95% VaR, the model was meant for daily use to analyze what may happen to our portfolio in the case of a moderate shock. An exception at this point should happen around once a month. 

The results of the model were rates that were around twice as large as the target, so I switched to a t-distribution. The t-distribution should work better than the normal distribution because of the fatter tails, allowing for the VaR to move farther away from the middle compared to a normal distribution. With a larger VaR, the rates could go down closer to the target.

Results with the t-distribution didn't change much, and this was because the t-distribution adds probability to the extreme ends of the distribution from the moderate ends of the distribution. For example, the VaR for the 2008 financial crisis at the 95% range had slightly shifted down by 2.8% but this small change didn't actually change the number of exceptions. The exception rate did not change for the 2020 covid crash either, but it did drop from 9% to 8.6% for the 2022 bear market. However, at the extreme 99% range, the t-distribution's strengths became evident, giving us the more accurate results we have now. Switching to the 99% VaR, our target rate shifts from 5% to 1%, and the model changes its purpose. It now analyzes what could happen to our portfolio in the event of a major economic crash (largest 1% of losses), rather than daily use. A crash like this could be expected to happen around 2 to 3 times in a typical year with 252 trading days.

With the GARCH volatility model, its recursive formula means it is more accurate with gradual changes in the market, while falling behind when sudden market spikes happen. With the 2020 Covid Crash, it wasn't able to adjust quickly enough to the drastic sudden changes in the market, leading to a rate that was 7.7 times larger than the target. While the model failed to accurately predict how often an exception would happen, it performed very well in predicting the average loss that would occur in the event of an exception, matching the CVaR to the average real losses that occurred during that window.

2008 and 2022 had more gradual market changes than in 2020, allowing for the GARCH model to produce a more accurate VaR, leading to rates much closer to the target. The CVaR was moderately accurate to the average real losses but did not match it to the level of 2020.

The model excelled in all three testing windows in measuring the loss we can expect to have from a major market crash. It tends to be slightly too optimistic in measuring the worst 1% of losses (VaR) as the exception rates were always above 1%, but is way too optimistic and unable to adjust accordingly to sudden shocks like the 2020 Covid Crash.



## Why GARCH
Using a GARCH model, we are able to get a portfolio volatility estimate for each day using a recursive GARCH formula. It takes into account the real portfolio change in the last day along with yesterday's volatility estimate. Through this recursive formula with every day's volatility being based on the volatility estimate of the day before, every past day has some sort of weight to the current day's volatility estimate that slowly decays over time. If a rolling volatility estimate was used, it treats every day as a hard cut-off, where the entire 252-day window shifts over, day by day. A market shock that happened a week ago would be weighed the same as one that happened 250 days ago, which wouldn't provide as good of a volatility estimate to the current day. The rolling volatility model would lag behind in registering shocks in time, waiting for the 252-day window to shift enough.

A limitation of this model is that the GARCH model was fitted to 17 years of data at once, factoring in the shocks that happened throughout 17 years into the model. The project would indirectly use knowledge of market crashes that hadn't happened yet, and this would get factored into the volatility estimates. A model that is oriented towards looking forward would need to use market data available up to the current point, which is out of the scope of this project.  

 Structure
- src/data_loader.py — pulls and caches price data, computes returns
- src/volatility_models.py — rolling historical vol and GARCH
- src/monte_carlo.py — Monte Carlo simulation + VaR and CVaR 
- src/backtest.py — crash-window backtest, exception counting

## Portfolio 
| Ticker | Weight | Role |
|---|---|---|
| SPY | 40% | Core US Stocks |
| QQQ | 20% | Big Tech, Big Swings |
| AGG | 30% | Bonds, small and steady |
| GLD | 10% | Gold, holds or gains value when stocks fall |


## Setup
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/backtest.py
```
