# What drives bid-ask spreads on Nasdaq Copenhagen: informed trading or volatility?

**FINS5566 Trading in Financial Securities, T3 2026. Group project.** Christian Bork, Emanuel Annvik, Mathilde Sørensen.

## 1. Introduction and research objective

A market maker gets hurt in two ways. It can end up holding shares it never wanted while the price moves, and it can trade with someone who knows more than it does. Ho and Stoll (1981) build the spread on the first problem, so it should widen with volatility. Glosten and Milgrom (1985) build it on the second, so it should widen with informed order flow. We wanted to know which one a market maker in Copenhagen should actually watch.

We took six Nasdaq Copenhagen stocks through October 2026 and asked three things. Which explains the intraday spread better, the permanent price impact of order flow or a volatility forecast? Does either forecast the spread five minutes ahead better than the spread's own last value? And is any of it worth money to someone posting quotes?

The tick size gave us a natural experiment. Novo Nordisk, DSV and ISS sit at the one-tick minimum spread more than half the time, so there the drivers can only show up as a higher chance of a second tick. Vestas, Pandora and GN have room to move.

## 2. Data and background analysis

The data are FactSet Time and Sales exports with the best bid and ask attached to every print. We keep lit continuous-session trades with a valid quote: about 720,000 trades over 22 days (Table 1). We estimate on 1 to 21 October and test on 22 to 30 October. A second test week, 2 to 6 November, was downloaded after the models were fixed.

*[Table 1: report_table1_data_summary.csv.]*

Trades are signed with the quote rule (94 to 98% of trades), then the tick test. Prints sharing a timestamp are treated as one aggressive order walking the book. Against four full quote streams the sign agrees in at least 98.6% of cases.

The first fifteen minutes are a different market: spreads are about twice their midday level and volatility three times (Figure 1). Novo also jumps at the US open, which moved from 15:30 to 14:30 Copenhagen time in the last week of October because Europe changed its clocks a week before America. Our time-of-day controls follow the real US open.

*[Figure 1: report_fig1_intraday_tick.png.]*

Measured volatility depends on how often you look. Sample the mid every few seconds and you pick up quotes flickering between ticks, which is not information (Andersen et al., 2000). Across nine sampling intervals and every hour of the day, volatility flattens between 2 and 15 minutes (Figure 2). At the 1-minute interval our models use, 20 to 27% of the variance in four of the six stocks is this noise.

*[Figure 2: 11_vol_surface.png.]*

Two days stand out. On DSV's Q3 report day volatility was ten standard deviations above normal, yet the spread was only 6% wider. On 29 October ISS fell 3.5% in fifteen minutes after Danish media reported that a Hong Kong inquiry partly blamed an ISS subsidiary for the Wang Fuk Court fire (MarketWire, 2026), and its half-spread went from 4 to 19 bps.

## 3. Strategy and model development

One rule runs through everything: whatever we use for bar t is known before bar t starts.

**Informed trading.** Following Hasbrouck (1991) we estimate a VAR of mid returns and order signs in order time, per stock and day. The permanent impact is how far the mid has moved twenty orders after a buy. The spread model gets yesterday's estimate and a rolling one over the last 300 orders. As a cross-check we split every trade's effective half-spread into what the liquidity provider keeps after five minutes and what the price takes back.

**Volatility.** We deseasonalise 1-minute returns and fit GARCH, GJR-GARCH, two rolling benchmarks and a HAR model (Corsi, 2009). HAR forecasts best out of sample and is our volatility input.

**Spread model.** For 5-minute bar t of stock i:

HalfSpread_{it} = α_{i} + β_{1} Impact_{roll,it} + β_{2} Impact_{prevday,it} + γ log(VolForecast_{it} / Seasonal_{it}) + time-of-day + activity controls + ε_{it},

estimated per group on in-sample days. Standard errors are panel Newey-West; because the residuals stay correlated for long, we only call a driver significant if it also survives longer lags, Driscoll and Kraay (1998) errors and clustering by day.

**Quoting rule.** To put a price on the forecasts we simulate a market maker. Each bar it posts a half-spread from a forecast or a benchmark, snapped to the tick grid. An aggressive order fills the quote if its first print paid at least that much. The market maker earns the half-spread and pays whatever the mid does over the next five minutes.

## 4. Testing and validation

**Adverse selection takes most of the spread.** This was the first surprise. Five minutes after a trade the price has moved against the liquidity provider by 91 to 115% of the effective half-spread, in every stock (Figure 3). What is left for whoever posted the quote is at most 0.6 bps.

*[Figure 3: report_fig2_impact_by_stock.png.]*

**In-sample, volatility is the stronger driver.** In the unconstrained group a one standard deviation rise in the volatility forecast widens the half-spread by 0.30 bps and the same rise in rolling impact by 0.15 bps (Table 2). In the constrained group only volatility matters, and it works through the chance of a two-tick spread. Volatility survives every standard-error choice. Impact is shakier: its t-statistic drops to 1.9 with long lags or day clusters.

*[Table 2: 08_r1_coefficients.csv.]*

**Out of sample, the drivers do not beat persistence.** The honest benchmark for a spread forecast is the spread five minutes ago, and it is hard to beat. Our linear models beat yesterday's spread but not the last bar (Table 3). Only a LightGBM refinement does: RMSE 1.08 against 1.32 bps in the constrained group and 1.32 against 1.66 in the unconstrained group.

**The November week repeated the ranking.** It was a harder week, with Q3 reports from Novo, Pandora and GN. In the constrained group every model beat the last bar. In the unconstrained group only the model with the lagged spread and LightGBM did (Figure 4).

*[Table 3: 08_oos_rmse.csv, 09_fwd_rmse.csv.]*

*[Figure 4: 09_fwd_rmse.png.]*

**Do the drivers add anything once the last spread is known?** We held everything else fixed and added volatility, impact, or both (Table 4). On plain accuracy the answer is no: no driver gives a significant Diebold-Mariano gain in either week. But that test is unfair to the larger of two nested models, which pays for estimating extra coefficients. The Clark and West (2007) correction says the drivers do carry information, in both weeks and both groups. So the information is there, but fifteen days is too little to estimate it sharply enough to help.

*[Table 4: 14_nested_driver_rmse.csv, 14_nested_driver_clark_west.csv.]*

## 5. Evaluation of results

The effects are real and small. A one standard deviation volatility shock adds about 3% to the half-spread of a constrained stock and 6% for an unconstrained one. The opening bell adds 70 to 150%.

No quoting rule makes money (Table 5, Figure 5). A market maker that matches the market's last spread loses 0.5 bps per fill in the constrained group and 3.8 bps in the unconstrained group out of sample. This is the adverse-selection result from the other side of the trade: the price moves against the quote by more than the half-spread pays.

*[Table 5: 10_quoting_oos.csv, 10_quoting_fwd.csv.]*

*[Figure 5: 10_quoting_frontier.png.]*

The model rules lose less, mostly because they quote wider. Is there anything beyond width? In October, yes. With every rule scaled to the same average width, the rule using the volatility and impact forecasts beat the same rule without them on all seven days (t = 3.2), also with widths matched inside the test week (t = 2.9). In November, no. It was quoting 7% wider in a volatile week, and with widths matched the advantage is gone (t = 0.4). LightGBM, the best forecaster, never beat the simple rule.

## 6. Model and strategy refinement

**Stock-by-time controls.** Letting every stock have its own time-of-day pattern raises the in-sample R² from 0.69 to 0.77 and 0.66 to 0.70, and makes rolling impact stronger in the unconstrained group (t = 3.9). Out of sample it barely matters.

**LightGBM with SHAP.** The boosted model is the one that beats the last bar, and the reason is a little deflating. The lagged spread dominates its SHAP values and volatility barely registers (Figure 6). Retraining without the lagged spread raises the error by 15%; without either driver, by under 1%. LightGBM wins by modelling persistence more flexibly, not by reading the drivers better.

*[Figure 6: report_fig4_shap.png.]*

**Regimes.** A two-state HMM on volatility, filtered forward so it never sees the future, splits the bars into a low and a high state. In the constrained group volatility widens spreads only in the high state (t = 3.7, against 0.4 in the low state).

## 7. Critical reflection and recommendations

**What holds up.** The trade signing is checked against real quote streams, and the models were frozen before the November data existed. We ran both the unfair and the fair test on our own drivers. And we report a result we did not hope for: the drivers are real, but five minutes ahead they do not beat the spread you can already see.

**Limitations.** We only see quotes when a trade happens, so our spreads are off by about 5% for ISS and 18% for Novo in the hour we could check. We see one venue. Twenty-seven days is short, and three of the last five were report days. The simulation puts our market maker first in the queue on ties, which flatters it, and charges the full five-minute move on every fill, which does not. Both test weeks were scored more than once, though never fitted on. The timing result holds in one test week out of two.

**Recommendations.**

1. Buy the full quote stream. Trade-attached quotes are worst where a market maker trades most, and a realistic fill model needs every quote.
2. Measure volatility over 15 minutes when setting the spread. A fifth to a quarter of 1-minute variance is bid-ask bounce, so the 1-minute number is partly the spread talking to itself.
3. Quote wider than the touch, and wider still in the high-volatility state. Every rule at the market's spread lost money, and widening with volatility cut the loss in both test weeks, though only in October by more than width alone. That is a reason to paper-trade it, not to commit capital.

## References
Andersen, T. G., Bollerslev, T., Diebold, F. X. and Labys, P. (2000). Great realizations. *Risk*, 13(3), 105–108.
Clark, T. E. and West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291–311.
Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
Driscoll, J. C. and Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4), 549–560.
Glosten, L. R. and Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71–100.
Hasbrouck, J. (1991). Measuring the information content of stock trades. *Journal of Finance*, 46(1), 179–207.
Ho, T. and Stoll, H. R. (1981). Optimal dealer pricing under transactions and return uncertainty. *Journal of Financial Economics*, 9(1), 47–73.
MarketWire (2026). ISS får del af skylden for dødsbrand. Proinvestor.com, 29 October 2026.
