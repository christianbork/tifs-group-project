# What drives bid-ask spreads on Nasdaq Copenhagen: informed trading or volatility?

**FINS5566 Trading in Financial Securities, T3 2026. Group project.**
Christian Bork, Emanuel Annvik, Mathilde Sørensen.

> Draft 3 (8 Nov 2026). About 2,350 words in sections 1 to 7, excluding headings, tables, figure captions, the appendices and the reference list. Every number comes from a table in `outputs/tables/` (named in the caption notes); the tables were regenerated on 8 Nov from a full re-export of the October data plus the 2 to 6 Nov forward week.

---

## 1. Introduction and research objective

A market maker earns the spread and loses money in two ways: it ends up with inventory it did not want, and it trades against people who know more than it does. The two classic models of the spread focus on one each. Ho and Stoll (1981) treat the spread as payment for carrying inventory through price moves, so it should widen when volatility goes up. Glosten and Milgrom (1985) treat it as payment for losing to informed traders, so it should widen when more of the order flow is informed. In the market-making set-up from Lecture 6 both end up in the same place, the risk adjustment between the fair value estimate and the quotes. We wanted to know which of the two a market maker on Nasdaq Copenhagen should pay most attention to.

We use six Copenhagen stocks over 1 to 30 October 2026 and ask three questions. Which explains the intraday quoted spread better, a measure of informed trading (the permanent price impact of order flow) or a volatility forecast? Does either of them forecast the spread five minutes ahead better than the spread's own persistence, out of sample? And does it change what a simulated market maker earns if the forecasts are used to set quotes?

The tick size gives us a natural experiment. Novo Nordisk, DSV and ISS trade at the one-tick minimum spread more than half of the time, so for them the drivers can only show up as a higher chance of a two-tick spread. Vestas, Pandora and GN are not constrained in this way. If adverse selection and volatility matter, we should see it most clearly where the tick does not bind.

## 2. Data and background analysis

### 2.1 Source and cleaning

The data are FactSet Time and Sales exports for NOVO.B, DSV, VWS, PNDORA, GN and ISS, one file per stock and day from 08:45 to 17:20 Copenhagen time, with the best bid and ask attached to each print. FactSet cuts an export at 25,000 rows, so the busiest days were exported in two time windows and stitched together. We remove indicative auction prints, label auction prints, dark prints (venue DCSE), off-book and other-venue prints, and keep lit trades in the continuous session that have a valid quote. That leaves about 720,000 trades over 22 days (Table 1). The in-sample period is 1 to 21 October (15 days) and the out-of-sample period 22 to 30 October (7 days). A third sample, 2 to 6 November, was exported after all models were fixed, and we use it once, in Section 4.4.

*[Table 1: report_table1_data_summary.csv. Lit trades per day range from 1,132 (ISS) to 14,738 (Novo), turnover from DKK 30m to 542m per day, the median spread from 2.0 bps (Novo) to 13.9 bps (GN), and the share of time at one tick from 4% (Pandora) to 62% (ISS).]*

### 2.2 Trade signing

Trades are signed with the quote rule against the attached mid, and with the tick test when the price is at the mid. Between 94 and 98% of trades are signed by the quote rule and none are left unsigned. Fills that share a timestamp are treated as one aggressive order walking the book (2.3 fills per order on average, and 98% of multi-fill orders are on one side). This also explains why 8 to 32% of prints are outside the attached quote: the later fills in an order execute against the next price level. We checked the signing against four full quote streams (ISS on 30 Oct and 5 Nov, GN on 5 Nov, and one hour of Novo). The attached quote equals the prevailing quote 92 to 95% of the time and the trade sign agrees in at least 98.6% of cases (`02_quote_validation.csv`).

### 2.3 Intraday patterns

Spreads in the first fifteen minutes are 1.7 to 2.5 times the midday level, and volatility is roughly three times, in all six stocks. Both settle down by around 10:00. Novo's turnover and volatility also jump at the US open. That jump moved from 15:30 to 14:30 Copenhagen time on 26 to 30 October, because European clocks changed one week before the American ones, so our time-of-day controls are tied to the actual US open and not to the clock (Figure 1).

### 2.4 Tick constraint

Novo, DSV and ISS have a one-tick spread 53 to 62% of the time. For Vestas it is 22%, GN 8% and Pandora 4%. This split defines the two groups we use in the rest of the report.

### 2.5 Event days

DSV published its Q3 report on 21 October. Realised volatility that day was about ten robust standard deviations above its in-sample median and turnover 32 above, but the spread was only 6% wider than usual. ISS had a volatility shock on 29 October that we have not been able to link to an announcement.

*[Figure 1: report_fig1_intraday_tick.png.]*

## 3. Strategy and model development

The analysis has three building blocks that feed one regression, and the regression feeds a quoting rule. Everything used for bar t is known before bar t starts.

### 3.1 Informed trading: permanent price impact

Following Hasbrouck (1991) we estimate a bivariate VAR in order time for each stock and day (over rolling five-day windows for Pandora, GN and ISS, which have too few orders per day):

r_{k} = c + Σ_{i=0}^{p} b_{i} x_{k−i} + Σ_{i=1}^{p} a_{i} r_{k−i} + u_{k},    x_{k} = d + Σ_{i=1}^{p} d_{i} x_{k−i} + Σ_{i=1}^{p} g_{i} r_{k−i} + v_{k},

where r_k is the log mid return from order k to order k+1 and x_k is the sign of order k. The permanent impact is the cumulative mid response twenty orders after a unit buy. The lag length is chosen by BIC on in-sample days. Two versions go into the spread model: the previous day's estimate, and a rolling estimate over the last 300 orders smoothed with an EWMA. As a check we also split each trade's effective half-spread into a realised half-spread and a price impact measured five minutes after the trade.

### 3.2 Volatility forecasts

One-minute mid returns are deseasonalised with an in-sample time-of-day factor, where the US-open bump follows the actual US open. On the deseasonalised series we fit GARCH(1,1) and GJR-GARCH with in-sample parameters, a rolling 30-minute standard deviation, a 14-bar ATR, and a HAR model on realised variance over the last bar, hour and day (Corsi, 2009). The forecast for bar t is made at the end of bar t−1 and is expressed relative to the seasonal level, so the regression sees unusual volatility rather than the normal opening pattern.

### 3.3 Spread model

For 5-minute bar t of stock i we estimate

HalfSpread_{it} = α_{i} + β_{1} Impact_{roll,it} + β_{2} Impact_{prevday,it} + γ log(VolForecast_{it} / Seasonal_{it}) + time-of-day + activity controls + ε_{it},

separately for the two groups, on in-sample days only, with panel Newey-West standard errors. The dependent variable is the time-weighted quoted half-spread in basis points. We also estimate the same model for the effective half-spread, and for the constrained group a fractional logit for the share of time the spread is wider than one tick. The activity controls are the lagged log number of trades, log turnover and absolute order imbalance.

### 3.4 Quoting rule

At the start of each bar the simulated market maker posts a half-spread h_t equal to a forecast or a benchmark, but never less than half a tick. A lit trade fills the quote if the aggressor paid at least h_t, up to the stock's median trade size. Revenue per filled share is h_t, the cost is the mid move over the following five minutes, and realised profit is the difference. We compare the in-sample average, yesterday's average, the last bar's spread, each model forecast, and the realised spread of the bar itself as an upper bound.

### 3.5 No look-ahead

Lag lengths, hyper-parameters, seasonal factors and coefficients are all chosen on in-sample days. The out-of-sample window was used once by the original model and two more times by the refinements in Section 6. That is why Section 4.4 adds a week that was used only once.

*[Figure 2: report_fig2_impact_by_stock.png, permanent impact against the effective half-spread.]*

## 4. Testing and validation

### 4.1 Adverse selection takes most of the spread

The price impact five minutes after a trade is 91 to 115% of the effective half-spread in every stock. The realised half-spread, what is left for the liquidity provider, is at most 0.6 bps and is negative for Pandora and ISS (`05_impact_summary.csv`). The VAR impact goes from 1.4 bps (Novo) to 7.2 bps (GN) and agrees with the decomposition. In these stocks the spread is mostly a payment to informed traders, not a margin for the dealer.

### 4.2 In-sample: volatility is the stronger driver, impact matters only where the tick does not bind

In the unconstrained group a one within-stock standard deviation increase in the volatility forecast widens the half-spread by 0.28 bps (t = 5.5) and the same increase in rolling impact by 0.14 bps (t = 2.4). The median half-spreads in this group are 3 to 7 bps. In the constrained group only volatility is significant (0.09 bps, t = 3.7), and it works through the probability of a two-tick spread (fractional logit, t = 5.7). The results hold with 15-minute bars, when the first fifteen minutes are dropped, with log spreads, with GJR instead of HAR volatility, and with longer Newey-West lags.

*[Table 2: report_table2_spread_regressions.csv.]*

### 4.3 Out of sample (22 to 30 Oct): the drivers do not beat persistence

The linear models beat the in-sample average and yesterday's spread (Diebold-Mariano statistics between −2.2 and −3.3 in the unconstrained group) but they do not beat the last bar's spread (DM −0.25 and 0.02). The LightGBM refinement is the only model that does: RMSE 1.07 against 1.32 bps in the constrained group and 1.32 against 1.66 in the unconstrained group, DM −2.2 and −3.5.

*[Figure 3 / Table 3: report_fig3_oos_race.png, report_table3_oos_race.csv.]*

### 4.4 Forward test (2 to 6 Nov): used once

The week contained the Q3 reports of Novo and Pandora (4 Nov) and GN (5 Nov), so it was a harder test than we had planned for. In the constrained group every model beat the last bar: RMSE 1.01 bps with stock-by-time controls, 1.00 with the lagged spread added, and 0.94 for LightGBM, against 1.42 for the last bar, with DM between −2.6 and −2.8. In the unconstrained group the ranking from 4.3 came out the same. The linear driver models did not beat the last bar (1.77 to 1.81 against 1.85 bps, DM −0.4 to −0.6), the lagged-spread model did (1.58, DM −2.4) and LightGBM did best (1.46, DM −2.7). The constrained errors were lower than in October. All unconstrained errors were 7 to 29% higher, which is not surprising when two of the three stocks reported that week. Direction-of-change hit rates against the last bar are 61 to 67% for all models, so the gain is in the size of the move and not in its sign. On the report days the models failed together: for Pandora every rule had an RMSE close to 3 bps, and for Novo the linear model did worse than the last bar (tables `09_fwd_*.csv`).

*[Table 4 / Figure 6: 09_fwd_rmse.csv, 09_fwd_rmse.png.]*

## 5. Evaluation of results

### 5.1 Economic size

A one standard deviation volatility shock adds about 9% to the half-spread of a constrained stock and about 10% for an unconstrained one. An impact shock of the same size adds about 5% in the unconstrained group. These effects are real but small compared with the intraday pattern, where the open adds 70 to 150%.

### 5.2 What the quoting simulation says

None of the quoting rules make money (`10_quoting_oos.csv`, `10_quoting_fwd.csv`). A market maker that quotes the market's own half-spread, whether yesterday's, the last bar's or even the realised one, fills 46 to 62% of the lit trades and loses 1.0 to 1.4 bps per fill in the constrained group and 5.9 to 8.8 bps in the unconstrained group out of sample. In the forward week the losses are 2.4 to 2.8 and 3.5 to 4.0 bps. The reason is simple: the five-minute adverse-selection cost is larger than the half-spread earned. The model-based rules only change this at the margin. In the unconstrained group the lagged-spread model reduces the forward-week loss from 4.0 to 3.1 bps per fill, and it does so by quoting wider (6.0 against 5.4 bps) and filling fewer trades (46 against 50%). In the constrained group all rules end up within 0.1 bps of each other because the tick decides the quote. This fits with 4.1. If informed flow takes the whole spread, a quote at the market's spread cannot be profitable no matter how good the forecast is. A profitable quote has to be wider than the market's, or skewed away from the flow.

*[Table 5 / Figure 7: 10_quoting_fwd.csv, 10_quoting_frontier.png.]*

### 5.3 Stability

The volatility coefficient is stable across the robustness specifications. The rolling impact coefficient is significant only in the unconstrained group and only with stock-by-time controls. Spreads drifted down in the out-of-sample week (Pandora from about 9 to 7 bps), which is one more reason why the forward week was needed.

### 5.4 Event days

On DSV's report day volatility rose by ten standard deviations and the spread by 6%. In the forward week Novo's report day had volatility 21 standard deviations above its in-sample mean and a spread 47% wider than its in-sample average; the one-tick spread held 15% of the day instead of the usual 50%. Pandora's report day had volatility 6.5 standard deviations up and a spread 70% wider. GN's report day had turnover 48 standard deviations above normal but a spread only 8% wider than its in-sample average (`03_daily_stock_day.csv`). The forward-filtered HMM labelled 93% of Pandora's bars and 99% of GN's bars as stressed that day, but only 28% of Novo's (`08_hmm_stressed_share.csv`). So the same kind of news gave three quite different spread responses.

## 6. Model and strategy refinement

### 6.1 Stock-by-time controls

Replacing the shared time-of-day dummies and stock fixed effects with stock-by-time-bucket dummies raises the in-sample R² from 0.69 to 0.77 in the constrained group and from 0.66 to 0.70 in the unconstrained group. It makes the rolling impact coefficient in the unconstrained group stronger (0.23 bps per standard deviation, t = 3.9) and the volatility coefficient in the constrained group weaker. Out of sample it changes the RMSE by 0.02 to 0.03 bps, which is not significant.

### 6.2 LightGBM with SHAP

A gradient-boosted model on the same features plus the lagged spread, tuned with forward-chaining whole-day cross-validation on in-sample days, is the only model that beats the last-bar benchmark. The SHAP values show why. The lagged spread has 0.83 to 0.95 bps of mean absolute attribution, the stock 0.55, time of day and the previous day's impact 0.1 to 0.2 each, rolling impact 0.09, and the volatility forecast only 0.02 (Figure 4). The improvement comes from modelling persistence more flexibly, not from reading the drivers better. A linear model with the same lagged spread closes about half of the gap.

### 6.3 Regimes

A two-state Gaussian HMM on deseasonalised volatility, filtered forward so that the state at t only uses bars up to t, splits the bars into calm and stressed. In the constrained group volatility widens spreads only in the stressed state (0.46 bps per unit, t = 3.1, against −0.07 in the calm state; the difference has t = 3.0). In the unconstrained group the effect is there in both states and larger under stress (Figure 5). In other words, volatility matters for tick-constrained spreads when something is going on, and hardly at all otherwise.

*[Figures 4 and 5: report_fig4_shap.png, report_fig5_hmm_regimes.png.]*

## 7. Critical reflection and recommendations

### 7.1 What we think holds up

The trade signing is validated against full quote streams. Every regressor is known before the bar starts. The tick constraint works as a natural experiment. There is an out-of-sample window and a week that was used only once, and we have said how many times each was looked at. We also report the negative result as it is: the drivers are real, but they do not forecast five minutes ahead better than the spread itself.

### 7.2 Limitations

Quotes are only observed at trades, so the time-weighted spreads have a sampling bias of about 5% for ISS and 18% for Novo in the hour we checked (`02_quote_validation.csv`). We only see the primary lit venue; quotes on Cboe Europe, Aquis and Turquoise are missing. Twenty-two plus five days is a short sample, and three of the last five days were report days. Post-trade data on Nasdaq Copenhagen is anonymous, so a PIN-type measure based on broker identity was not possible. The quoting simulation assumes the market maker is first in the queue whenever its quote is at least as good as the market's, and it charges the full five-minute impact on every fill. Both assumptions work against the market maker.

### 7.3 Recommendations for a real implementation

1. Subscribe to the full L1 quote stream instead of trade-attached quotes. The bias is small in the thin stocks and large in the most liquid one, which is where a market maker trades most.
2. Consolidate quotes across venues. With the primary venue only, competition for the quote is understated and the fill model is too optimistic.
3. Run the quoting rule live for some weeks without executing, and use the regime state to decide when to widen. Volatility matters in stress, and quoting at the touch loses to informed flow also in calm markets.

---

## Appendix A. AI transparency statement (not counted)

We used generative AI in this project and this is how.

Emanuel used Claude (Anthropic, Claude Code) for: drafting and debugging the forward-test script (`analysis/09_forward_holdout.py`), the quoting-rule simulation (`analysis/10_quoting_rule.py`, `src/tifs_group_project/quoting.py`) and their unit tests; driving the FactSet Time and Sales exports for 1 to 30 October and 2 to 6 November and checking each file for the 25,000-row cap; re-running the full pipeline and comparing the regenerated tables with the committed ones; drafting the report text from the output tables, and editing it for length and language. Every number in the report was checked against the table it comes from, and the tables were produced by the scripts in the repository, not by the model.

[Christian: tools used for the data loader, signing, bars, price-impact, volatility, spread-model and refinement scripts (01 to 08), the tests and the figures notebook; how the outputs were checked (unit tests, the ISS quote-stream validation, cross-checks of the VAR impact against the spread decomposition).]

[Mathilde: tools used, for what, and how the output was checked.]

The research question, the choice of the tick-constraint design, the model specifications, the decision to add a forward week and a quoting simulation, and the interpretation of the results are our own. AI output was treated as a draft to be verified, and we are responsible for all content.

## Appendix B. Contribution statement (not counted)

| Member | Main responsibility | Deliverables | Evidence |
|---|---|---|---|
| Christian Bork | Pipeline design and estimation | Data loader and stitching, Lee-Ready signing, bars, Hasbrouck VAR and spread decomposition, volatility models, spread regressions, out-of-sample race, refinements (stock-by-time controls, LightGBM and SHAP, HMM), report figures 1 to 5 | `analysis/01` to `08`, `src/tifs_group_project/`, `notebooks/report_figures.ipynb`, 16 commits |
| Emanuel Annvik | Data, forward test, quoting simulation, report | FactSet exports for October and November, forward-test holdout (script 09), market-maker quoting simulation (script 10), quote-stream validation on four streams, report drafts 1 to 3, Word and PDF build | `analysis/09`, `analysis/10`, `src/tifs_group_project/quoting.py`, `data/README.md`, `report/`, 5 commits |
| Mathilde Sørensen | [role] | [deliverables] | [evidence] |

## References
Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
Glosten, L. R. and Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71–100.
Hasbrouck, J. (1991). Measuring the information content of stock trades. *Journal of Finance*, 46(1), 179–207.
Ho, T. and Stoll, H. R. (1981). Optimal dealer pricing under transactions and return uncertainty. *Journal of Financial Economics*, 9(1), 47–73.
Lee, C. M. C. and Ready, M. J. (1991). Inferring trade direction from intraday data. *Journal of Finance*, 46(2), 733–746.
Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
