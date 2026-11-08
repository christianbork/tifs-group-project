# What drives bid-ask spreads on Nasdaq Copenhagen: informed trading or volatility?

**FINS5566 Trading in Financial Securities, T3 2026. Group project.**
Christian Bork, Emanuel Annvik, Mathilde Sørensen.

> Draft 4 (8 Nov 2026, after the code review and pipeline re-run). Sections 1 to 7 are under 2,400 words, counted without headings, tables, figures, captions, appendices and references (the Student Guide limit is 2,500 words excluding charts, tables and diagrams). Every number comes from a table in `outputs/tables/`, named in the caption notes.

---

## 1. Introduction and research objective

A market maker earns the spread and loses money in two ways: holding inventory it did not want, and trading against people who know more. The two classic models of the spread focus on one each. Ho and Stoll (1981) treat the spread as payment for carrying inventory through price moves, so it should widen when volatility goes up. Glosten and Milgrom (1985) treat it as payment for losing to informed traders, so it should widen when more of the order flow is informed. In the market-making set-up from Lecture 6 both end up in the same place, the risk adjustment between fair value and the quotes. We ask which of the two matters more on Nasdaq Copenhagen.

We use six Copenhagen stocks over 1 to 30 October 2026 and ask three questions. Which explains the intraday quoted spread better, a measure of informed trading (the permanent price impact of order flow) or a volatility forecast? Does either of them forecast the spread five minutes ahead better than the spread's own persistence, out of sample? And does it change what a simulated market maker earns?

The tick size gives us a natural experiment. Novo Nordisk, DSV and ISS trade at the one-tick minimum spread more than half of the time, so for them the drivers can only show up as a higher chance of a two-tick spread. Vestas, Pandora and GN are not. If adverse selection and volatility matter, it should be most visible where the tick does not bind.

## 2. Data and background analysis

### 2.1 Source and cleaning

The data are FactSet Time and Sales exports for NOVO.B, DSV, VWS, PNDORA, GN and ISS, one file per stock and day from 08:45 to 17:20 Copenhagen time, with the best bid and ask attached to each print. We remove indicative auction prints, label auction, dark (venue DCSE), off-book and other-venue prints, and keep lit trades in the continuous session with a valid quote: about 720,000 trades over 22 days (Table 1). The in-sample period is 1 to 21 October (15 days), the out-of-sample period 22 to 30 October (7 days). A third sample, 2 to 6 November, was exported after all models were fixed and is used once, in Section 4.4.

*[Table 1: report_table1_data_summary.csv.]*

### 2.2 Trade signing

Trades are signed with the quote rule against the attached mid, then the tick test. Between 94 and 98% are signed by the quote rule and none are left unsigned. Fills that share a timestamp are treated as one aggressive order walking the book (2.3 fills per order; 98% of multi-fill orders are one-sided). Against four full quote streams (ISS on 30 Oct and 5 Nov, GN on 5 Nov, one hour of Novo) the attached quote equals the prevailing quote 92 to 95% of the time and the sign agrees in at least 98.6% of cases (`02_quote_validation.csv`).

### 2.3 Intraday patterns and the tick constraint

Spreads in the first fifteen minutes are 1.7 to 2.5 times the midday level and volatility roughly three times; both settle by about 10:00. Novo's turnover and volatility also jump at the US open, which moved from 15:30 to 14:30 Copenhagen time on 26 to 30 October because European clocks changed a week before the American ones, so our time-of-day controls follow the actual US open (Figure 1). Novo, DSV and ISS have a one-tick spread 53 to 62% of the time; Vestas 22%, GN 8% and Pandora 4%.

### 2.4 Event days

DSV published its Q3 report on 21 October. Realised volatility was about ten robust standard deviations above its in-sample median and turnover 32 above, but the spread was only 6% wider than usual. ISS had an unexplained volatility shock on 29 October.

*[Figure 1: report_fig1_intraday_tick.png.]*

## 3. Strategy and model development

Everything used for bar t is known before bar t starts.

### 3.1 Informed trading: permanent price impact

Following Hasbrouck (1991) we estimate a bivariate VAR in order time for each stock and day (five-day rolling windows for the three thinner stocks):

r_{k} = c + Σ_{i=0}^{p} b_{i} x_{k−i} + Σ_{i=1}^{p} a_{i} r_{k−i} + u_{k},    x_{k} = d + Σ_{i=1}^{p} d_{i} x_{k−i} + Σ_{i=1}^{p} g_{i} r_{k−i} + v_{k},

where r_k is the log mid return from order k to k+1 and x_k the sign of order k. The permanent impact is the cumulative mid response twenty orders after a unit buy; the lag length is chosen by BIC on in-sample days. Two versions enter the spread model: the previous day's estimate and a rolling estimate over the last 300 orders, smoothed with an EWMA. As a check we also split each trade's effective half-spread into a realised half-spread and the price impact five minutes after the trade.

### 3.2 Volatility forecasts

One-minute mid returns are deseasonalised with an in-sample time-of-day factor whose US-open bump follows the actual US open. On that series we fit GARCH(1,1) and GJR-GARCH with in-sample parameters, a rolling 30-minute standard deviation, a 14-bar ATR, and a HAR model on realised variance over the last bar, hour and day (Corsi, 2009). The forecast for bar t is made at the end of bar t−1 and expressed relative to the seasonal level.

### 3.3 Spread model

For 5-minute bar t of stock i we estimate

HalfSpread_{it} = α_{i} + β_{1} Impact_{roll,it} + β_{2} Impact_{prevday,it} + γ log(VolForecast_{it} / Seasonal_{it}) + time-of-day + activity controls + ε_{it},

separately for the two groups, on in-sample days only, with panel Newey-West standard errors. The dependent variable is the time-weighted quoted half-spread in basis points. We also estimate it for the effective half-spread and, for the constrained group, a fractional logit for the share of time the spread is wider than one tick. Lags run across the overnight gap, so the first bar of a day uses the last bar of the previous day.

### 3.4 Quoting rule

At the start of each bar the simulated market maker posts a half-spread h_t equal to a forecast or a benchmark, snapped to the tick grid and never below half a tick. Each aggressive lit order is a potential fill: if the aggressor paid at least h_t on its first fill, the market maker's quote was at least as good as the one that was hit, and it takes the order up to the stock's median order size. Revenue per filled share is h_t, the cost is the mid move over the following five minutes, and realised profit is the difference.

### 3.5 No look-ahead

Lag lengths, hyper-parameters, seasonal factors and coefficients are chosen on in-sample days. The out-of-sample window was used once by the original model and twice more by the refinements in Section 6, hence the once-used week in Section 4.4.

*[Figure 2: report_fig2_impact_by_stock.png.]*

## 4. Testing and validation

### 4.1 Adverse selection takes most of the spread

The price impact five minutes after a trade is 91 to 115% of the effective half-spread in every stock. The realised half-spread, what is left for the liquidity provider, is at most 0.6 bps and negative for Pandora and ISS (`05_impact_summary.csv`). The VAR impact runs from 1.4 bps (Novo) to 7.2 bps (GN) and agrees with the decomposition. The spread is mostly a payment to informed traders.

### 4.2 In-sample: volatility is the stronger driver, impact matters only where the tick does not bind

In the unconstrained group a one within-stock standard deviation increase in the volatility forecast widens the half-spread by 0.29 bps (t = 5.6) and the same increase in rolling impact by 0.14 bps (t = 2.4), against median half-spreads of 3 to 7 bps. In the constrained group only volatility is significant (0.09 bps, t = 3.7), and it works through the probability of a two-tick spread (fractional logit, t = 5.7). The results hold with 15-minute bars, log spreads, GJR volatility, longer Newey-West lags and without the first fifteen minutes.

*[Table 2: report_table2_spread_regressions.csv.]*

### 4.3 Out of sample (22 to 30 Oct): the drivers do not beat persistence

The linear models beat the in-sample average and yesterday's spread (Diebold-Mariano between −2.4 and −3.3 in the unconstrained group) but not the last bar's spread (DM −0.25 and 0.01). The LightGBM refinement is the only model that does: RMSE 1.07 against 1.32 bps in the constrained group and 1.32 against 1.66 in the unconstrained group, DM −2.1 and −3.4.

*[Figure 3 / Table 3: report_fig3_oos_race.png, report_table3_oos_race.csv.]*

### 4.4 Forward test (2 to 6 Nov): used once

The week contained the Q3 reports of Novo and Pandora (4 Nov) and GN (5 Nov). In the constrained group every model beat the last bar: RMSE 1.01 bps with stock-by-time controls, 1.00 with the lagged spread added and 0.94 for LightGBM, against 1.42, DM −2.6 to −2.8. In the unconstrained group the ranking from 4.3 repeated: the linear driver models did not beat the last bar (1.77 to 1.81 against 1.85 bps, DM −0.4 to −0.7), the lagged-spread model did (1.58, DM −2.4) and LightGBM did best (1.46, DM −2.7). Constrained errors were lower than in October; all unconstrained errors were 7 to 29% higher, because two of the three stocks reported. Direction-of-change hit rates are 61 to 67% for every model, including the in-sample average, which simply bets on mean reversion, so the gain is in the size of the move, not its sign. On the report days the models failed together: every rule had an RMSE near 3 bps for Pandora, and for Novo the linear model was worse than the last bar (`09_fwd_*.csv`).

*[Table 4 / Figure 6: 09_fwd_rmse.csv, 09_fwd_rmse.png.]*

## 5. Evaluation of results

### 5.1 Economic size

A one standard deviation volatility shock adds about 9% to the half-spread of a constrained stock and about 10% for an unconstrained one; an impact shock of the same size adds about 5% in the unconstrained group. These effects are real but small next to the intraday pattern, where the open adds 70 to 150%.

### 5.2 What the quoting simulation says

None of the quoting rules make money (`10_quoting_oos.csv`, `10_quoting_fwd.csv`). Quoting the market's own half-spread (yesterday's, the last bar's or the realised one) fills 75 to 79% of aggressive orders in the constrained group and loses 0.5 to 0.6 bps per fill out of sample; in the unconstrained group it fills 58 to 63% and loses 3.2 to 3.8 bps. In the forward week the losses are 1.0 to 1.2 and 1.3 to 2.8 bps. The reason is 4.1: the adverse-selection cost is larger than the half-spread earned, so a quote at the market's spread cannot pay however well it is forecast.

The model-based rules change this at the margin. They quote wider than the last bar and fill less often. In the constrained group out of sample, LightGBM quotes 2.3 bps against 1.5 for the last bar, fills 47% of orders instead of 75%, and loses 0.27 bps per fill instead of 0.49; the only rule positive on more than half of the days. In the unconstrained group the forward-week loss falls from 2.8 bps per fill (last bar) to 1.7 to 1.8 bps for the linear models, again by quoting wider (5.2 to 5.4 against 5.0 bps) and filling fewer orders (52 to 54% against 56%). A profitable quote would have to be wider still, or skewed away from the flow.

*[Table 5 / Figure 7: 10_quoting_fwd.csv, 10_quoting_frontier.png.]*

### 5.3 Stability and event days

The volatility coefficient is stable across the robustness specifications; the rolling impact coefficient is significant only in the unconstrained group and only with stock-by-time controls. Spreads drifted down in the out-of-sample week (Pandora from about 9 to 7 bps). The three report days in the forward week gave three different responses. Novo's had volatility 21 standard deviations above its in-sample mean and a spread 46% wider; the one-tick spread held 15% of the day instead of the usual 58%. Pandora's had volatility 6.5 standard deviations up and a spread 70% wider. GN's had turnover 48 standard deviations above normal but a spread only 8% wider (`03_daily_stock_day_fwd.csv`). The forward-filtered HMM labelled 93% of Pandora's and 99% of GN's bars as stressed that day, but only 28% of Novo's (`08_hmm_stressed_share.csv`).

## 6. Model and strategy refinement

### 6.1 Stock-by-time controls

Replacing the shared time-of-day dummies and stock fixed effects with stock-by-time-bucket dummies raises the in-sample R² from 0.69 to 0.77 in the constrained group and from 0.66 to 0.70 in the unconstrained group. The rolling impact coefficient in the unconstrained group becomes stronger (0.23 bps per standard deviation, t = 3.9) and the volatility coefficient in the constrained group weaker. Out of sample the RMSE changes by 0.02 to 0.03 bps (not significant).

### 6.2 LightGBM with SHAP

A gradient-boosted model on the same features plus the lagged spread, tuned by forward-chaining whole-day cross-validation on in-sample days, is the only model that beats the last-bar benchmark. The SHAP values show why. The lagged spread has 0.82 to 0.95 bps of mean absolute attribution, the stock 0.55, time of day and the previous day's impact 0.1 to 0.2 each, rolling impact about 0.1, and the volatility forecast only 0.02 (Figure 4). The improvement comes from modelling persistence more flexibly, not from reading the drivers better.

### 6.3 Regimes

A two-state Gaussian HMM on deseasonalised volatility, filtered forward so the state at t only uses bars up to t, splits the bars into calm and stressed. In the constrained group volatility widens spreads only in the stressed state (0.47 bps per unit, t = 3.2, against −0.07 in the calm state; the difference has t = 3.0). In the unconstrained group the effect is present in both states and larger under stress (Figure 5).

*[Figures 4 and 5: report_fig4_shap.png, report_fig5_hmm_regimes.png.]*

## 7. Critical reflection and recommendations

### 7.1 What we think holds up

The trade signing is validated against full quote streams. Every regressor is known before the bar starts. The tick constraint works as a natural experiment. There is an out-of-sample window and a week that was used only once. We report the negative result as it is: the drivers are real, but they do not forecast five minutes ahead better than the spread itself.

### 7.2 Limitations

Quotes are only observed at trades, so time-weighted spreads carry a sampling bias of about 5% for ISS and 18% for Novo in the hour we checked (`02_quote_validation.csv`). We only see the primary lit venue; quotes on Cboe Europe, Aquis and Turquoise are missing. Twenty-two plus five days is a short sample, and three of the last five days were report days. Post-trade data on Nasdaq Copenhagen is anonymous, so a PIN-type measure from broker identity was not possible. The quoting simulation assumes the market maker is first in the queue whenever its quote is at least as good as the market's (which flatters it), charges the full five-minute impact on every fill (which does not), and ignores fees and rebates. The seasonal volatility factor is a 15-day mean that includes DSV's report day, and the BIC lag choice for the VAR hit the top of its grid in every stock; neither changes the ranking of the drivers.

### 7.3 Recommendations for a real implementation

1. Subscribe to the full L1 quote stream instead of trade-attached quotes. The bias is small in the thin stocks and large in the most liquid one.
2. Consolidate quotes across venues. With the primary venue only, competition for the quote is understated and the fill model is too optimistic.
3. Run the quoting rule live for some weeks without executing, and use the regime state to decide when to widen: volatility matters in stress, and quoting at the touch loses to informed flow even in calm markets.

---

## Appendix A. AI transparency statement (not counted)

We used generative AI in this project and this is how.

Emanuel used Claude (Anthropic, Claude Code) for: drafting and debugging the forward-test script (`analysis/09_forward_holdout.py`), the quoting-rule simulation (`analysis/10_quoting_rule.py`, `src/tifs_group_project/quoting.py`) and their unit tests; driving the FactSet Time and Sales exports for 1 to 30 October and 2 to 6 November and checking each file for the 25,000-row cap; re-running the full pipeline and comparing the regenerated tables with the committed ones; an independent code review of all scripts, which found and fixed three errors in the first version of the quoting simulation (fills counted per print instead of per order, quotes not snapped to the tick grid, a wrong inventory figure); drafting the report text from the output tables, and editing it for length and language. Every number in the report was checked against the table it comes from, and the tables were produced by the scripts in the repository, not by the model.

[Christian: tools used for the data loader, signing, bars, price-impact, volatility, spread-model and refinement scripts (01 to 08), the tests and the figures notebook; how the outputs were checked (unit tests, the ISS quote-stream validation, cross-checks of the VAR impact against the spread decomposition).]

[Mathilde: tools used, for what, and how the output was checked.]

The research question, the tick-constraint design, the model specifications, the decision to add a forward week and a quoting simulation, and the interpretation of the results are our own. AI output was treated as a draft to be verified, and we are responsible for all content.

## Appendix B. Contribution statement (not counted)

| Member | Main responsibility | Deliverables | Evidence |
|---|---|---|---|
| Christian Bork | Pipeline design and estimation | Data loader and stitching, Lee-Ready signing, bars, Hasbrouck VAR and spread decomposition, volatility models, spread regressions, out-of-sample race, refinements (stock-by-time controls, LightGBM and SHAP, HMM), report figures 1 to 5 | `analysis/01` to `08`, `src/tifs_group_project/`, `notebooks/report_figures.ipynb`, 16 commits |
| Emanuel Annvik | Data, forward test, quoting simulation, report | FactSet exports for October and November, forward-test holdout (script 09), market-maker quoting simulation (script 10), quote-stream validation on four streams, code review and fixes, report drafts 1 to 4, Word and PDF build | `analysis/09`, `analysis/10`, `src/tifs_group_project/quoting.py`, `data/README.md`, `report/`, 6 commits |
| Mathilde Sørensen | [role] | [deliverables] | [evidence] |

## References
Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
Glosten, L. R. and Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71–100.
Hasbrouck, J. (1991). Measuring the information content of stock trades. *Journal of Finance*, 46(1), 179–207.
Ho, T. and Stoll, H. R. (1981). Optimal dealer pricing under transactions and return uncertainty. *Journal of Financial Economics*, 9(1), 47–73.
Lee, C. M. C. and Ready, M. J. (1991). Inferring trade direction from intraday data. *Journal of Finance*, 46(2), 733–746.
Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
