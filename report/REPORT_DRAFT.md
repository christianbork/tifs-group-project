# What drives bid-ask spreads on Nasdaq Copenhagen: informed trading or volatility?

**FINS5566 Trading in Financial Securities, T3 2026. Group project.**
Christian Bork, Emanuel Annvik, Mathilde Sørensen.

> Draft 7 (18 Nov 2026): draft 6 rewritten in plain language, with the nested driver tests (script 14, Clark-West added) as Section 4.5 and the quote-timing claim corrected after a day-level and width-matched check. Sections 1 to 7 are 2,408 words, counted without headings, tables, figures, captions, appendices and references (Student Guide limit: 2,500 words excluding charts, tables and diagrams).

---

## 1. Introduction and research objective

A market maker gets hurt in two ways. It can end up holding shares it never wanted while the price moves, and it can trade with someone who knows more than it does. Ho and Stoll (1981) build the spread on the first problem, so in their model it widens with volatility. Glosten and Milgrom (1985) build it on the second, so it widens when more of the order flow is informed. In the Lecture 6 market-making set-up both land in the same box, the risk adjustment between fair value and the quotes. We wanted to know which one a market maker in Copenhagen should actually watch.

We took six Nasdaq Copenhagen stocks through October 2026 and asked three things. Which explains the intraday spread better, the permanent price impact of order flow or a volatility forecast? Does either forecast the spread five minutes ahead better than the spread's own recent level? And is any of it worth money to someone posting quotes?

The tick size gave us a natural experiment. Novo Nordisk, DSV and ISS sit at the one-tick minimum spread more than half the time, so there the drivers can only show up as a higher chance of a second tick. Vestas, Pandora and GN have room to move.

## 2. Data and background analysis

### 2.1 Source and cleaning

The data are FactSet Time and Sales exports for NOVO.B, DSV, VWS, PNDORA, GN and ISS, one file per stock and day, with the best bid and ask attached to every print. We keep lit continuous-session trades with a valid quote and label the rest (auctions, dark, off-book, other venues): about 720,000 trades over 22 days (Table 1). We estimate on 1 to 21 October and test on 22 to 30 October. A third sample, 2 to 6 November, was downloaded after the models were fixed and is used only for testing.

*[Table 1: report_table1_data_summary.csv.]*

### 2.2 Trade signing

Trades are signed with the quote rule against the attached mid (94 to 98% of trades), then the tick test. Prints sharing a timestamp are one aggressive order walking the book. Against four full quote streams the sign agrees in at least 98.6% of cases.

### 2.3 Intraday patterns and the tick constraint

The first fifteen minutes are a different market: spreads are 1.7 to 2.5 times their midday level and volatility about three times, and both calm down by ten. Novo also jumps at the US open. That jump moved from 15:30 to 14:30 Copenhagen time in the last week of October, because Europe changed its clocks a week before America, so our time-of-day controls follow the real US open (Figure 1). Novo, DSV and ISS are at one tick 53 to 62% of the time, Vestas 22%, GN 8% and Pandora 4%.

### 2.4 Volatility surface: how much of the volatility is noise

Measured volatility depends on how often you look. Sample the mid every few seconds and you pick up quotes flickering between ticks, which is not information (Andersen et al., 2000). Across nine sampling intervals and every hour of the day we get a surface that flattens between 2 and 15 minutes (Figure 2, Table 2). At the 1-minute interval our models use, 20 to 27% of the variance in Novo, Vestas, Pandora and ISS is this transitory part, 12% in DSV and almost none in GN. The open is the noisiest hour at every interval.

*[Table 2 / Figure 2: 11_vol_signature.csv, 11_vol_surface.png.]*

### 2.5 Event days

On DSV's Q3 report day (21 October) volatility was ten robust standard deviations above normal and turnover 32 above, yet the spread was only 6% wider. On 29 October ISS fell 3.5% between 15:20 and 15:35, with a half-spread of 18.6 bps against a normal 4 to 5. A Hong Kong inquiry had that morning partly blamed an ISS subsidiary for the Wang Fuk Court fire, and Danish media picked it up at 15:20 (MarketWire, 2026).

*[Figure 1: report_fig1_intraday_tick.png.]*

## 3. Strategy and model development

One rule runs through everything: whatever we use for bar t is known before bar t starts.

### 3.1 Informed trading: permanent price impact

Following Hasbrouck (1991) we estimate a two-equation VAR in order time for each stock and day (five-day rolling windows for the three thinner stocks):

r_{k} = c + Σ_{i=0}^{p} b_{i} x_{k−i} + Σ_{i=1}^{p} a_{i} r_{k−i} + u_{k},    x_{k} = d + Σ_{i=1}^{p} d_{i} x_{k−i} + Σ_{i=1}^{p} g_{i} r_{k−i} + v_{k},

where r_k is the log mid return from order k to k+1 and x_k the sign of order k. The permanent impact is how far the mid has moved twenty orders after a buy, with day-block bootstrap intervals. We fix p = 20, where the impact has stopped changing. The spread model gets yesterday's estimate and a rolling one over the last 300 orders. As a cross-check we split every trade's effective half-spread into what the liquidity provider keeps after five minutes and what the price takes back.

### 3.2 Volatility forecasts

We deseasonalise 1-minute mid returns with an in-sample time-of-day factor and fit GARCH(1,1), GJR-GARCH, two rolling benchmarks and a HAR model on realised variance over the last bar, hour and day (Corsi, 2009). HAR has the lowest out-of-sample QLIKE, so it is our volatility input, measured relative to the seasonal level.

### 3.3 Spread model

For 5-minute bar t of stock i we estimate

HalfSpread_{it} = α_{i} + β_{1} Impact_{roll,it} + β_{2} Impact_{prevday,it} + γ log(VolForecast_{it} / Seasonal_{it}) + time-of-day + activity controls + ε_{it},

separately for the two groups, on in-sample days only. The dependent variable is the time-weighted quoted half-spread in bps; for the constrained group we add a fractional logit on the share of time above one tick. Standard errors are panel Newey-West with seven lags. The residuals stay correlated for much longer, so we only call a driver significant if it also survives 24 and 95 lags, Driscoll and Kraay (1998) errors, and clustering by stock-day and by day.

### 3.4 Quoting rule

To put a price on the forecasts we simulate a market maker. At the start of each bar it posts a half-spread h_t from a forecast or a benchmark, snapped to the tick grid. An aggressive order fills the quote if its first print paid at least h_t. The market maker earns h_t and pays whatever the mid does over the next five minutes. Wider quotes fill less and earn more per fill, so we also rerun the rules at equal average in-sample width.

### 3.5 No look-ahead

Lag lengths, hyper-parameters, seasonal factors and coefficients all come from in-sample days. The October test window was looked at by the original model and by each refinement, which is why we added the November week. That week has been scored more than once too, but never fitted or tuned on.

*[Figure 3: report_fig2_impact_by_stock.png.]*

## 4. Testing and validation

### 4.1 Adverse selection takes most of the spread

This was the first surprise. Five minutes after a trade the price has moved against the liquidity provider by 91 to 115% of the effective half-spread, in every stock. What is left for whoever posted the quote is at most 0.6 bps, and negative for Pandora and ISS. The VAR agrees: the permanent impact runs from 1.3 bps (Novo) to 6.3 bps (GN), every bootstrap interval well clear of zero (Figure 3). The spread is mostly a payment to people who know something.

### 4.2 In-sample: volatility is the stronger driver

In the unconstrained group (Table 3) a one standard deviation rise in the volatility forecast widens the half-spread by 0.30 bps (t = 5.7) and the same rise in rolling impact by 0.15 bps (t = 2.5). In the constrained group only volatility matters (0.11 bps, t = 4.1), and it works the way the tick forces it to, through the chance of a two-tick spread (fractional logit, t = 5.7). Volatility survives all six standard-error choices (lowest t 3.1 and 2.5). Impact is shakier: it drops to t = 1.9 with 24 lags or day clusters, and comes back once the last observed spread is in the model (t = 2.2).

*[Table 3: report_table2_spread_regressions.csv.]*

### 4.3 Out of sample (22 to 30 Oct): the drivers do not beat persistence

The honest benchmark for a spread forecast is the spread five minutes ago, and it is hard to beat. Our linear models beat the in-sample average and yesterday's spread in the unconstrained group (Diebold-Mariano −3.2 and −2.6), but not the last bar (DM −0.2 in both groups). Only the LightGBM refinement does: RMSE 1.08 against 1.32 bps constrained and 1.32 against 1.66 unconstrained, DM −2.0 and −3.4 (Table 4, Figure 4).

### 4.4 Forward test (2 to 6 Nov)

November turned out harder than planned, with Q3 reports from Novo and Pandora on the 4th and GN on the 5th. In the constrained group every model beat the last bar: RMSE 1.00 to 1.01 bps for the refined linear models and 0.93 for LightGBM, against 1.42 (DM −2.6 to −2.8). In the unconstrained group the October ranking repeated. The linear driver models did not beat the last bar (1.77 to 1.81 against 1.85 bps), the model with the lagged spread did (1.57, DM −2.5) and LightGBM did best (1.49, DM −2.5).

*[Table 4 / Figure 4: 08_oos_rmse.csv, 09_fwd_rmse.csv, 09_fwd_rmse.png.]*

### 4.5 Do the drivers add anything once the last spread is known?

Are the drivers useless for forecasting, or just crowded out? We held everything else fixed (lagged spread, time and activity controls) and added volatility, impact, or both (Table 6). On plain accuracy the answer is no: no driver gives a significant Diebold-Mariano gain in the linear models in either week. But that test is unfair to the larger of two nested models, which pays for estimating extra coefficients. The Clark and West (2007) correction says the drivers do carry information: adding both gives z = 2.6 and 3.9 in October (constrained, unconstrained) and 4.0 and 3.8 in November. So the information is there, but fifteen days is too little to estimate it sharply enough to help. LightGBM shows the cost of chasing it: adding the drivers helped in October's unconstrained group (DM −3.9) and hurt in November's (DM +3.8).

*[Table 6: 14_nested_driver_rmse.csv, 14_nested_driver_clark_west.csv.]*

## 5. Evaluation of results

### 5.1 Economic size

The effects are real and small. A one standard deviation volatility shock adds about 3% to the average half-spread of a constrained stock (0.11 of 3.2 bps) and about 6% for an unconstrained one (0.30 of 4.9 bps). The opening bell adds 70 to 150%.

### 5.2 What the quoting simulation says

No rule makes money (Table 5, Figure 5). A market maker that matches the market's spread (yesterday's, the last bar's, even the realised one) loses 0.5 to 0.6 bps per fill out of sample in the constrained group and 3.2 to 3.8 bps in the unconstrained group; in the forward week, 1.0 to 1.2 and 1.3 to 2.8 bps. This is Section 4.1 from the other side of the trade: the price moves against the quote by more than the half-spread pays.

The model rules lose less, mostly because they quote wider. Is there anything beyond width? In October, yes. With every rule scaled to the same average in-sample width, the rule using the volatility and impact forecasts beat the same rule without them on all seven days (t = 3.2 on daily totals, p = 0.02), also when we forced equal width inside the test week (t = 2.9). In November, no. It won on all five days, but it was quoting 7% wider in a volatile week, and with widths matched the advantage is gone (t = 0.4). LightGBM, the best forecaster, never beat the simple rule. In the constrained stocks the tick grid makes most rules identical anyway.

*[Table 5 / Figure 5: 10_quoting_fwd.csv, 10_quoting_frontier.png.]*

### 5.3 Stability and event days

Spreads drifted down in the test week (Pandora from about 9 to 7 bps); a QLR test dates the shift to 16 to 22 October for DSV, Vestas and Pandora, though with 21 daily observations that is a description more than a proof (Appendix C). The November report days did not behave alike: Novo's spread was 46% wider and Pandora's 70%, while GN's was only 8% wider on turnover 48 standard deviations above normal. And in the ISS sell-off no rule that sets its quote five minutes ahead got out of the way of the first bar (Figure 6).

*[Figure 6: 13_iss_event.png.]*

## 6. Model and strategy refinement

### 6.1 Stock-by-time controls

Letting every stock have its own time-of-day pattern raises the in-sample R² from 0.69 to 0.77 (constrained) and 0.66 to 0.70 (unconstrained), and makes rolling impact stronger in the unconstrained group (0.23 bps per standard deviation, t = 3.9). Out of sample it barely matters: RMSE moves by 0.02 to 0.03 bps.

### 6.2 LightGBM with SHAP

A gradient-boosted model on the same inputs plus the lagged spread, tuned by forward-chaining cross-validation on in-sample days, is the one model that beats the last bar. The reason is a little deflating. In mean absolute SHAP the lagged spread leads by a distance (0.78 to 0.86 bps), then the stock (0.70), then time of day and yesterday's impact (0.1 to 0.3). Volatility gets 0.02 (Figure 7). Retraining without a feature group says the same: the error rises 14.6% without the lagged spread and under 1% without either driver. LightGBM wins by modelling persistence more flexibly, not by reading the drivers better.

### 6.3 Regimes

A two-state HMM on deseasonalised volatility, filtered forward so it never sees the future, splits the bars into a low and a high state of roughly half each. In the constrained group volatility widens spreads only in the high state (0.51 bps per unit, t = 3.7, against 0.05 in the low state; difference t = 2.8). In the unconstrained group it matters in both and more in the high one (Figure 8).

*[Figures 7 and 8: report_fig4_shap.png, report_fig5_hmm_regimes.png.]*

## 7. Critical reflection and recommendations

### 7.1 What we think holds up

The trade signing is checked against real quote streams, and the models were frozen before the November data existed. We ran both the unfair and the fair test on our own drivers. And we report a result we did not hope for: the drivers are real, but five minutes ahead they do not beat the spread you can already see.

### 7.2 Limitations

We only see quotes when a trade happens, so our time-weighted spreads are off by about 5% for ISS and 18% for Novo in the hour we could check. We see one venue, not Cboe Europe, Aquis or Turquoise. Twenty-seven days is short, and three of the last five were report days. The simulation puts our market maker first in the queue on ties, which flatters it, and charges the full five-minute move on every fill, which does not. The timing result in Section 5.2 holds in one test week out of two.

### 7.3 Recommendations for a real implementation

1. Buy the full quote stream. Trade-attached quotes are 5% off for ISS and 18% off for Novo, worst where a market maker trades most, and a realistic fill model needs every quote.
2. Measure volatility over 15 minutes when setting the spread. A fifth to a quarter of 1-minute variance is bid-ask bounce (Section 2.4), so the 1-minute number is partly the spread talking to itself.
3. Quote wider than the touch, and wider still in the high-volatility state. Every rule at the market's spread lost money, and widening with volatility cut the loss in both test weeks, though only in October by more than width alone. One good week out of two is a reason to paper-trade it for a few months, not to commit capital.

---

## Appendix A. AI transparency statement (not counted)

We used generative AI in this project and this is how.

Emanuel used Claude (Anthropic, Claude Code) for: drafting and debugging the forward-test script (`analysis/09_forward_holdout.py`), the quoting-rule simulation (`analysis/10_quoting_rule.py`, `src/tifs_group_project/quoting.py`) and their unit tests; driving the FactSet Time and Sales exports for 1 to 30 October and 2 to 6 November and checking each file for the 25,000-row cap; re-running the full pipeline and comparing the regenerated tables with the committed ones; an independent code review of all scripts, which found and fixed three errors in the first version of the quoting simulation (fills counted per print instead of per order, quotes not snapped to the tick grid, a wrong inventory figure); drafting the report text from the output tables, and editing it for length and language. Every number in the report was checked against the table it comes from, and the tables were produced by the scripts in the repository, not by the model.

[Draft, Christian to confirm:] Christian used Claude (Anthropic, Claude Code) for: writing and debugging the data loader, signing, bars, price-impact, volatility, spread-model and refinement scripts (01 to 08), their unit tests and the figures notebook; a review of the first draft against the Student Guide and the Applied Econometrics material, which led to the second version (standard-error robustness, a dynamic specification, bootstrap intervals for the impact, GARCH diagnostics, ADF, Granger and QLR tests, the equal-width quote-timing test, scripts 12 and 13); and merging that version with the forward test. Outputs were checked with unit tests, the quote-stream validation, the cross-check of the VAR impact against the spread decomposition, an exact re-run of the previous version before any change, and a reconciliation of every number in the report with the output tables.

[Draft, Mathilde to confirm:] Mathilde used OpenAI Codex to help draft and review the nested driver tests (`analysis/14_nested_driver_tests.py`), which ask whether volatility and impact add forecasting information once the last spread is known. She checked the design, ran the script, audited that every model is scored on identical rows (`14_nested_driver_sample_audit.csv`) and interpreted the results. Emanuel added the Clark-West version of the test with Claude after a review.

The research question, the tick-constraint design, the model specifications, the decision to add a forward week and a quoting simulation, and the interpretation of the results are our own. AI output was treated as a draft to be verified, and we are responsible for all content.

## Appendix B. Contribution statement (not counted)

Roles follow the Student Guide. The work was coordinated through a shared private GitHub repository: numbered pipeline scripts with unit tests, pull requests reviewed before merging, and a plan file (`docs/PLAN.md`) that records every design decision and who made it.

| Member | Role (Student Guide) | Deliverables | Evidence |
|---|---|---|---|
| Christian Bork | Quantitative Analysis and Technology | Repository and pipeline design, data loader and stitching, Lee-Ready signing, bars, Hasbrouck VAR and spread decomposition, volatility models, spread regressions, out-of-sample race, refinements (stock-by-time controls, LightGBM and SHAP, HMM), second version (standard-error robustness, dynamic model, bootstrap, econometrics checks, equal-width quote test, ISS event), merge of all branches, figures notebook | `analysis/01` to `08`, `12`, `13`, `src/tifs_group_project/`, `notebooks/`, `CHANGELOG.md`, 27 commits |
| Emanuel Annvik | Risk and Sales (evaluation, refinement, report) | FactSet data exports (October re-export and November forward week), forward-test holdout (09), market-maker quoting simulation (10), volatility surface (11), quote-stream validation, two code reviews and fixes, Clark-West and timing robustness checks, report drafts 1 to 5 and 7, Word and PDF build | `analysis/09` to `11`, `src/tifs_group_project/quoting.py`, `data/README.md`, `report/`, 20+ commits |
| Mathilde Sørensen | Trading (market interpretation and validation) [to confirm] | Nested driver tests (14) answering the second research question, sample audit, review of the merged version before submission [add: report review, peer coordination] | `analysis/14`, `outputs/tables/14_*`, pull request 2, 3 commits |

## Appendix C. Applied Econometrics checks (not counted)

*[Table 7: 12_ae_appendix.csv.]*

## References
Andersen, T. G., Bollerslev, T., Diebold, F. X. and Labys, P. (2000). Great realizations. *Risk*, 13(3), 105–108.
Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821–856.
Clark, T. E. and West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291–311.
Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
Dickey, D. A. and Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366), 427–431.
Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
Driscoll, J. C. and Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4), 549–560.
Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
Glosten, L. R. and Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71–100.
Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438.
Hasbrouck, J. (1991). Measuring the information content of stock trades. *Journal of Finance*, 46(1), 179–207.
Ho, T. and Stoll, H. R. (1981). Optimal dealer pricing under transactions and return uncertainty. *Journal of Financial Economics*, 9(1), 47–73.
MarketWire (2026). ISS får del af skylden for dødsbrand: Medarbejder slukkede brandalarmer. Proinvestor.com, 29 October 2026, 15:20 CET. https://proinvestor.com/nyheder/iss-faar-del-af-skylden-for-doedsbrand-medarbejder-slukkede-brandalarmer-1793283616537
Lee, C. M. C. and Ready, M. J. (1991). Inferring trade direction from intraday data. *Journal of Finance*, 46(2), 733–746.
Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
