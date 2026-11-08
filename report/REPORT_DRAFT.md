# What drives bid-ask spreads on Nasdaq Copenhagen: informed trading or volatility?

**FINS5566 Trading in Financial Securities, T3 2026. Group project.**
Christian Bork, Emanuel Annvik, Mathilde Sørensen.

> Draft 6 (8 Nov 2026): draft 5 merged with the second code version (standard-error robustness, dynamic specification, bootstrap intervals, Applied Econometrics checks in Appendix C, equal-width quote-timing test); every number re-read from the regenerated tables. Sections 1 to 7 are about 2,400 words, counted without headings, tables, figures, captions, appendices and references (the Student Guide limit is 2,500 words excluding charts, tables and diagrams). Every number comes from a table in `outputs/tables/`, named in the caption notes.

---

## 1. Introduction and research objective

A market maker loses money in two ways: holding inventory it did not want, and trading against people who know more. In Ho and Stoll (1981) the spread pays for inventory risk, so it widens with volatility; in Glosten and Milgrom (1985) it pays for losses to informed traders, so it widens with informed order flow. In the Lecture 6 market-making set-up both enter the same risk adjustment between fair value and the quotes. We ask which matters more on Nasdaq Copenhagen.

Using six stocks over 1 to 30 October 2026 we ask: which explains the intraday quoted spread better, the permanent price impact of order flow or a volatility forecast? Does either forecast the spread five minutes ahead better than its own persistence, out of sample? And does it change what a simulated market maker earns?

The tick size gives us a natural experiment. Novo Nordisk, DSV and ISS trade at the one-tick minimum spread more than half of the time, so for them the drivers can only show up as a higher chance of a two-tick spread. Vestas, Pandora and GN are not, so the drivers should be most visible there.

## 2. Data and background analysis

### 2.1 Source and cleaning

The data are FactSet Time and Sales exports for NOVO.B, DSV, VWS, PNDORA, GN and ISS, one file per stock and day from 08:45 to 17:20 Copenhagen time, with the best bid and ask attached to each print. We keep lit continuous-session trades with a valid quote and label the rest (auction, dark, off-book, other venue): about 720,000 trades over 22 days (Table 1). The in-sample period is 1 to 21 October (15 days), the out-of-sample period 22 to 30 October (7 days). A third sample, 2 to 6 November, was exported after all models were fixed and is used only for the test in Section 4.4.

*[Table 1: report_table1_data_summary.csv.]*

### 2.2 Trade signing

Trades are signed with the quote rule against the attached mid (94 to 98% of trades), then the tick test. Fills sharing a timestamp are one aggressive order walking the book (2.3 fills per order). Against four full quote streams the attached quote equals the prevailing one 92 to 95% of the time and the sign agrees in at least 98.6% of cases.

### 2.3 Intraday patterns and the tick constraint

Spreads in the first fifteen minutes are 1.7 to 2.5 times the midday level and volatility roughly three times; both settle by about 10:00. Novo's turnover and volatility also jump at the US open, which moved from 15:30 to 14:30 Copenhagen time on 26 to 30 October because European clocks changed a week before the American ones, so our time-of-day controls follow the actual US open (Figure 1). Novo, DSV and ISS have a one-tick spread 53 to 62% of the time; Vestas 22%, GN 8% and Pandora 4%.

### 2.4 Volatility surface: how much of the volatility is noise

Realised volatility from mid returns depends on the sampling interval: quote flicker and tick discreteness add variance at short horizons (Andersen et al., 2000). By interval and hour it gives a surface that flattens between 2 and 15 minutes (Figure 2, Table 2). At the 1-minute horizon our models use, 20 to 27% of the variance in Novo, Vestas, Pandora and ISS is transitory, 12% in DSV and almost none in GN. The open is noisier at every horizon, so part of the opening volatility a market maker reacts to is spread, not information.

*[Table 2 / Figure 2: 11_vol_signature.csv, 11_vol_surface.png.]*

### 2.5 Event days

On DSV's Q3 report day (21 October) volatility was ten robust standard deviations above its in-sample median and turnover 32 above, but the spread only 6% wider. ISS fell 3.5% between 15:20 and 15:35 on 29 October, with about 400 trades in one 5-minute bar and a half-spread of 18.6 bps, normally 4 to 5 (Figure 8). That morning a Hong Kong inquiry partly blamed an ISS subsidiary for the Wang Fuk Court fire; Danish media reported it at 15:20 (MarketWire, 2026).

*[Figure 1: report_fig1_intraday_tick.png.]*

## 3. Strategy and model development

Everything used for bar t is known before bar t starts.

### 3.1 Informed trading: permanent price impact

Following Hasbrouck (1991) we estimate a bivariate VAR in order time for each stock and day (five-day rolling windows for the three thinner stocks):

r_{k} = c + Σ_{i=0}^{p} b_{i} x_{k−i} + Σ_{i=1}^{p} a_{i} r_{k−i} + u_{k},    x_{k} = d + Σ_{i=1}^{p} d_{i} x_{k−i} + Σ_{i=1}^{p} g_{i} r_{k−i} + v_{k},

where r_k is the log mid return from order k to k+1 and x_k the sign of order k. The permanent impact is the cumulative mid response twenty orders after a unit buy, with day-block bootstrap intervals. BIC keeps falling up to 40 lags at our sample sizes, so we fix p = 20, where the impact has settled (less than 5% change to p = 40). Two versions enter the spread model: the previous day's estimate and a rolling estimate over the last 300 orders, smoothed with an EWMA. We also decompose each trade's effective half-spread into a realised part and the five-minute price impact.

### 3.2 Volatility forecasts

One-minute mid returns are deseasonalised with an in-sample time-of-day factor. On that series we fit GARCH(1,1) and GJR-GARCH with in-sample parameters, two rolling benchmarks, and a HAR model on realised variance over the last bar, hour and day (Corsi, 2009). HAR has the lowest out-of-sample QLIKE of the five and is the main volatility input. The forecast for bar t is made at the end of bar t−1 and expressed relative to the seasonal level.

### 3.3 Spread model

For 5-minute bar t of stock i we estimate

HalfSpread_{it} = α_{i} + β_{1} Impact_{roll,it} + β_{2} Impact_{prevday,it} + γ log(VolForecast_{it} / Seasonal_{it}) + time-of-day + activity controls + ε_{it},

separately for the two groups, on in-sample days only. The dependent variable is the time-weighted quoted half-spread in bps; we also estimate it for the effective half-spread and, for the constrained group, a fractional logit for the share of time the spread exceeds one tick. Standard errors are panel Newey-West with seven lags (the Applied Econometrics rule); because residuals stay autocorrelated much longer, a driver counts as significant only if it also survives 24 and 95 lags, Driscoll and Kraay (1998) errors and stock-day and day clusters.

### 3.4 Quoting rule

At the start of each bar the simulated market maker posts a half-spread h_t equal to a forecast or a benchmark, snapped to the tick grid and never below half a tick. An aggressive lit order fills the quote if it paid at least h_t on its first print, up to the stock's median order size. Revenue per filled share is h_t, the cost is the mid move over the following five minutes, and realised profit is the difference. A wider quote fills less and earns more per fill, so we also rerun every rule scaled to the same in-sample average width, leaving only the timing to differ.

### 3.5 No look-ahead

Lag lengths, hyper-parameters, seasonal factors and coefficients are chosen on in-sample days. The out-of-sample window was used once by the original model and twice more by the refinements in Section 6, hence the forward week in Section 4.4. That week was scored with the original specification and, after a code review, once more with the revised one; the revisions were decided without November data.

*[Figure 3: report_fig2_impact_by_stock.png.]*

## 4. Testing and validation

### 4.1 Adverse selection takes most of the spread

The price impact five minutes after a trade is 91 to 115% of the effective half-spread in every stock. The realised half-spread, what is left for the liquidity provider, is at most 0.6 bps and negative for Pandora and ISS. The VAR impact runs from 1.3 bps (Novo) to 6.3 bps (GN), every bootstrap interval is well above zero, and it agrees with the decomposition (Figure 3). The spread is mostly a payment to informed traders.

### 4.2 In-sample: volatility is the stronger driver, impact matters only where the tick does not bind

In the unconstrained group (Table 3) a one within-stock standard deviation increase in the volatility forecast widens the half-spread by 0.30 bps (t = 5.7) and the same increase in rolling impact by 0.15 bps (t = 2.5). In the constrained group only volatility is significant (0.11 bps, t = 4.1), and it works through the probability of a two-tick spread (fractional logit, t = 5.7). Volatility survives all six standard-error choices (lowest t 3.1 and 2.5); rolling impact falls to t = 1.9 with 24 lags or day clusters, but is significant once the last observed spread enters the model (t = 2.2) and in a Granger-type test (p = 0.03). Both hold with 15-minute bars, log spreads, GJR volatility and without the first fifteen minutes.

*[Table 3: report_table2_spread_regressions.csv.]*

### 4.3 Out of sample (22 to 30 Oct): the drivers do not beat persistence

The linear models beat the in-sample average and yesterday's spread in the unconstrained group (Diebold-Mariano −3.2 and −2.6) but not the last bar's spread (DM −0.2 in both groups). The LightGBM refinement is the only model that does: RMSE 1.08 against 1.32 bps in the constrained group and 1.32 against 1.66 in the unconstrained group, DM −2.0 and −3.4 (Table 4, Figure 4).

### 4.4 Forward test (2 to 6 Nov): never used for fitting or tuning

The week held the Q3 reports of Novo and Pandora (4 Nov) and GN (5 Nov). In the constrained group every model beat the last bar: RMSE 1.00 to 1.01 bps for the refined linear models and 0.93 for LightGBM, against 1.42 (DM −2.6 to −2.8). In the unconstrained group the ranking from 4.3 repeated: the linear driver models did not beat the last bar (1.77 to 1.81 against 1.85 bps), the lagged-spread model did (1.57, DM −2.5) and LightGBM did best (1.49, DM −2.5). Direction-of-change hit rates are 61 to 69% for every model, including the in-sample average, so the gain is in the size of the move, not its sign. On the report days the models failed together: every rule had an RMSE near 3 bps for Pandora.

*[Table 4 / Figure 4: 08_oos_rmse.csv, 09_fwd_rmse.csv, 09_fwd_rmse.png.]*

## 5. Evaluation of results

### 5.1 Economic size

A one standard deviation volatility shock adds about 3% to the average half-spread of a constrained stock (0.11 of 3.2 bps) and about 6% for an unconstrained one (0.30 of 4.9 bps); an impact shock of the same size adds about 3% in the unconstrained group. Both are small next to the open, which adds 70 to 150%.

### 5.2 What the quoting simulation says

No quoting rule makes money (Table 5, Figure 5). Quoting the market's own half-spread (yesterday's, the last bar's or the realised one) loses 0.5 to 0.6 bps per fill out of sample in the constrained group and 3.2 to 3.8 bps in the unconstrained group, and 1.0 to 1.2 and 1.3 to 2.8 bps in the forward week. As 4.1 shows, adverse selection costs more than the half-spread earns, however well the spread is forecast.

Model rules lose less, mostly by quoting wider: in the unconstrained forward week, 1.7 to 1.8 bps per fill for the linear models against 2.8 for the last bar, quoting 5.2 to 5.4 against 5.0 bps. Holding the average width fixed isolates timing (Table 6): the rule using the volatility and impact forecasts then earns more per day than the same rule without them in both weeks (paired t over stock-days 2.3 and 2.5); LightGBM, the best forecaster, does not (t = −0.6 and 0.4). In the constrained stocks the tick grid erases most differences between rules.

*[Table 6: 13_quote_timing.csv.]*

*[Table 5 / Figure 5: 10_quoting_fwd.csv, 10_quoting_frontier.png.]*

### 5.3 Stability and event days

Spreads drifted down in the out-of-sample week (Pandora from about 9 to 7 bps); a QLR test dates breaks to 16 to 22 October for DSV, Vestas and Pandora (Appendix C). The three report days responded differently: Novo's volatility was 21 standard deviations above its in-sample mean and its spread 46% wider (one tick only 15% of the day instead of 58%); Pandora's volatility 6.5 standard deviations up and its spread 70% wider; GN's turnover 48 standard deviations up but its spread only 8% wider. A three-state HMM puts 94 to 99% of all three days in its top state. In the ISS sell-off (Figure 8) no rule set five minutes ahead avoided the first bar.

## 6. Model and strategy refinement

### 6.1 Stock-by-time controls

Stock-by-time-bucket dummies instead of shared time dummies and stock fixed effects raise the in-sample R² from 0.69 to 0.77 in the constrained group and from 0.66 to 0.70 in the unconstrained group. The rolling impact coefficient in the unconstrained group becomes stronger (0.23 bps per standard deviation, t = 3.9) and the volatility coefficient in the constrained group weaker. Out of sample the RMSE changes by 0.02 to 0.03 bps (not significant).

### 6.2 LightGBM with SHAP

A gradient-boosted model on the same features plus the lagged spread, tuned by forward-chaining cross-validation on in-sample days, is the only model that beats the last bar. In mean absolute SHAP the lagged spread leads (0.78 to 0.86 bps), then the stock (0.70), time of day and previous-day impact (0.1 to 0.3); volatility is 0.02 (Figure 6). Dropping a feature group and re-training raises the cross-validated error by 14.6% for the lagged spread, 3.8% for time of day and under 1% for either driver. The improvement comes from modelling persistence more flexibly, not from reading the drivers better.

### 6.3 Regimes

A two-state Gaussian HMM on deseasonalised volatility, forward-filtered, splits the bars into low- and high-volatility states (about half each, spells of about 25 minutes). In the constrained group volatility widens spreads only in the high-volatility state (0.51 bps per unit, t = 3.7, against 0.05 in the low state; the difference has t = 2.8). In the unconstrained group the effect is present in both states and larger in the high one (Figure 7).

*[Figure 8: 13_iss_event.png.]*

*[Figures 6 and 7: report_fig4_shap.png, report_fig5_hmm_regimes.png.]*

## 7. Critical reflection and recommendations

### 7.1 What we think holds up

The trade signing is validated against full quote streams. There is an out-of-sample window and a week that was used only once. We report the negative result as it is: the drivers are real, but they do not forecast five minutes ahead better than the spread itself.

### 7.2 Limitations

Quotes are only observed at trades, so time-weighted spreads carry a sampling bias of about 5% for ISS and 18% for Novo in the hour we checked. We only see the primary lit venue, not Cboe Europe, Aquis or Turquoise. Twenty-seven days is short, and three of the last five were report days. Post-trade data is anonymous, so a PIN-type measure was not possible. The simulation puts the market maker first in the queue on ties and charges the full five-minute impact on every fill; the two biases go opposite ways. About a fifth of 1-minute variance is transitory (Section 2.4), so part of the volatility regressor reflects the spread it explains. The forward week was scored twice, once per code version; the second version's changes were decided without November data.

### 7.3 Recommendations for a real implementation

1. Replace trade-attached quotes with the full L1 stream. The sampling bias is 5% on ISS and 18% on Novo, largest where a market maker trades most, and a tick-level fill model needs it.
2. Measure volatility at the 15-minute plateau for the spread decision. 20 to 27% of 1-minute variance is bid-ask bounce (Section 2.4), so the 1-minute input is partly the spread itself.
3. Widen with the volatility forecast and condition on the regime. Every rule at the market's spread loses; at equal average width the volatility-adjusted rule still loses less per day in both test weeks, and volatility widens constrained spreads only in the high-volatility state. Run it live without executing before committing capital.

---

## Appendix A. AI transparency statement (not counted)

We used generative AI in this project and this is how.

Emanuel used Claude (Anthropic, Claude Code) for: drafting and debugging the forward-test script (`analysis/09_forward_holdout.py`), the quoting-rule simulation (`analysis/10_quoting_rule.py`, `src/tifs_group_project/quoting.py`) and their unit tests; driving the FactSet Time and Sales exports for 1 to 30 October and 2 to 6 November and checking each file for the 25,000-row cap; re-running the full pipeline and comparing the regenerated tables with the committed ones; an independent code review of all scripts, which found and fixed three errors in the first version of the quoting simulation (fills counted per print instead of per order, quotes not snapped to the tick grid, a wrong inventory figure); drafting the report text from the output tables, and editing it for length and language. Every number in the report was checked against the table it comes from, and the tables were produced by the scripts in the repository, not by the model.

[Draft, Christian to confirm:] Christian used Claude (Anthropic, Claude Code) for: writing and debugging the data loader, signing, bars, price-impact, volatility, spread-model and refinement scripts (01 to 08), their unit tests and the figures notebook; a review of the first draft against the Student Guide and the Applied Econometrics material, which led to the second version (standard-error robustness, a dynamic specification, bootstrap intervals for the impact, GARCH diagnostics, ADF, Granger and QLR tests, the equal-width quote-timing test, scripts 12 and 13); and merging that version with the forward test. Outputs were checked with unit tests, the quote-stream validation, the cross-check of the VAR impact against the spread decomposition, an exact re-run of the previous version before any change, and a reconciliation of every number in the report with the output tables.

[Mathilde: tools used, for what, and how the output was checked.]

The research question, the tick-constraint design, the model specifications, the decision to add a forward week and a quoting simulation, and the interpretation of the results are our own. AI output was treated as a draft to be verified, and we are responsible for all content.

## Appendix B. Contribution statement (not counted)

Roles follow the Student Guide. The work was coordinated through a shared private GitHub repository: numbered pipeline scripts with unit tests, pull requests reviewed before merging, and a plan file (`docs/PLAN.md`) that records every design decision and who made it.

| Member | Role (Student Guide) | Deliverables | Evidence |
|---|---|---|---|
| Christian Bork | Quantitative Analysis and Technology | Repository and pipeline design, data loader and stitching, Lee-Ready signing, bars, Hasbrouck VAR and spread decomposition, volatility models, spread regressions, out-of-sample race, refinements (stock-by-time controls, LightGBM and SHAP, HMM), figures notebook | `analysis/01` to `08`, `src/tifs_group_project/`, `notebooks/`, 16 commits |
| Emanuel Annvik | Risk and Sales (evaluation, refinement, report) | FactSet data exports (October re-export and November forward week), forward-test holdout (09), market-maker quoting simulation (10), volatility surface (11), quote-stream validation, code review and fixes, report drafts 1 to 5, Word and PDF build | `analysis/09` to `11`, `src/tifs_group_project/quoting.py`, `data/README.md`, `report/`, 9 commits |
| Mathilde Sørensen | [Trading, or as agreed] | [deliverables] | [evidence] |

## Appendix C. Applied Econometrics checks (not counted)

*[Table 7: 12_ae_appendix.csv.]*

## References
Andersen, T. G., Bollerslev, T., Diebold, F. X. and Labys, P. (2000). Great realizations. *Risk*, 13(3), 105–108.
Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821–856.
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
