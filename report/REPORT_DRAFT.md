# What drives bid-ask spreads on Nasdaq Copenhagen: adverse selection, volatility, and what a market maker can do with either

**FINS5566 Trading in Financial Securities, T3 2026. Group Project.**
Christian Bork, Emanuel Annvik, Mathilde Sørensen.

> Draft 2 (8 Nov 2026, Emanuel). Target 2,500 words excluding tables, figures and captions. Every number is taken from the tables in `outputs/tables/` (file named in the margin notes), regenerated on 8 Nov from a full re-export of the October data plus the 2-6 Nov forward week; scripts 01-08 reproduce Christian's committed tables exactly. Word budget per section in the headings is a guide, not a rule.

---

## 1. Introduction and research objective (~250 words)

A market maker earns the spread and loses to two things: inventory it did not want and traders who knew more than it did. The two classic models of the spread put the weight in different places. Ho and Stoll (1981) see the spread as compensation for holding inventory through price moves, so it should widen with volatility. Glosten and Milgrom (1985) see it as compensation for losing to informed traders, so it should widen with the share of informed flow. In the market-making architecture from Lecture 6 both enter the same box, the risk adjustment between fair value and the quotes. This project asks which of the two a market maker on Nasdaq Copenhagen should actually watch.

We study six Copenhagen stocks over 1 to 30 October 2026 and ask three questions. First, which explains the intraday quoted spread better, a measure of informed trading (the permanent price impact of order flow) or a volatility forecast? Second, does either help forecast the spread five minutes ahead out of sample, against the naive benchmark of the spread's own persistence? Third, does any of this change what a market maker earns when the forecasts are turned into quotes?

The design uses the tick size as a natural experiment. Three of the six stocks (Novo Nordisk, DSV, ISS) sit at the one-tick minimum spread more than half of the time, so the economic drivers can only show up in the probability of a two-tick spread. The other three (Vestas, Pandora, GN) are unconstrained. If adverse selection and volatility matter, they should matter most where the tick does not bind.

## 2. Data and background analysis (~400 words)

**Source and cleaning.** FactSet Time and Sales exports for NOVO.B, DSV, VWS, PNDORA, GN and ISS on Nasdaq Copenhagen, one file per stock-day, 08:45 to 17:20 local time, with the best bid and ask attached to each print. Days above the export's 25,000-row cap were exported in time windows and stitched. We drop indicative auction prints, label auction, dark (MPM, venue DCSE), off-book and other-venue prints, and keep continuous-session lit trades with a valid quote: about 720,000 trades over 22 days (Table 1). The in-sample period is 1 to 21 October (15 days), the out-of-sample period 22 to 30 October (7 days). A third sample, 2 to 6 November, was exported on 8 November after every model was fixed and is used exactly once in Section 4.4.

*[Table 1: report_table1_data_summary.csv. Lit trades per day from 1,132 (ISS) to 14,738 (Novo); turnover from DKK 30m to 542m per day; median spread 2.0 bps (Novo) to 13.9 bps (GN); time at one tick 4% (Pandora) to 62% (ISS).]*

**Trade signing.** Trades are signed with the quote rule against the attached mid and the tick test for midpoint prints; 94 to 98% are signed by the quote rule and none are left unsigned. Fills that share a timestamp are one aggressive order walking the book (2.3 fills per order; 98% of multi-fill orders are one-sided), which is why 8 to 32% of prints are outside the attached quote. Against four full quote streams (ISS 30 Oct, ISS 5 Nov, GN 5 Nov, one Novo hour) the attached quote equals the prevailing quote 92 to 95% of the time and the sign agrees 98.6 to 99.8% of the time (`02_quote_validation.csv`).

**Intraday patterns.** Spreads in the first fifteen minutes are 1.7 to 2.5 times the midday level and volatility 2.8 to 3.6 times; both settle by 10:00. In the last fifteen minutes spreads are about 0.85 times midday. Novo's turnover and volatility spike at the US open, which moved from 15:30 to 14:30 Copenhagen time on 26 to 30 October when European clocks changed a week before American ones, so time-of-day controls are keyed to the actual US open (Figure 1).

**Tick constraint.** Novo, DSV and ISS sit at one tick 53 to 62% of the time; Vestas 22%, GN 8%, Pandora 4%. This defines the two groups used throughout.

**Event days.** DSV reported Q3 on 21 October (realised volatility 9.8 robust standard deviations above its in-sample median, turnover 32 above), yet its spread rose only 7%. ISS had an unexplained volatility shock on 29 October.

*[Figure 1: report_fig1_intraday_tick.png.]*

## 3. Strategy and model development (~500 words)

Three building blocks feed one regression, which feeds a quoting rule. Every input for bar t is known before bar t starts.

**3.1 Informed trading: permanent price impact.** Following Hasbrouck (1991) we estimate a bivariate VAR in order time for each stock-day (rolling five days for the thinner Pandora, GN and ISS),

r_k = c + Σ_{i=0}^{p} b_i x_{k-i} + Σ_{i=1}^{p} a_i r_{k-i} + u_k,   x_k = d + Σ d_i x_{k-i} + Σ g_i r_{k-i} + v_k,

where r_k is the log mid return from order k to order k+1 and x_k the order sign. The permanent impact is the cumulative mid response twenty orders after a unit buy, with lag length chosen by BIC on in-sample days. Two versions enter the spread model: the previous day's estimate and a rolling estimate over the last 300 orders smoothed with an EWMA. As a cross-check we decompose each trade's effective half-spread into a realised half-spread and a price impact at five minutes.

**3.2 Volatility forecasts.** One-minute mid returns are deseasonalised by an in-sample time-of-day factor whose US-open bump follows the real US open. On the deseasonalised series we fit GARCH(1,1) and GJR-GARCH with in-sample parameters, a rolling 30-minute standard deviation, a 14-bar ATR, and a HAR model on bar, hour and day realised variance (Corsi, 2009). Each forecast for bar t is made at the end of bar t−1 and expressed as a ratio to the routine seasonal level, so the regression sees unusual volatility, not the open.

**3.3 Spread model.** For 5-minute bar t of stock i,

HalfSpread_it = α_i + β₁ Impact_roll,it + β₂ Impact_prevday,it + γ log(VolForecast_it / Seasonal_it) + time-of-day + activity controls + ε_it,

estimated separately for the constrained and unconstrained groups with panel Newey-West standard errors, on in-sample days only. The dependent variable is the time-weighted quoted half-spread in basis points; we also fit the effective half-spread and, for the constrained group, a fractional logit for the share of time the spread exceeds one tick. Activity controls are the lagged log trade count, log turnover and absolute order imbalance.

**3.4 Quoting rule.** At the start of each bar the simulated market maker posts a half-spread h_t equal to a forecast or a benchmark, floored at half a tick. A lit trade fills it if the aggressor paid at least h_t, up to the stock's median trade size. Revenue per filled share is h_t; the cost is the five-minute mid move after the trade; realised profit is the difference. Rules compared: in-sample average, yesterday's average, the last bar's spread, each model forecast, and the realised spread of the bar as an oracle.

**No look-ahead.** Lag lengths, hyper-parameters, seasonal factors and model coefficients are chosen on in-sample days. The out-of-sample window was touched once by the original model and twice more by the refinements in Section 6, which is why Section 4.4 adds a week that was touched once.

*[Figure 2: report_fig2_impact_by_stock.png, permanent impact vs effective half-spread.]*

## 4. Testing and validation (~350 words)

**4.1 Adverse selection dominates the spread.** The price impact at five minutes equals 91 to 115% of the effective half-spread in every stock; the realised half-spread is zero or negative (`05_impact_summary.csv`). The VAR impact ranges from 1.4 bps (Novo) to 7.2 bps (GN) and agrees with the decomposition. On Copenhagen, the spread is almost entirely a payment to informed flow, not a dealer margin.

**4.2 In-sample: volatility is the stronger driver, impact matters only where the tick does not bind.** In the unconstrained group a one within-stock standard deviation rise in the volatility forecast widens the half-spread by 0.28 bps (t = 5.5) and a one standard deviation rise in rolling impact by 0.14 bps (t = 2.4), against a median half-spread of 2.7 to 7 bps. In the constrained group only volatility is significant (0.09 bps, t = 3.7), and it works through the probability of a two-tick spread (fractional logit, t = 5.7). Results survive 15-minute bars, dropping the first fifteen minutes, log spreads, GJR instead of HAR volatility and longer Newey-West lags.

*[Table 2: report_table2_spread_regressions.csv.]*

**4.3 Out of sample (22 to 30 Oct): the drivers do not beat persistence.** The linear models beat the in-sample average and yesterday's spread (Diebold-Mariano −2.2 to −3.3 in the unconstrained group) but not the last bar's spread (DM −0.25 and 0.02). The LightGBM refinement is the only model that beats the last bar (RMSE 1.07 vs 1.32 bps constrained, 1.32 vs 1.66 unconstrained; DM −2.2 and −3.5), and Section 6 shows why.

*[Figure 3 / Table 3: report_fig3_oos_race.png, report_table3_oos_race.csv.]*

**4.4 Forward test (2 to 6 Nov): used once.** The week held the Q3 reports of Novo and Pandora (4 Nov) and GN (5 Nov). In the constrained group every model beat the last bar: RMSE 1.01 bps (stock-by-time controls), 1.00 (with the lagged spread) and 0.94 (LightGBM) against 1.42, Diebold-Mariano −2.6 to −2.8. In the unconstrained group the ranking from 4.3 repeated exactly: the linear driver models did not beat the last bar (1.77 to 1.81 against 1.85 bps, DM −0.4 to −0.6), the lagged-spread model did (1.58, DM −2.4) and LightGBM did most (1.46, DM −2.7). Constrained errors were 12 to 22% lower than in October; every unconstrained model was 7 to 29% worse, because two of its three stocks reported. Direction-of-change hit rates against the last bar are 61 to 67% for every model, so the gain is in size, not sign. On the report days the models failed together: on Pandora's every rule had an RMSE of 2.8 to 3.0 bps, and on Novo's the linear model (0.53) trailed the last bar (0.36) (tables `09_fwd_*.csv`).

## 5. Evaluation of results (~350 words)

**Economic size.** A one standard deviation volatility shock adds about 9% to a constrained stock's half-spread and about 10% to an unconstrained one; a one standard deviation impact shock adds about 5% in the unconstrained group. These are real but modest next to the intraday pattern itself: the open adds 70 to 150%.

**What the quoting simulation says.** No rule is profitable (`10_quoting_oos.csv`, `10_quoting_fwd.csv`). Quoting the market's own half-spread, whether yesterday's, the last bar's or the realised one, fills 46 to 62% of lit trades and loses 1.0 to 1.4 bps per fill in the constrained group and 5.9 to 8.8 bps in the unconstrained group out of sample (2.4 to 2.8 and 3.5 to 4.0 bps in the forward week), because the five-minute adverse-selection cost exceeds the half-spread earned. The model-based rules move this only at the margin: in the unconstrained group the lagged-spread model cuts the forward-week loss from 3.97 to 3.09 bps per fill by quoting wider (6.0 against 5.4 bps) and filling less (46 against 50%); in the constrained group all rules sit within 0.1 bps of each other because the tick fixes the quote. This is the mirror image of 4.1: if informed flow takes the whole spread, a quote at the market's spread cannot pay, however well it is forecast; only a wider quote, or one skewed away from the flow, can.

**Stability.** The volatility coefficient is stable across robustness specifications; the rolling impact coefficient is significant only in the unconstrained group and only in the specifications with stock-by-time controls. Spreads drifted lower in the out-of-sample week (Pandora from 9 to 10 bps to 6 to 7 bps), which is why beating yesterday's spread is the fairer benchmark and why the forward week matters.

**Event days.** On DSV's report day volatility rose by ten standard deviations and the spread by 7%. In the forward week Novo's report day had volatility 21 standard deviations above its in-sample mean and a spread 43% wider (4.2 against 3.0 bps); Pandora's had volatility 6.5 up and a spread 74% wider (15.6 against 9.0 bps); GN's had turnover 48 up and a spread inside its normal range (`03_daily_stock_day.csv`). The forward-filtered HMM called 93% of Pandora's and 99% of GN's bars stressed that day but only 28% of Novo's, where the volume came as a wall of small orders and the one-tick spread mostly held (`08_hmm_stressed_share.csv`).

## 6. Model and strategy refinement (~300 words)

**Stock-by-time controls.** Replacing shared time-of-day dummies and stock fixed effects with stock-by-coarse-bucket dummies lifts in-sample R² from 0.69 to 0.77 (constrained) and 0.66 to 0.70 (unconstrained), strengthens the rolling impact coefficient in the unconstrained group (0.23 bps per standard deviation, t = 3.9) and shrinks the volatility coefficient in the constrained group. Out of sample it changes RMSE by 0.02 to 0.03 bps, not significantly.

**LightGBM with SHAP.** A gradient-boosted model on the same no-look-ahead features plus the lagged spread, tuned by forward-chaining whole-day cross-validation on in-sample days, is the only model that beats the last-bar benchmark. SHAP values show why: the lagged spread carries 0.83 to 0.95 bps of mean absolute attribution, the stock 0.55, time of day 0.15 to 0.20, previous-day impact 0.12 to 0.19, rolling impact 0.09, and the volatility forecast only 0.02 (Figure 4). The gain is persistence modelled flexibly, not a better reading of the drivers. A linear model with the same lagged spread closes about half of the gap.

**Regimes.** A two-state Gaussian HMM on deseasonalised volatility, forward-filtered so the state at t uses bars up to t only, splits bars into calm and stressed. In the constrained group volatility widens spreads only in the stressed state (0.46 bps per unit, t = 3.1, against −0.07 in calm; difference t = 3.0). In the unconstrained group the effect is present in both states and larger under stress (Figure 5). The driver that matters most is the one that matters when it matters.

*[Figures 4 and 5: report_fig4_shap.png, report_fig5_hmm_regimes.png.]*

## 7. Critical reflection and recommendations (~350 words)

**Strengths.** Trade signing validated against full quote streams (98.6 to 99.8% agreement). Every regressor known before the bar starts. The tick constraint as a natural experiment. An out-of-sample window plus a week used once, with the number of looks at each stated. A negative result stated as such: the drivers are real but do not forecast five minutes ahead better than the spread itself.

**Limitations, with evidence.** Quotes are observed only at trades, so time-weighted spreads carry a sampling bias of about 5% on ISS and 16% on Novo in the sampled hour (`02_quote_validation.csv`). Only the primary lit venue is observed; Cboe Europe, Aquis and Turquoise quotes are missing. Twenty-two plus five days is a short sample with three report days in the last week. Post-trade data on Nasdaq Copenhagen is anonymous (every broker field reads 001), so a PIN-style measure from broker identity is not possible. The quoting simulation assumes the market maker is first in the queue when its quote is at least as good and charges the full five-minute impact; both lean against the market maker.

**Three recommendations for a real implementation.**
1. Subscribe to the full L1 quote stream rather than trade-attached quotes. The bias is small in thin stocks and large in the most liquid one, which is where a market maker trades most.
2. Consolidate across venues. With the primary venue only, the competition for the quote is understated and the fill model is optimistic.
3. Run the quoting rule live without executing for several weeks and use the regime state to decide when to widen: volatility matters in stress, and quoting at the touch loses to informed flow even in calm markets.

---

## Appendix A. AI transparency statement (not counted)
[To write jointly. Tools used by each member, for what (code generation, debugging, drafting, editing), how outputs were verified (unit tests, quote-stream validation, cross-checks against the spreadsheets and the FactSet screens), what was not AI (research design, model choice, interpretation).]

## Appendix B. Contribution statement (not counted)
[Table: member, role, deliverables, evidence (scripts, commits, report sections).]

## References
Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
Glosten, L. R. and Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71–100.
Hasbrouck, J. (1991). Measuring the information content of stock trades. *Journal of Finance*, 46(1), 179–207.
Ho, T. and Stoll, H. R. (1981). Optimal dealer pricing under transactions and return uncertainty. *Journal of Financial Economics*, 9(1), 47–73.
Lee, C. M. C. and Ready, M. J. (1991). Inferring trade direction from intraday data. *Journal of Finance*, 46(2), 733–746.
Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
