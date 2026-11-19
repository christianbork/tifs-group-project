# What drives bid-ask spreads on Nasdaq Copenhagen: informed trading or volatility?

**FINS5566 Trading in Financial Securities, T3 2026. Group project.** Christian Bork, Emanuel Annvik, Mathilde Sørensen.

**Key findings.**

- Adverse selection sets the spread's level; volatility drives its variation through the day.
- The drivers carry information out of sample, but do not beat the last spread five minutes ahead.
- No quoting rule makes money; an inventory limit removes over 90% of the P&L risk.

## 1. Introduction and research objective

A market maker gets hurt in two ways: holding inventory while the price moves, and trading with someone better informed. Ho and Stoll (1981) build the spread on the first, so it should widen with volatility; Glosten and Milgrom (1985) on the second, so it should widen with informed order flow. Which one should a market maker in Copenhagen watch?

For six Nasdaq Copenhagen stocks we ask: which explains the intraday spread better, the permanent price impact of order flow or a volatility forecast? Does either forecast the spread five minutes ahead better than its own last value? And is any of it worth money to someone posting quotes?

The tick size gives a natural experiment. Novo Nordisk, DSV and ISS sit at the one-tick minimum more than half the time (constrained group); Vestas, Pandora and GN have room to move.

## 2. Data and background analysis

FactSet Time and Sales exports carry the best bid and ask at every print. We keep lit continuous-session trades: about 720,000 over 22 days (Table 1). We estimate on 1 to 21 October and test on 22 to 30 October; a second test week, 2 to 6 November, was downloaded after the models were fixed.

*[Table 1: report_table1_data_summary.csv.]*

Trades are signed with the quote rule, then the tick test (Lee and Ready, 1991); prints sharing a timestamp form one aggressive order. Against four full quote streams the sign agrees in at least 98.6% of cases.

The first fifteen minutes are a different market: spreads twice their midday level, volatility three times (Figure 1). Novo also jumps at the US open, which moved to 14:30 Copenhagen time in late October; our time-of-day controls follow it.

*[Figure 1: report_fig1_intraday_tick.png.]*

Measured volatility depends on how often you look: at a few seconds it picks up quotes flickering between ticks (Andersen et al., 2000). It flattens between 2 and 15 minutes (Figure 2); at the 1-minute interval our models use, 20 to 27% of the variance in four stocks is noise.

*[Figure 2: 11_vol_surface.png.]*

On DSV's Q3 report day volatility was ten standard deviations above normal, yet the spread only 6% wider. On 29 October ISS fell 3.5% in fifteen minutes after Danish media reported a Hong Kong fire inquiry blaming an ISS subsidiary (MarketWire, 2026); its half-spread went from 4 to 19 bps.

## 3. Strategy and model development

Whatever we use for bar t is known before bar t starts.

**Informed trading.** Following Hasbrouck (1991), a VAR of mid returns and order signs per stock and day gives the permanent impact: the mid move twenty orders after a buy. The spread model gets yesterday's estimate and a rolling one over the last 300 orders. As a cross-check we split each trade's effective half-spread into what the liquidity provider keeps after five minutes and what the price takes back.

**Volatility.** On deseasonalised 1-minute returns, a HAR model (Corsi, 2009) forecasts better out of sample than GARCH, GJR-GARCH and rolling benchmarks.

**Spread model.** For 5-minute bar t of stock i:

HalfSpread_{it} = α_{i} + β_{1} Impact_{roll,it} + β_{2} Impact_{prevday,it} + γ log(VolForecast_{it} / Seasonal_{it}) + time-of-day + activity controls + ε_{it},

estimated per group. A driver counts as significant only if it survives Newey and West (1987) errors with several lags, Driscoll and Kraay (1998) errors and day clusters.

**Quoting rule.** A simulated market maker posts each bar a half-spread from a forecast or benchmark, on the tick grid. An aggressive order fills it if its first print paid at least that much; the market maker earns the half-spread and pays the mid move over the next five minutes.

## 4. Testing and validation

**Adverse selection takes most of the spread.** Five minutes after a trade the price has moved against the liquidity provider by 91 to 115% of the effective half-spread in every stock, leaving at most 0.6 bps (Figure 3). The five-minute move also holds transitory effects, so this is adverse selection in a broad sense.

*[Figure 3: report_fig2_impact_by_stock.png.]*

**In-sample, volatility is the stronger driver.** In the unconstrained group a one-standard-deviation rise in the volatility forecast widens the half-spread by 0.30 bps, rolling impact by 0.15 bps (Table 2). In the constrained group only volatility matters, through the chance of a two-tick spread. Volatility survives every standard-error choice; impact falls to t = 1.8. So adverse selection sets the spread's level, and volatility drives its variation.

*[Table 2: 08_r1_coefficients.csv.]*

**Out of sample, nothing clearly beats persistence.** The linear models beat yesterday's spread but not the last bar (Table 3). Adding the lagged spread does (1.21 against 1.32 bps constrained, 1.45 against 1.66 unconstrained); LightGBM does best (1.08 and 1.32). November repeated the ranking (Figure 4), but the constrained win is mostly ISS: for Novo the linear models lost to the last bar (DM +2.3 to +3.2).

*[Table 3: 08_oos_rmse.csv, 09_fwd_rmse.csv.]*

*[Figure 4: 09_fwd_rmse.png.]*

**Do the drivers add anything once the last spread is known?** On Diebold and Mariano (1995) tests, no (Table 4). But that test penalises the larger nested model; the Clark and West (2007) correction says the drivers carry information in both weeks and groups, just too little to estimate sharply from fifteen days. LightGBM shows this as instability: its drivers helped in October (DM −3.9, unconstrained) and hurt in November (+3.8). And in October impact alone forecast as well as both drivers, although volatility led in-sample.

*[Table 4: 14_nested_driver_rmse.csv, 14_nested_driver_clark_west.csv.]*

## 5. Evaluation of results

The effects are real but small: a one-standard-deviation volatility shock adds about 3% to the half-spread (constrained) and 6% (unconstrained); the opening bell adds 70 to 150%.

No quoting rule makes money. Matching the market's last spread loses 0.5 bps per fill (constrained) and 3.8 bps (unconstrained) in October (Table 5). Why more than the market's liquidity providers? Selection on the fill: our quote is hit only when the aggressor paid at least as much, so fills cluster where the flow is toxic. Unconstrained, our fills had paid 6.0 bps against 4.6 for the average order, yet we earn our own 3.4. The market maker breaks even only if it unwinds within five seconds (Figure 5).

*[Figure 5: 15_markout_curve.png.]*

Model rules lose less, mostly by quoting wider. At equal average width, the rule with the volatility and impact forecasts beat the same rule without them on all seven October days (t = 3.2; 2.9 with widths matched in the test week), but not in November (t = 0.4). LightGBM never beat the simple rule.

## 6. Model and strategy refinement

**Stock-by-time controls** raise the in-sample R² from 0.69 to 0.77 (constrained) and 0.66 to 0.70, but barely matter out of sample (Table 3).

**LightGBM with SHAP.** The lagged spread dominates the best forecaster; volatility barely registers (Figure 6). Dropping the lagged spread raises the error 15%, dropping both drivers under 1%: LightGBM wins by modelling persistence, not the drivers.

*[Figure 6: report_fig4_shap.png.]*

**Regimes.** In a forward-filtered two-state volatility HMM, volatility widens constrained spreads only in the high state (t = 3.7 against 0.4).

**Refining the quoting strategy.** Two refinements, tuned on 1 to 21 October and scored once per test week (Table 5, Figure 7). Inventory control (Ho and Stoll, 1981; Avellaneda and Stoikov, 2008) skews quotes against inventory and stops quoting the side that would breach a limit; a hard limit of one quote size beat every skew on mean daily P&L minus one standard deviation (Sharpe ratios reward risk when means are negative). Selective quoting pulls both quotes when the volatility or impact forecast exceeds its in-sample 80th percentile.

*[Table 5: 15_refined_quoting.csv.]*

*[Figure 7: 15_refined_pnl.png.]*

The unmanaged rule's P&L was mostly an accidental directional bet: positions up to 39m DKK, which made the constrained group look profitable. The limit cuts daily P&L volatility by over 90% and, unconstrained, the loss per fill from 3.8 to 2.4 bps (October) and 2.6 to 1.7 (November; daily t = 2.5 and 2.4), because refusing to keep buying into one-sided flow filters informed trading. Selective quoting cuts the daily loss by another 40 to 85%, mostly by trading less. Nothing makes money.

## 7. Critical reflection and recommendations

**Strengths.**

- Signs validated against quote streams (≥98.6%), 28 unit tests, every table reproducible.
- Models frozen before November existed; quoting parameters tuned in-sample only.
- Fair nested tests, and unwelcome results reported: real drivers, no forecasting edge, no profitable rule.

**Limitations.**

- Trade-attached quotes misstate spreads by about 5% (ISS) and 18% (Novo); one venue only.
- Twenty-seven days, three of the last five report days; both test weeks scored more than once (never fitted); the timing result holds in one week.
- The simulation puts us first in the queue and ignores fees.
- 1-minute volatility is a fifth to a quarter noise (Figure 2).

**Recommendations.**

1. **Buy the full order book.** Quote errors peak where a market maker trades most, and selection on the fill depends on queue position, which only the book shows.
2. **Quote risk-limited and selectively, unwinding fast.** The limit cut P&L risk by over 90% and unconstrained losses per fill in both weeks; the markout curve says any edge lasts seconds, so hedge in OMXC25 futures rather than wait.
3. **Pull quotes on a fast toxicity signal and paper-trade it live.** Clark-West says the information exists but is too thin at five minutes, and LightGBM's driver gains flipped sign; an order-imbalance signal updated every few orders, forward-tested without execution, is next.

## References
Andersen, T. G., Bollerslev, T., Diebold, F. X. and Labys, P. (2000). Great realizations. *Risk*, 13(3), 105–108.
Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217–224.
Clark, T. E. and West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291–311.
Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
Driscoll, J. C. and Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4), 549–560.
Glosten, L. R. and Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71–100.
Hasbrouck, J. (1991). Measuring the information content of stock trades. *Journal of Finance*, 46(1), 179–207.
Ho, T. and Stoll, H. R. (1981). Optimal dealer pricing under transactions and return uncertainty. *Journal of Financial Economics*, 9(1), 47–73.
Lee, C. M. C. and Ready, M. J. (1991). Inferring trade direction from intraday data. *Journal of Finance*, 46(2), 733–746.
MarketWire (2026). ISS får del af skylden for dødsbrand. Proinvestor.com, 29 October 2026.
Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
