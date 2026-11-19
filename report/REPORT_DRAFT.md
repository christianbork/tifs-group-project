# What drives bid-ask spreads on Nasdaq Copenhagen: informed trading or volatility?

**FINS5566 Trading in Financial Securities, T3 2026. Group project.** Christian Bork, Emanuel Annvik, Mathilde Sørensen.

**Key findings.**

- Adverse selection sets the level of the spread; volatility drives how it moves through the day.
- Both drivers carry information out of sample (Clark-West), but five minutes ahead neither beats the spread's own last value by enough to matter.
- No quoting rule makes money. An inventory limit removes over 90% of the P&L risk and, in the unconstrained stocks, about a third of the loss per fill.

## 1. Introduction and research objective

A market maker gets hurt in two ways. It can end up holding shares it never wanted while the price moves, and it can trade with someone who knows more than it does. Ho and Stoll (1981) build the spread on the first problem, so it should widen with volatility. Glosten and Milgrom (1985) build it on the second, so it should widen with informed order flow. We wanted to know which one a market maker in Copenhagen should actually watch.

We took six Nasdaq Copenhagen stocks through October 2026 and asked three things. Which explains the intraday spread better, the permanent price impact of order flow or a volatility forecast? Does either forecast the spread five minutes ahead better than the spread's own last value? And is any of it worth money to someone posting quotes?

The tick size gave us a natural experiment. Novo Nordisk, DSV and ISS sit at the one-tick minimum spread more than half the time, so there the drivers can only show up as a higher chance of a second tick. Vestas, Pandora and GN have room to move.

## 2. Data and background analysis

The data are FactSet Time and Sales exports with the best bid and ask attached to every print. We keep lit continuous-session trades with a valid quote: about 720,000 trades over 22 days (Table 1). We estimate on 1 to 21 October and test on 22 to 30 October. A second test week, 2 to 6 November, was downloaded after the models were fixed.

*[Table 1: report_table1_data_summary.csv.]*

Trades are signed with the quote rule (94 to 98% of trades), then the tick test (Lee and Ready, 1991). Prints sharing a timestamp are treated as one aggressive order walking the book. Against four full quote streams the sign agrees in at least 98.6% of cases.

The first fifteen minutes are a different market: spreads are about twice their midday level and volatility three times (Figure 1). Novo also jumps at the US open, which moved from 15:30 to 14:30 Copenhagen time in the last week of October because Europe changed its clocks a week before America. Our time-of-day controls follow the real US open.

*[Figure 1: report_fig1_intraday_tick.png.]*

Measured volatility depends on how often you look. Sample the mid every few seconds and you pick up quotes flickering between ticks, which is not information (Andersen et al., 2000). Across nine sampling intervals and every hour of the day, volatility flattens between 2 and 15 minutes (Figure 2). At the 1-minute interval our models use, 20 to 27% of the variance in four of the six stocks is this noise.

*[Figure 2: 11_vol_surface.png.]*

Two days stand out. On DSV's Q3 report day volatility was ten standard deviations above normal, yet the spread was only 6% wider. On 29 October ISS fell 3.5% in fifteen minutes after Danish media reported that a Hong Kong inquiry partly blamed an ISS subsidiary for the Wang Fuk Court fire (MarketWire, 2026), and its half-spread went from 4 to 19 bps.

## 3. Strategy and model development

One rule runs through everything: whatever we use for bar t is known before bar t starts.

**Informed trading.** Following Hasbrouck (1991) we estimate a VAR of mid returns and order signs in order time, per stock and day. The permanent impact is how far the mid has moved twenty orders after a buy. The spread model gets yesterday's estimate and a rolling one over the last 300 orders. As a cross-check we split every trade's effective half-spread into what the liquidity provider keeps after five minutes and what the price takes back.

**Volatility.** We deseasonalise 1-minute returns and fit GARCH (Bollerslev, 1986), GJR-GARCH (Glosten et al., 1993), two rolling benchmarks and a HAR model (Corsi, 2009). HAR forecasts best out of sample and is our volatility input.

**Spread model.** For 5-minute bar t of stock i:

HalfSpread_{it} = α_{i} + β_{1} Impact_{roll,it} + β_{2} Impact_{prevday,it} + γ log(VolForecast_{it} / Seasonal_{it}) + time-of-day + activity controls + ε_{it},

estimated per group on in-sample days. Standard errors are panel Newey and West (1987); because the residuals stay correlated for long, we only call a driver significant if it also survives longer lags, Driscoll and Kraay (1998) errors and clustering by day.

**Quoting rule.** To put a price on the forecasts we simulate a market maker. Each bar it posts a half-spread from a forecast or a benchmark, snapped to the tick grid. An aggressive order fills the quote if its first print paid at least that much. The market maker earns the half-spread and pays whatever the mid does over the next five minutes. Section 6 adds inventory control and selective quoting.

## 4. Testing and validation

**Adverse selection takes most of the spread.** This was the first surprise. Five minutes after a trade the price has moved against the liquidity provider by 91 to 115% of the effective half-spread, in every stock (Figure 3). What is left for whoever posted the quote is at most 0.6 bps. The five-minute move also holds transitory and inventory effects, so this is adverse selection in a broad sense.

*[Figure 3: report_fig2_impact_by_stock.png.]*

**In-sample, volatility is the stronger driver.** In the unconstrained group a one standard deviation rise in the volatility forecast widens the half-spread by 0.30 bps and the same rise in rolling impact by 0.15 bps (Table 2). In the constrained group only volatility matters, and it works through the chance of a two-tick spread. Volatility survives every standard-error choice. Impact is shakier: its t-statistic drops to 1.8 with long lags or day clusters. Read together with Figure 3: adverse selection sets the level of the spread, and volatility drives its variation through the day.

*[Table 2: 08_r1_coefficients.csv.]*

**Out of sample, the drivers do not beat persistence.** The honest benchmark is the spread five minutes ago, and it is hard to beat. Our linear models beat yesterday's spread but not the last bar (Table 3). Adding the lagged spread does (1.21 against 1.32 bps in the constrained group, 1.45 against 1.66 in the unconstrained group), and LightGBM does best (1.08 and 1.32).

**The November week repeated the ranking.** It was a harder week, with Q3 reports from Novo, Pandora and GN. In the constrained group every model beat the last bar, but mostly through ISS: for Novo, the most liquid stock, the linear models were significantly worse than the last bar (DM +2.3 to +3.2). In the unconstrained group only the model with the lagged spread and LightGBM won (Figure 4).

*[Table 3: 08_oos_rmse.csv, 09_fwd_rmse.csv.]*

*[Figure 4: 09_fwd_rmse.png.]*

**Do the drivers add anything once the last spread is known?** We held everything else fixed and added volatility, impact, or both (Table 4). On plain accuracy the answer is no for the linear model: no driver gives a significant Diebold and Mariano (1995) gain in either week. But that test is unfair to the larger of two nested models, which pays for estimating extra coefficients. The Clark and West (2007) correction says the drivers do carry information, in both weeks and both groups. So the information is there, but fifteen days is too little to estimate it sharply enough to help. LightGBM shows the same thing as instability: in the unconstrained group its drivers helped significantly in October (DM −3.9) and hurt as much in November (+3.8). And in October rolling impact alone forecast as well as both drivers together (Clark-West z = −0.1 for adding volatility), although volatility was the stronger driver in-sample. In-sample significance is not forecasting value.

*[Table 4: 14_nested_driver_rmse.csv, 14_nested_driver_clark_west.csv.]*

## 5. Evaluation of results

The effects are real and small. A one standard deviation volatility shock adds about 3% to the half-spread of a constrained stock and 6% for an unconstrained one. The opening bell adds 70 to 150%.

No quoting rule makes money (Table 5, Figure 5). A market maker that matches the market's last spread loses 0.5 bps per fill in the constrained group and 3.8 bps in the unconstrained group out of sample. Why lose 3.8 bps when the market's liquidity providers roughly break even? Selection on the fill. Our quote is hit only when the aggressor paid at least as much, so fills cluster where the spread was wide and the flow toxic. In the unconstrained group in October, the orders that filled us had paid 6.0 bps against 4.6 bps for the average order, and we earn only our own 3.4 bps. The markout curve shows the market maker only breaks even if it can unwind within five seconds, and only in the unconstrained group (Figure 5).

*[Table 5: 10_quoting_oos.csv, 10_quoting_fwd.csv.]*

*[Figure 5: 15_markout_curve.png.]*

The model rules lose less, mostly because they quote wider. Is there anything beyond width? In October, yes. With every rule scaled to the same average width, the rule using the volatility and impact forecasts beat the same rule without them on all seven days (t = 3.2), also with widths matched inside the test week (t = 2.9). In November, no. It was quoting 7% wider in a volatile week, and with widths matched the advantage is gone (t = 0.4). LightGBM, the best forecaster, never beat the simple rule.

## 6. Model and strategy refinement

**Stock-by-time controls.** Letting every stock have its own time-of-day pattern raises the in-sample R² from 0.69 to 0.77 and 0.66 to 0.70, and makes rolling impact stronger in the unconstrained group (t = 3.9). Out of sample it barely matters (Table 3).

**LightGBM with SHAP.** The boosted model (Ke et al., 2017) forecasts best, and the reason is a little deflating. The lagged spread dominates its SHAP values (Lundberg and Lee, 2017) and volatility barely registers (Figure 6). Retraining without the lagged spread raises the error by 15%; without either driver, by under 1%. LightGBM wins by modelling persistence more flexibly, not by reading the drivers better.

*[Figure 6: report_fig4_shap.png.]*

**Regimes.** A two-state HMM on volatility (Hamilton, 1989), filtered forward so it never sees the future, splits the bars into a low and a high state. In the constrained group volatility widens spreads only in the high state (t = 3.7, against 0.4 in the low state).

**Refining the quoting strategy.** We chose the parameters on 1 to 21 October and scored each test week once (Table 6, Figure 7). Inventory control follows Ho and Stoll (1981) and Avellaneda and Stoikov (2008): skew the quotes against inventory, and stop quoting the side that would push the position past a limit. In-sample, a hard limit of one quote size beat every linear skew on mean daily P&L minus one standard deviation. We did not use the Sharpe ratio, because when the mean is negative it rewards taking more risk. Selective quoting pulls both quotes when the volatility or impact forecast is above its in-sample 80th percentile.

*[Table 6: 15_refined_quoting.csv, 15_refined_tests.csv.]*

*[Figure 7: 15_refined_pnl.png.]*

Most of the unmanaged rule's P&L was an accidental directional bet. It ran up to 18m DKK (October) and 39m DKK (November) in the constrained stocks, and in the constrained group the drift made it look profitable. The limit cuts daily P&L volatility by more than 90% and the largest position to about 0.1m DKK. In the unconstrained group it also cuts the loss per fill, from 3.8 to 2.4 bps in October and from 2.6 to 1.7 in November (daily-total t = 2.5 and 2.4, every day better), because refusing to keep buying into one-sided flow is itself a filter for informed trading. Selective quoting on top cuts the daily loss by another 40 to 85%, mostly by trading less. No version makes money.

## 7. Critical reflection and recommendations

**What holds up.**

- Checked data: trade signs agree with four full quote streams in at least 98.6% of cases, 28 unit tests pass, and every table reproduces from a clean checkout.
- The models were frozen before the November data existed, and the quoting parameters were chosen on in-sample days only.
- We ran the fair nested test as well as the unfair one, and report results we did not hope for: the drivers are real but do not beat the spread you can already see, and no quoting rule makes money.

**Limitations.**

- We only see quotes when a trade happens, so our spreads are off by about 5% for ISS and 18% for Novo in the hour we could check. We see one venue.
- Twenty-seven days is short, and three of the last five were report days. Both test weeks were scored more than once, though never fitted on, and the timing result holds in one of the two.
- The simulation puts our market maker first in the queue whenever its price is good enough, which flatters it, and ignores fees and rebates.
- Our volatility input is measured at one minute, where 20 to 27% of the variance is bid-ask noise (Figure 2), which may be why volatility looks weaker out of sample.

**Recommendations.**

1. **Buy the full order book.** Trade-attached quotes are worst where a market maker trades most (18% error for Novo), and selection on the fill depends on queue position, which only the book shows. A fill model with queue position is the first thing to rebuild.
2. **Run the market maker as a risk-limited, selective quoter and judge it at the horizon it can unwind.** The inventory limit cut P&L risk by over 90% and improved the loss per fill in both weeks in the unconstrained group. The markout curve says the edge exists only within seconds, so pair the limit with a fast hedge in OMXC25 index futures instead of waiting for the other side.
3. **Replace the 5-minute spread forecast with a fast toxicity signal for pulling quotes, and paper-trade it live.** The Clark-West tests show the information is real but too thin to estimate at five minutes, and LightGBM's driver gains flipped sign between weeks. An order-imbalance or volume-synchronised measure such as VPIN (Easley et al., 2012), updated every few orders and forward-tested without execution, is the natural next step.

## References
Andersen, T. G., Bollerslev, T., Diebold, F. X. and Labys, P. (2000). Great realizations. *Risk*, 13(3), 105–108.
Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217–224.
Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
Clark, T. E. and West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291–311.
Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
Driscoll, J. C. and Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4), 549–560.
Easley, D., López de Prado, M. M. and O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies*, 25(5), 1457–1493.
Glosten, L. R., Jagannathan, R. and Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.
Glosten, L. R. and Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71–100.
Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.
Hasbrouck, J. (1991). Measuring the information content of stock trades. *Journal of Finance*, 46(1), 179–207.
Ho, T. and Stoll, H. R. (1981). Optimal dealer pricing under transactions and return uncertainty. *Journal of Financial Economics*, 9(1), 47–73.
Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154.
Lee, C. M. C. and Ready, M. J. (1991). Inferring trade direction from intraday data. *Journal of Finance*, 46(2), 733–746.
Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.
MarketWire (2026). ISS får del af skylden for dødsbrand. Proinvestor.com, 29 October 2026.
Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
