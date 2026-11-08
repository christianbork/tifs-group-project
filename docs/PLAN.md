# Group Project — Process, Methods & Coding Plan (updated 2-Nov-2026)

Repo: github.com/christianbork/tifs-group-project (private). Decisions log in `CLAUDE.md`.

## Research question
What drives bid-ask spreads on Nasdaq Copenhagen stocks, **informed trading (adverse selection) or volatility (inventory risk)**, and can a market maker use each to **forecast spreads out of sample**? Framing: the "risk adjustment function" of the Week 6 market-making diagram. Appendix E (adverse selection, volatility), linked to Appendix C.

## 1. Data (done)
- FactSet trade exports (bid/ask attached to each trade): Novo, DSV, Vestas, Pandora, GN, ISS; 22 days (1–30 Oct 2026), 08:45–17:20; ~720k continuous lit trades. All 132 stock-days complete.
- In-sample 1–21 Oct (15 days); out-of-sample 22–30 Oct (7 days). DSV 21-Oct = earnings (event flag).
- Assumptions/limitations:
  - Copenhagen exchange only; other venues (Cboe Europe, Aquis, …) are not observed.
  - October 2026 is assumed representative.
- Export cap ≈ 25k rows with silent truncation of the morning → busy days split into `-partN`.

## 2. Cleaning (done, `01_load_clean.py`)
- Drop IND (indicative) rows. Label prints: lit / auction / close / dark (MPM, DCSE) / off-book (XLT, CPX, SSL, VWAP…) / other venue (MCSE).
- Keep continuous-session (09:00–17:00) lit trades with a valid quote.
- Stitch parts, removing duplicates only in the overlap. Flag truncation.
- Full-day medians: spread Novo 2.0 bps, DSV 7.1, Vestas 5.3, Pandora 7.6, GN 14.0, ISS 6.9.

## 3. Trade signing (done, `02_sign_trades.py`)
- Quote rule on the **trade-attached** mid, then the tick test → ~0% unsigned (quote rule handles 94–98% of trades).
- Attached quote beats lagged quotes on % of trades outside the quote (e.g. Novo 18.3% vs 25.2% / 33.1%).
- Same-timestamp fills = one aggressive order walking the book: ~2.3 fills/order; 98–99% one-sided. The outside rate rises with fill position (Novo 11% → 36%), so outside prints are genuine fills further into the book, not bad data.
- ISS 30-Oct full quote stream validation:
  - attached quote = prevailing quote 95.1% of the time;
  - sign agreement 99.8%;
  - spread 11.95 bps time-weighted vs 11.33 bps trade-weighted (~5% sampling bias).

## 4. Background analysis (done, `03_background.py`)
- **U-shape mainly at the open:** first 15 min spread 1.7–2.5× midday and 1-min vol 2.8–3.6× midday; both settle by ~10:00. After that, spreads stay flat even when vol moves (Novo at the US open: vol ~3×, spread ~1.2×). Last 15 min: spreads ~0.85× midday.
- **Tick constraint (% of time spread = 1 tick):** ISS 62%, Novo 58%, DSV 53%, Vestas 22%, GN 8%, Pandora 4%. Tick size in bps: Novo 2.0, ISS 6.8, DSV 4.2, Vestas 2.6, GN 4.7, Pandora 2.4.
- **US-open clock shift:** Novo turnover spike at 15:30 CPH on 1–23 Oct (3.4× normal) moves to **14:30** on 26–30 Oct (3.0×), because EU DST ended 25-Oct and US DST ends 1-Nov.
- **Big vol days barely move spreads:** DSV 21-Oct earnings had RV z≈9.8 and turnover z≈32, but the spread rose only ~7%. ISS 29-Oct (OOS) had RV z≈15 and turnover z≈18 (news event? check).
- **OOS drift:** spreads lower in the OOS period (Pandora ~9–10 → 6–7 bps; DSV and Vestas eased).
- **Interval activity at 5 min:** Novo/DSV/Vestas ~0% empty; Pandora 6% with <5 trades; GN/ISS ~28% with <5 trades (~8% empty).

## 4b. Design decisions after script 03 (agreed 2-Nov)
1. **Tick constraint as a natural experiment:** constrained group = Novo, DSV, ISS; unconstrained = Vestas, Pandora, GN.
   - Estimate the spread model by group (or with group interactions). Hypothesis: impact/vol effects show up mainly in the unconstrained group.
   - For the constrained group, also model the effective half-spread and P(spread > 1 tick).
2. **Time-of-day controls follow the US open:** Copenhagen clock-time dummies (open/close) PLUS a minutes-since-US-open variable/dummies keyed to the real US open (15:30 CPH up to 23-Oct, 14:30 CPH 26–30 Oct).
3. **Intervals:** 5-min main; 15-min robustness (GN/ISS are thin at 5-min).
4. **Keep the open, with time-of-day controls,** so the vol coefficient picks up unusual volatility, not the routine open pattern. Robustness: drop 09:00–09:15.
5. **Benchmarks:** in-sample stock average AND yesterday's average spread (persistence). Beating yesterday's spread is the fairer test given the OOS drift.
6. **Event flags:** DSV 21-Oct (IS, earnings), ISS 29-Oct (OOS, news; = OOS stress test).

## 4c. Bars (next, `04_bars.py`)
- 1-min and 5-min bars per stock (and 15-min for robustness), continuous session, saved to data/processed/bars_<freq>.parquet (all stocks, one file per frequency).
- Columns:
  - time-weighted quoted half-spread (bps, from trade-attached quotes carried forward within the day);
  - trade-weighted effective half-spread (bps);
  - % of time at 1 tick; mid at interval end;
  - 1-min mid log returns and realised vol per interval;
  - trade count, order count, volume, DKK turnover, signed volume and order-flow imbalance;
  - clock time and minute-of-day;
  - minutes since the US open (15:30 CPH up to 23-Oct, 14:30 CPH 26–30 Oct);
  - group (constrained/unconstrained), sample (IS/OOS), event flag;
  - number of trades (to flag thin intervals).
- Reads from the signed_<stock>.parquet files. Scripts 05–08 read the bars.

## 5. Informed trading (`05_price_impact.py`)
- **Hasbrouck (1991) VAR in trade/order time:** mid return + signed order flow (sign, signed volume); permanent impact = cumulative IRF after ~10–20 events.
  - Per stock-day for Novo/DSV/Vestas; rolling 5-day window for Pandora/GN/ISS.
- Cross-check: effective half-spread = realised half-spread + price impact (mid at +1 / +5 min).
- Assumptions:
  - The mid proxies the efficient price.
  - Impact is linear and stable within a day.
  - **Previous-day impact** is the predictor (no look-ahead).

## 6. Volatility (`06_volatility.py`)
- 1-min mid returns, deseasonalised by an in-sample time-of-day factor; overnight return excluded.
- GARCH(1,1) and GJR-GARCH. Benchmarks: rolling 30-min std, ATR, HAR-RV (Corsi 2009).
- OOS evaluation vs realised 5-min volatility: RMSE, QLIKE, Mincer–Zarnowitz.

## 7. Spread model (`07_spread_model.py`)
- Half-spread(bps, 5-min) = a + b·impact(t−1 day) + c·vol forecast + Copenhagen time-of-day + US-open controls + activity controls + stock fixed effects. Newey–West SEs; estimated on in-sample days.
- **Estimated separately for the constrained (Novo, DSV, ISS) and unconstrained (Vestas, Pandora, GN) groups**, or pooled with group interactions.
- Dependent variables: quoted half-spread (main); effective half-spread; for the constrained group, also P(spread > 1 tick) (logit/probit).
- OOS race:
  - in-sample-average benchmark;
  - yesterday's-average-spread benchmark;
  - vol-only, impact-only, combined;

  compared on RMSE + Diebold–Mariano, by group.
- Robustness: 15-min intervals; drop 09:00–09:15; the event-day flags.

## 8. Refinement (`08_refine.py`)
- Econometric options: GJR vs GARCH, HAR, trade-size effects, event-day flag, stock-specific slopes (large vs mid cap), open/close buffers.
- **ML refinement (main):** LightGBM / XGBoost spread forecast using the same no-look-ahead features:
  - impact and vol forecast;
  - time of day and stock;
  - lagged spread, volume, trade count, order-flow imbalance.

  It is the 5th model in the OOS race. SHAP values show which driver the forecasts rely on and the interactions (e.g. vol × open, impact × thin stock). ~8.6k in-sample training rows.
- **Robustness:** 2-state HMM (or clustering) on vol/volume/spread → calm vs stressed regimes; do the drivers change by regime? DSV earnings day = stress case.
- Optional: Lasso / elastic net to choose controls.
- Ground rules:
  - Validate in blocks of whole days, in time order; never shuffle.
  - Tune on in-sample days only; touch OOS once.
  - Present as original vs refined, compared.
- Not doing: deep learning for volatility (too little data; GARCH/HAR are the right benchmarks); ML trade classification (already 99.8% agreement).
- Extra packages: lightgbm, shap, hmmlearn.

## 9. Critical reflection — top-3 recommendations
1. Full L1 quote stream: removes the trade-time sampling bias (ISS evidence ~5%). Blocked by the 25k-row export cap (~220 exports needed).
2. Consolidated data across all European venues.
3. Longer sample + live forward test of a volatility- and impact-adjusted quoting rule.

Others: order-book depth / broker IDs for PIN; adding inventory.

## Outcomes
- A ranking of informed trading vs volatility as spread drivers across large and mid caps.
- Whether each forecasts spreads OOS (linear and ML).
- A practical quote-widening rule for market makers.

## Open decisions
- Main informed-trading measure: Hasbrouck VAR (recommended), with the decomposition as a cross-check.
- Check what happened to ISS on 29-Oct (results/news?).
- Due date: likely Fri 20-Nov-2026, 9pm → confirm. Convenor approval for the Appendix E angle.

## 10. v2 review round (8-Nov-2026)
Review: `docs/REVIEW_V1.md`; full list of changes and before/after numbers: `CHANGELOG.md`.
- **New scripts:** `09_ae_diagnostics.py` (ADF, Granger-style Wald, QLR, AE appendix table) and
  `10_quote_rule.py` (quote-widening rule scored by markout P&L; ISS 29-Oct event study).
- **Design decisions:**
  - Activity controls lagged within the day; the lagged spread stays "last observed" (includes the previous close).
  - Hasbrouck VAR p = 20 (BIC never stops falling at this n; impact stable from p = 20).
  - Main SEs stay Newey–West (AE lag rule, L = 7) but a driver is only called robust if |t| > 1.96 under
    NW 7/24/95, Driscoll–Kraay and stock-day / day clusters.
  - HMM states are low-/high-volatility, not calm/stressed.
- **Story after v2:**
  - Volatility is the robust driver in both groups (min |t| 3.1 constrained, 2.5 unconstrained).
  - Informed trading (rolling impact) matters only in the unconstrained group, and only robustly once spread
    persistence is modelled (dynamic spec, DK t = 2.2; Granger p = 0.03); nothing in the tick-constrained group.
  - Forecasting power is mostly persistence and time of day (drop-column); LightGBM beats last bar's spread for
    4 of 6 stocks.
  - As a quoting rule, conditioning on volatility reduces a market maker's adverse-selection losses
    (paired t = 2.3 vs the no-driver rule), but at a 5-min markout passive quoting loses money in every rule:
    realised spreads ≈ 0, consistent with HFT competition (Week 8).
- **Still open:** source of the ISS 29-Oct sell-off (FactSet news ~15:20 CPH / 10:20 ET); due date and
  Convenor approval; report write-up (≤ 2,500 words), AI transparency statement, team roles.
