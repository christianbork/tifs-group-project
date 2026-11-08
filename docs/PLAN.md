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
- **Big vol days barely move spreads:** DSV 21-Oct earnings had RV z≈9.8 and turnover z≈32, but the spread rose only ~7%. ISS 29-Oct (OOS) had RV z≈15 and turnover z≈18 (Wang Fuk Court fire inquiry, see below).
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

## 8b. Forward test (`09_forward_holdout.py`, added 8-Nov by Emanuel)
- Why: the OOS window (22–30 Oct) was evaluated three times (07, refinement 1, refinement 2), so it no longer counts as untouched. Lecture 7: out-of-sample data should be used once. Student Guide A.5 rewards forward testing.
- Data: 2–6 Nov 2026 for all six stocks, exported from FactSet Time & Sales on 8-Nov in the same format (Exchange time zone, Venue column); Novo 4/5/6-Nov split into `-partN` files because of the 25k-row cap. Details in `data/README.md`. Sample label `FWD` from `config.sample_label()`.
- Method: every model is re-fitted on IS days with the specification and LightGBM hyper-parameters chosen in 07/08 (read from `08_r2_lgbm_cv.csv`); FWD rows are prediction targets only. Same benchmarks (IS average, yesterday, last bar), RMSE + Diebold–Mariano, plus a direction-of-change hit rate vs the last bar (market makers act on widen/tighten, not on the level).
- Stress: three event days fell in the week — Novo 4-Nov (~12m shares, 3x normal), Pandora 4-Nov (+11%), GN 5-Nov (+12%, 6x volume). Reported all days vs ex event days and per stock-day.
- Extra validation data: ISS 5-Nov and GN 5-Nov full quote streams and a Novo 10:00–11:00 window on 6-Nov feed the sampling-bias check in 02 (Novo: 3.54 bps time-weighted vs 2.99 trade-weighted in that hour, i.e. the trade-time sample understates the quoted spread of the most liquid stock by ~16%).
- Rule: read the 09 output once, report it as is, do not go back and change 07/08.

## 8c. Quoting-rule simulation (`10_quoting_rule.py`, added 8-Nov by Emanuel)
- Why: the rubric asks whether results are economically meaningful; the Week 6 market-making diagram ends in P&L, not RMSE. Also the natural way to present the "negative" forecasting result: a forecast that does not beat persistence in RMSE may still change what a market maker earns, or it may not.
- Rule: at each 5-min bar post half-spread h_t (a forecast or benchmark), floored at half a tick. A lit trade fills the market maker if the aggressor's effective half-spread e >= h_t (our quote was at least as good as the one hit), up to the stock's median trade size. Revenue h_t, cost = 5-min mid move (05 decomposition), realised = h - impact. Report fills, fill rate, spread captured, adverse selection, realised bps, DKK P&L per day, daily Sharpe, max inventory; frontier plot fill-rate vs realised bps.
- Rules: IS average, yesterday, last bar, original / refined / log / refined+lag / LightGBM (from 08 and 09 saved predictions), oracle = realised bar spread.
- Fills are counted per aggressive order (first fill's quote, order's total volume), quotes are snapped to the tick grid, the tick is measured per stock-day, and inventory is accumulated in DKK in time order (fixes from the 8-Nov code review; the first version counted every print as a fill and overstated losses 2-3x). Result on the full data: no rule is profitable. Market-own-spread rules lose 0.45-0.6 bps per fill (constrained) and 3.2-3.8 bps (unconstrained) OOS, 1.0-1.3 and 1.3-2.8 bps in the forward week; model rules quote wider, fill less and lose less (LightGBM constrained OOS: -0.19 bps at 47% fills vs -0.46 at 75% for the last bar). Consistent with 05: adverse selection takes the whole effective spread.

## 8d. Volatility surface (`11_vol_surface.py`, added 8-Nov by Emanuel)
- Volatility signature (Andersen, Bollerslev, Diebold, Labys 2000) per in-sample day and hour: realised variance per minute of trade-attached mid returns on grids from 5 s to 30 min. Outputs `11_vol_signature.csv`, `11_vol_surface_by_hour.csv`, `11_vol_surface.png`.
- Results: signature flattens between 2 and 15 min; noise share of 1-min variance 20-27% (Novo, Vestas, Pandora, ISS), 12% (DSV), 0% (GN, spread wide vs tick). Novo's excess per 5-s sample is 1.1 bps = its half-spread (pure bounce). Open hour: 5-s sigma 2.1-2.8x midday vs 1.3-2.2x at 15 min, so part of the opening volatility is transitory.
- Use in the report: Section 2.4 + Figure 8 + Table 6; limitation that the 1-min volatility regressor contains a spread-driven component.

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
- ~~Check what happened to ISS on 29-Oct~~ Done: Hong Kong Wang Fuk Court fire inquiry partly blamed ISS's subsidiary; Danish news 15:20 CPH.
- Due date: Student Guide says Fri 20-Oct-2026 but that is a Tuesday; Week 10 Friday is 20-Nov-2026 → confirm with Shane. Convenor approval for the Appendix E angle.
- Broker IDs: the FactSet export has Buy ID / Sell ID columns but Nasdaq Copenhagen post-trade is anonymous (all "001"), so a PIN-style measure from broker IDs is not possible with this data.

## 10. v2 review round (8-Nov-2026)
Review: `docs/REVIEW_V1.md`; full list of changes and before/after numbers: `CHANGELOG.md`.
- **New scripts:** `12_ae_diagnostics.py` (ADF, Granger-style Wald, QLR, AE appendix table) and
  `13_quote_timing.py` (quoting rules at equal average width on the quoting engine of script 10; ISS 29-Oct event study).
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
  - Forecasting power is mostly persistence and time of day (drop-column); LightGBM beats last bar's spread in
    both test weeks (significantly for DSV, Vestas, GN; ISS borderline; ties for Novo and Pandora out of sample).
  - As a quoting rule (script 13, equal average width, script 10's engine), the volatility + impact forecasts beat
    the no-driver rule in both weeks (paired t = 2.3 OOS, 2.5 forward); LightGBM does not. Every rule still loses
    at a 5-min markout: realised spreads ≈ 0, consistent with HFT competition (Week 8).
- **Merged with the forward test (8-Nov, v2.1):** see CHANGELOG; scripts 12 (AE checks) and 13 (quote timing).
- **ISS 29-Oct identified (8-Nov):** Hong Kong inquiry into the Wang Fuk Court fire (released 29-Oct, HK morning) partly blamed ISS
  EastPoint; first Danish report (MarketWire) 15:20 CPH = start of the sell-off.
- **Still open:** due date and
  Convenor approval; report write-up (≤ 2,500 words), AI transparency statement, team roles.

