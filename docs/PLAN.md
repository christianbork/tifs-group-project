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

## 4. Background analysis (next, `03_background.py`)
- Intraday U-shapes (spread, volume, volatility by minute), daily series, event days.
- **Tick-constraint check:** share of time the spread = 1 tick (Novo likely pinned).
- Decides the interval length (5-min) and the treatment of the open/close.

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
- Half-spread(bps, 5-min) = a + b·impact(t−1 day) + c·vol forecast + time-of-day + activity controls + stock fixed effects. Quoted and effective spreads; Newey–West SEs; estimated on in-sample days.
- OOS race: average-spread benchmark / vol-only / impact-only / combined → RMSE + Diebold–Mariano.
- Risk: **tick-constrained spreads** (Novo) → use the effective spread or treat as a case study (tick size mutes the risk response).

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
- Novo tick constraint: effective-spread model vs case study (decide after script 03).
- Due date: likely Fri 20-Nov-2026, 9pm → confirm. Convenor approval for the Appendix E angle.
