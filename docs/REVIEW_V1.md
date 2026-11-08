# First-draft review (v1, 8-Nov-2026)

Review of commit f4c9b9d (tag `v1-draft`) against the Student Guide, the project Guidance, the coding plan and the Applied Econometrics blueprint (`AE Blueprint Final.R`). Every item is addressed in v2; see `CHANGELOG.md` for what changed and the before/after numbers.

Scope: repo `C:\Users\chris\Projects\TIFS` at commit f4c9b9d (16 commits, scripts 01–08, src/, report_figures.ipynb, all tables/figures), checked against the Student Guide, the Guidance doc, the Coding Plan and `AE Blueprint Final.R`. The main regression was re-run independently from the saved panels (coefficients match to 4 dp), plus extra SE / ADF / Granger / ADL checks.

## Bottom line
- The pipeline is careful: no look-ahead, assertions, forward-filtered HMM, day-wise CV, OOS looks disclosed. No material bugs found.
- **Volatility** result is robust to every SE choice tried.
- **Impact (unconstrained group)** is marginal in the static model (t 1.8–2.0 with longer HAC / clustering / Driscoll–Kraay) but significant once spread dynamics are modelled (ADL t = 2.58; Granger p = 0.015).
- Group-level OOS "wins" of the linear models hide per-stock losses; only LightGBM clearly beats the naive last-bar benchmark.
- AE methods are used in substance (NW HAC with the AE lag rule, VAR/IRF, GARCH, Wald) but not the R code, and several AE diagnostics are missing (ADF, ARCH tests, standardised-residual checks, Granger, QLR, bootstrap IRF CIs).
- Biggest gap vs the plan: the promised **practical quote-widening rule / market-maker P&L test** is not built.

## 1. Applied Econometrics usage
| AE blueprint method | In the project | Status |
|---|---|---|
| OLS + Newey–West HAC (Lab 7) | `fit_ols`, hac-panel; L = 7 | ✓ L = 7 equals AE rule floor(0.75·T^(1/3))−1 with T = 1,425 (and NW-1994 rule in AE `granger_tbl`) |
| VAR + IRF (`VAR`, `irf`) | Hasbrouck VAR hand-coded, cumulative IRF | ✓ but no bootstrap CIs |
| IC lag selection (`VARselect`) | BIC over p = 5..10 | ✗ p = 10 chosen for all 6 stocks → corner solution, extend grid |
| ARCH/GARCH (Lab 8) | `arch` GARCH(1,1), GJR | ✓ but no ARCH-effect test before, no standardised-residual ACF after; normal errors only |
| Joint Wald (`linearHypothesis`) | MZ unbiasedness, regime t-tests | partial — not used for joint impact terms |
| Granger causality | not used | added in review: unconstrained impact p = 0.015, vol p = 0.0006; constrained ≈ 0.11 |
| ADF unit root | not used | added: all spread & vol-ratio series stationary (τ −3.9 to −18.4) → levels OK |
| QLR structural break | not used | natural fit for OOS spread drift / DST shift |
| ADL / dynamic multipliers | not in main spec | added: lagged spread removes residual AC; long-run effects |
| FE with clustered SEs (`feols`) | stock dummies, HAC within stock | clustering not reported |
| Beyond AE | DM (HLN), MZ, QLIKE, HAR, LightGBM/SHAP, HMM | good for "beyond course material" |

## 2. Accuracy
Matches plan claims: impact 1.39–7.19 bps; impact/effective 91–115%; vol t = 3.7 / 5.5; LightGBM RMSE 1.07 / 1.32 vs yesterday 1.38 / 1.85; SHAP ordering.
Imprecise: "MZ R² 0.2–0.3" → per stock 0.20–0.38 for five stocks, ISS 0.01–0.03, pooled 0.03–0.08.

SE robustness (t-stats, in-sample):
| | Constrained impact roll | Constrained vol | Unconstrained impact roll | Unconstrained vol |
|---|---|---|---|---|
| NW L=7 (draft) | −0.78 | 3.66 | 2.44 | 5.52 |
| NW L=24 | −0.80 | 3.32 | 1.94 | 4.49 |
| NW L=95 | −0.98 | 3.01 | 1.96 | 3.69 |
| Cluster stock-day | −0.57 | 2.51 | 1.81 | 3.25 |
| Cluster day (15) | −0.59 | 2.40 | 1.78 | 2.66 |
| Driscoll–Kraay L=24 | −0.80 | 3.24 | 1.99 | 4.26 |
| ADL(1) + NW L=7 | −0.30 | 2.19 | 2.58 | 4.01 |
Residual ACF lag 1: 0.30–0.68; Novo lag 95 (same bar yesterday) 0.71.

Other issues:
- Per-stock OOS: original combined loses to yesterday for Novo (0.55 vs 0.26), DSV (0.73 vs 0.60), Vestas (1.08 vs 0.73), Pandora (1.51 vs 1.13). Linear models ≈ last bar at group level (DM p > 0.4). LightGBM beats last bar (DM −2.15 / −3.46); per stock it wins for DSV, Vestas, GN and ISS, and ties for Novo and Pandora.
- Constrained OOS gain comes from ISS 29-Oct (ex-event DM p 0.3–0.4).
- HMM "stressed" = 45–57% of bars, stay prob 0.79–0.82 (~25-min spells), ISS 29-Oct only 41% stressed → it's a high- vs low-vol state, not stress.
- SHAP of prevday impact (0.12–0.19) > vol, but it's constant within stock-day → may act as a day-level identifier; check with drop-column CV.
- LightGBM best config at grid corner (7 leaves, min_child 20).
- GARCH persistence 0.983–0.997; ISS ex-event GARCH RMSE 17.7 vs ATR 7.9 (shock carried into 30-Oct) → supports HAR.
- GN/ISS 300-order windows span ~20 h (median 1,211 / 1,267 min) → "intraday" impact is ~2-day average for thin stocks.
- Minor: 1-bar lag controls cross overnight (126/132 first bars); use groupby(stock, date).

ISS 29-Oct: genuine intraday shock, 15:20–15:35 CPH (~50 min after the 14:30 US open), mid 287 → 280 (−3.5%), 395 trades in one 5-min bar, half-spread 18.6 bps (vs ~4–5), partial recovery by 16:00. Not the Q3 trading update (4-Nov) or DT settlement (May). Check FactSet news ~10:20 ET.

## 3. Course & rubric links
- Week 6: market-making functionality (risk adjustment), tick size "constrains the spread" (tick natural experiment), ATR/volatility regime, event risk/event study.
- Week 4: Glosten–Milgrom / Kyle adverse selection & Bayes mid update → Hasbrouck impact; Ho & Stoll inventory → vol as inventory-risk proxy (inventory itself not modelled).
- Week 3: fundamental vs transitory volatility; informed vs uninformed; market-making edge → permanent impact vs realised spread ≈ 0.
- Week 2: GARCH; spread–volatility link. Week 7: IS/OOS, data snooping. Week 8: latency/HFT, European fragmentation.
- Rubric risks: only one git author (group collaboration 25%); AI transparency statement required; 2,500-word limit (Guidance 600 words is from the IAs); due date still to confirm; Convenor approval for Appendix E still open.

## 4. Next steps (priority order)
1. SE robustness table (DK, stock-day cluster, L=24/95) + residual ACF in 07.
2. Dynamic spec (within-day lagged spread) as main or robustness; long-run effects.
3. Per-stock OOS figure (RMSE ÷ last-bar, stocks × models, ex-event).
4. Quote-widening rule + simple OOS market-maker P&L / spread-capture test (closes the Week 6 loop; promised in plan Outcomes).
5. AE diagnostics appendix: ADF, ARCH-LM & Ljung–Box on standardised GARCH residuals, Granger, bootstrap IRF CIs, BIC grid to 20.
6. Rename HMM states; ISS 29-Oct event-study figure; identify the news.
7. SHAP caveat via drop-column CV; widen LightGBM grid.
8. Fix overnight lags; QLR test on daily spreads; optional inventory proxy (cumulative signed volume).
9. Write-up: 7-section report ≤ 2,500 words, AI transparency, README results, team roles/commits.
