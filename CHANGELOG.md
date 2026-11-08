# Changelog

## v2.3 (8-Nov-2026): Christian's original October exports restored
- `data/raw` October files (136) replaced by Christian's original FactSet exports; Emanuel's 21-column re-export stays in git history. Differences: 5 extra columns not used by the code, the `Tick` flag on the first IND row (dropped), and the split point of the 6 part-files.
- Scripts 01-13, the report notebook and the 28 tests re-run on the originals: every number in every table and processed file is identical. Outputs refreshed only for the new ISS event label (03 and 08 tables and figures).

## v2.2 (8-Nov-2026): ISS 29-Oct event identified
- The 15:20-15:35 sell-off followed the Hong Kong Wang Fuk Court fire inquiry, which partly blamed ISS EastPoint (staff switched off the fire alarms). The report was public from the Hong Kong morning (~09:00 CPH); the first Danish report (MarketWire) came at 15:20 CPH, when the sell-off starts.
- `config.EVENT_DAYS` label, report Section 2.5 (+1 reference), CLAUDE.md and docs/PLAN.md updated. No numbers change.

## v2.1 (8-Nov-2026): v2 merged with the forward test, quoting simulation and volatility surface
`main` had moved on from `v1-draft` in parallel (Emanuel: forward test on 2-6 Nov, quoting simulation,
volatility surface, code-review fixes, data in git, report drafts 1-5). v2 was merged into it, then:

- **Renumbered:** v2's `09_ae_diagnostics.py` -> `12_ae_diagnostics.py`, `10_quote_rule.py` -> `13_quote_timing.py`
  (Emanuel's 09 forward test, 10 quoting simulation and 11 volatility surface keep their numbers).
- **One fill model:** v2's separate quote simulation (`quote_rule.py`, fills at the deepest level reached) was
  dropped; script 13 now runs Emanuel's engine (`quoting.py`: per order, first fill, half-tick grid, quote size
  = median order) with every rule scaled to the same in-sample average width, plus paired stock-day tests, for
  both 22-30 Oct and 2-6 Nov. Helpers `scale_to_is_width` and `paired_daily_test` moved into `quoting.py`.
- **Consistency:** `09_forward_holdout.py` uses `build_panel`'s lags (within-day `lag_ofi`, as 07/08); QLR in 12 uses
  the October days only; ISS 29-Oct labelled "intraday sell-off 15:20".
- **Full re-run** of 01-13 on the committed data and the report notebook; 01-04 reproduce `main` exactly (only the
  ISS event label changes). Tests: 28 passing.
- **Report (draft 6):** numbers re-read from the regenerated tables; Section 4.2 now states the standard-error
  robustness (volatility robust; rolling impact only with the lagged spread / Granger); 5.1 corrected (a 1-SD
  volatility shock is about 3% / 6% of the average half-spread, not 9% / 10%); 5.2 adds the equal-width
  timing result (Table 6); HMM states renamed low/high volatility; 7.2 drops the fixed BIC limitation and
  discloses the second scoring of the forward week; Appendix C (Applied Econometrics checks, Table C1); Figure 8
  (ISS sell-off); Christian's AI-statement paragraph drafted (to confirm); 2,384 words in sections 1-7.

### Key numbers after the merge
| Result | Value |
|---|---|
| Volatility, lowest t over 6 SE choices (constrained / unconstrained) | 3.14 / 2.49 |
| Rolling impact, unconstrained: NW 7 / lowest / dynamic DK | 2.49 / 1.85 / 2.20 |
| Granger-style p, unconstrained (impact / vol, DK) | 0.031 / 0.006 |
| LightGBM vs last bar, OOS RMSE (C / U) | 1.08 vs 1.32 / 1.32 vs 1.66 (DM -2.0 / -3.4) |
| LightGBM vs last bar, forward week (C / U) | 0.93 vs 1.42 / 1.49 vs 1.85 (DM -2.8 / -2.5) |
| Equal-width quote timing, vol + impact vs controls-only, paired t (OOS / FWD) | 2.32 / 2.48 |
| Same for LightGBM (OOS / FWD) | -0.57 / 0.41 |
| Quoting at the market's spread, realised per fill OOS (C / U) | -0.5 to -0.6 / -3.2 to -3.8 bps |

### Disclosed
The forward week has now been scored with the v1 specification (Emanuel, before and after his bug fixes) and
with the merged v2 specification; the v2 changes were decided on 8-Nov without access to the November data.

## v2 (8-Nov-2026): review and improvement round
Branch `v2`, starting from tag `v1-draft` (commit f4c9b9d). The review that drove these changes is
`docs/REVIEW_V1.md`. v1 outputs remain available at the `v1-draft` tag
(`git checkout v1-draft -- outputs/` to inspect them).

### Process
1. Reviewed v1 code and outputs; re-ran the main regression independently (coefficients matched to 4 dp).
2. Reproduced v1 exactly in a clean environment with the same package versions (pandas 3.1.0,
   numpy 2.5.4, statsmodels 0.15.0, arch 8.0.0, lightgbm 4.7.0, shap 0.54.1, hmmlearn 0.3.3) before
   changing anything, so every difference below comes from a deliberate change.
3. Made the changes script by script, re-ran 05–10 and the report notebook, and added tests (14 → 22).

### Changes, by review item
| # | Review finding | Change | Where |
|---|---|---|---|
| 1 | NW lag 7 too short (residual ACF 0.3–0.7) | SE robustness table: NW 7/24/95, Driscoll–Kraay, stock-day and day clusters; residual ACF per stock | `spread_model.fit_ols(cov=...)`, `residual_acf`; 07 |
| 2 | Spread persistence not modelled | Dynamic ADL(1,0) spec with the last observed spread; long-run effects b/(1−ρ) with delta-method SEs | `spread_model.long_run`; 07 |
| 3 | No joint tests of the impact terms | Joint Wald tests (AE `linearHypothesis`) | `spread_model.wald`; 07, 09 |
| 4 | Lag controls crossed the overnight gap (126/132 first bars) | Activity controls lagged within the day; first bar gets the stock's in-sample mean (09:00 dummy absorbs it). `lag_quoted_hs` deliberately stays the last observed spread (incl. previous close) | `spread_model.build_panel`, `lag_within_day` |
| 5 | VAR lag by BIC hit the grid edge (p = 10 for all) | BIC keeps falling to p = 40 (n ≈ 10⁴–10⁵); p fixed at 20 where impact is stable (< 4% change to p = 40); BIC path + sensitivity saved | 05, `05_lag_choice.csv` |
| 6 | No IRF confidence bands | Day-block bootstrap 95% CI (500 resamples) of each stock's permanent impact | `impact.bootstrap_impact`; 05 |
| 7 | GARCH without AE Lab 8 diagnostics | Ljung–Box and ARCH-LM before; standardised-residual checks after; normal vs Student-t | `volatility.garch_diagnostics`; 06 |
| 8 | Group OOS RMSE hides per-stock losses | RMSE relative to last bar, stocks × models, heatmap | 08, `08_oos_relative_to_last_bar.csv`, Fig 3b |
| 9 | SHAP may over-credit previous-day impact | Drop-column importance in the in-sample CV | 08, `08_r2_drop_column.csv`, Fig 4 |
| 10 | LightGBM best config at grid corner | Grid widened (leaves 3–31, min_child 10–100); best now interior (5 leaves, 20) | 08 |
| 11 | "Stressed" regime held ~50% of bars | Renamed low-vol / high-vol; 3-state HMM added (top state 8–15% of bars) | 08, Fig 5 |
| 12 | Promised quote-widening rule not built | Quote rules from the forecasts, scored by markout P&L of a stylised market maker | new 10, `quote_rule.py`, Fig 6 |
| 13 | ISS 29-Oct "news (check)" | Identified as a 15:20–15:35 sell-off (−3.5%, ~400 trades/5 min, half-spread 18.6 bps); event study | 10, Fig 7 |
| 14 | AE checks missing (ADF, Granger, QLR) | ADF, Granger-style Wald, QLR, plus one appendix table of every AE check | new 09, Appendix A |

### Key numbers, v1 → v2
In-sample spread regressions (original spec, quoted half-spread, NW L = 7; t-stats):

| Group | Driver | v1 t | v2 t | v2 min \|t\| over 6 SE choices | v2 dynamic spec, DK t |
|---|---|---|---|---|---|
| Constrained | rolling impact | −0.78 | −1.05 | 0.74 | −1.20 |
| Constrained | previous-day impact | 1.35 | 1.31 | 1.10 | 1.10 |
| Constrained | vol forecast | 3.66 | 4.03 | **3.12** | **3.17** |
| Unconstrained | rolling impact | 2.44 | 2.51 | 1.86 | **2.22** |
| Unconstrained | previous-day impact | −1.08 | −0.03 | 0.01 | 0.02 |
| Unconstrained | vol forecast | 5.52 | 5.63 | **2.47** | **3.63** |

Permanent impact (Hasbrouck, median across windows, bps): v1 VAR(10) Novo 1.39 … GN 7.19 →
v2 VAR(20) Novo 1.34 … GN 6.30; pooled in-sample with 95% CI: Novo 1.32 [1.24, 1.41], DSV 2.93 [2.44, 3.36],
Vestas 2.79 [2.51, 3.01], Pandora 5.14 [4.67, 5.56], GN 6.02 [5.33, 6.79], ISS 3.78 [3.16, 4.42].

OOS RMSE (bps, constrained / unconstrained): LightGBM 1.07 / 1.32 (v1) → 1.08 / 1.32 (v2); last bar 1.32 / 1.66.
LightGBM vs last bar DM: −2.00 / −3.40; per stock it beats last bar for DSV, Vestas, GN, ISS and ties for Novo, Pandora.

Drop-column CV importance: lagged spread +14.6%, time of day +4.0%, previous-day impact +0.8%,
vol forecast +0.2%, rolling impact −0.3%.

Quote rules (OOS, equal in-sample average width, 5-min markout, all stocks): every rule loses at the
5-min markout (−3.0 to −3.2 bps per DKK filled: realised spreads ≈ 0); the vol rule cuts losses vs the
controls-only rule (daily P&L paired t = 2.32, better on 60% of stock-days) and vs the constant rule
(t = 2.51), mainly in the unconstrained group. LightGBM's better spread forecast does not translate
into better P&L (t = 0.88 vs controls only).

AE diagnostics: all 18 ADF tests reject a unit root; Granger-style tests significant for impact
(p = 0.03) and vol (p = 0.006) in the unconstrained group only (DK SEs); QLR finds a 5% break in the
daily spread for DSV (22-Oct), Vestas (16-Oct) and Pandora (20-Oct); ARCH effects in all six stocks,
GARCH(1,1) removes them except DSV (borderline) and ISS.

### Disclosed
- The OOS period is evaluated a 4th time in v2 (after 07, refinement 1 and refinement 2 in v1). The v2
  changes were driven by the review of in-sample diagnostics, and LightGBM was re-tuned on in-sample
  days only, but OOS numbers should be read as confirmatory.
- Runtime: script 05 now takes ~4–5 min (bootstrap); everything else runs in under a minute.

### New outputs
Tables: `05_impact_bootstrap`, `06_diagnostics`, `07_se_robustness`, `07_residual_acf`, `07_dynamic`,
`07_dynamic_long_run`, `07_wald`, `08_r2_drop_column`, `08_oos_relative_to_last_bar`,
`08_hmm_highvol_share` (replaces `08_hmm_stressed_share`), `09_adf`, `09_granger`, `09_qlr`,
`09_ae_appendix`, `10_quote_rule_pnl`, `10_quote_rule_tests`, `10_iss_event_bars`, `10_iss_event_pnl`,
`report_table2b_se_robustness`, `report_table6_quote_rule`, `report_tableA_ae_checks`.
Figures: `05_impact_bootstrap`, `08_oos_by_stock`, `09_qlr`, `10_quote_rule_pnl`, `10_iss_event`,
`report_fig3b_oos_by_stock`, `report_fig6_quote_rule` (and updated `report_fig2/4/5`).

## v1-draft (2-Nov-2026)
Scripts 01–08 and the report notebook: data cleaning, signing, background analysis, bars, Hasbrouck
impact, volatility forecasts, spread model by tick-constraint group, refinements (stock × time,
LightGBM + SHAP, HMM regimes).
