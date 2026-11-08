# tifs-group-project
FINS5566 (T3 2026) group project: what drives bid-ask spreads on Nasdaq Copenhagen stocks, informed trading vs volatility.

## Setup
```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
source .venv/bin/activate          # macOS / Linux (python3.12 -m venv .venv; LightGBM needs: brew install libomp)
pip install -r requirements.txt
pip install -e .
```
The FactSet exports (`data/raw/<Stock>/`, `data/raw/quotes_sample/`) and the processed parquet files (`data/processed/`) are in the repo since 8 Nov 2026, so a fresh clone runs end to end. The repo is private; the data is licensed to the team through UNSW's FactSet access and must not be shared outside it. Both the October sample (1–30 Oct, in-sample + out-of-sample) and the November forward-test week (2–6 Nov) can be re-exported from FactSet Time & Sales with the settings in `data/README.md`; the 8 Nov re-export reproduces the committed October tables exactly.

## Run order
Run the scripts in `analysis/` in number order from the project root, e.g. `python analysis/01_load_clean.py`,
then `notebooks/report_figures.ipynb` for the report figures and tables, and `report/build_docx.py` for the Word report.
Scripts 01–08 are the original study (in-sample 1–21 Oct, out-of-sample 22–30 Oct); 09 is the forward test on the
untouched 2–6 Nov week; 10–13 are the quoting simulation, volatility surface, Applied Econometrics checks and the
equal-width quote-timing test.

| Script | What it does | Runtime |
|---|---|---|
| 01_load_clean | stitch FactSet exports, label prints, data-quality table | ~1 min |
| 02_sign_trades | quote rule + tick test, order groups, validation against four full quote streams | ~1 min |
| 03_background | intraday shapes, tick constraint, unusual days | < 1 min |
| 04_bars | 1/5/15-min bars | < 1 min |
| 05_price_impact | Hasbrouck VAR(20) impact, bootstrap CIs, spread decomposition, rolling impact | ~5 min |
| 06_volatility | GARCH/GJR/HAR/ATR forecasts, AE Lab 8 diagnostics, OOS evaluation | < 1 min |
| 07_spread_model | spread model by tick group, SE robustness, dynamic spec, Wald tests, OOS race | < 1 min |
| 08_refine | stock x time spec, LightGBM + SHAP + drop-column, per-stock OOS, HMM regimes | ~1 min |
| 09_forward_holdout | frozen models scored once on the 2–6 Nov forward week | < 1 min |
| 10_quoting_rule | every forecast as a quoting rule: fills, spread capture, adverse selection, P&L, inventory | < 1 min |
| 11_vol_surface | volatility signature by sampling interval and hour | ~1 min |
| 12_ae_diagnostics | ADF, Granger, QLR, Applied Econometrics appendix table | < 1 min |
| 13_quote_timing | quoting rules at equal average width (timing value of the drivers), paired tests; ISS 29-Oct event study | < 1 min |

## Structure
```
data/raw/          original FactSet exports (tracked since 8 Nov)
data/processed/    cleaned parquet files (tracked since 8 Nov)
src/tifs_group_project/   shared functions
analysis/          numbered pipeline scripts (# %% cells, run in VS Code)
notebooks/         report_figures.ipynb (final figures and tables)
outputs/           figures and tables for the report
tests/
```

## Versions
- `v1-draft` (tag): first draft, scripts 01–08.
- `v2` (branch): review round, see `docs/REVIEW_V1.md` (the review) and `CHANGELOG.md` (changes, v1 → v2 numbers).

## Results (v2)
- Volatility is the robust spread driver in both tick-constraint groups (significant under every SE choice).
- Informed trading (Hasbrouck impact, 1.3–6.0 bps per buy order) takes 91–115% of the effective spread, but
  moves spreads only where the tick allows (unconstrained group), and robustly only once spread persistence is modelled.
- Out of sample, forecasts gain mainly from spread persistence and time of day; LightGBM beats last bar's spread
  for 4 of 6 stocks.
- A volatility-adjusted quoting rule reduces a stylised market maker's adverse-selection losses relative to
  rules without the drivers; passive quoting still loses at a 5-min markout (realised spreads ≈ 0).
