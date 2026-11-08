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
Run the scripts in `analysis/` in number order from the project root, e.g. `python analysis/01_load_clean.py`. Scripts 01–08 are the original study (in-sample 1–21 Oct, out-of-sample 22–30 Oct); `09_forward_holdout.py` evaluates the frozen models once on the untouched 2–6 Nov week; `10_quoting_rule.py` turns every forecast and benchmark into a market-maker quoting rule and simulates fills, spread capture, adverse selection and P&L on the trade feed (reads the predictions saved by 08 and 09).

## Structure
```
data/raw/          original FactSet exports (gitignored)
data/processed/    cleaned parquet files (gitignored)
src/tifs_group_project/   shared functions
analysis/          numbered pipeline scripts (# %% cells, run in VS Code)
notebooks/         report_figures.ipynb (final figures and tables)
outputs/           figures and tables for the report
tests/
```

## Results
_TBC_
