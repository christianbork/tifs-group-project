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
Raw data is not in the repo. Put the FactSet exports in `data/raw/<Stock>/` (DSV, GN, ISS, Novo, Pandora, Vestas) and the quote-stream samples in `data/raw/quotes_sample/`. October (1–30 Oct, in-sample + out-of-sample) comes from Christian's exports; the November forward-test week (2–6 Nov) is documented in `data/README.md`.

## Run order
Run the scripts in `analysis/` in number order from the project root, e.g. `python analysis/01_load_clean.py`. Scripts 01–08 are the original study (in-sample 1–21 Oct, out-of-sample 22–30 Oct); `09_forward_holdout.py` evaluates the frozen models once on the untouched 2–6 Nov week.

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
