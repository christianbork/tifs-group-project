# tifs-group-project
FINS5566 (T3 2026) group project: what drives bid-ask spreads on Nasdaq Copenhagen stocks, informed trading vs volatility.

## Setup
```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
pip install -r requirements.txt
pip install -e .
```
Raw data is not in the repo. Put the FactSet exports in `data/raw/<Stock>/` (DSV, GN, ISS, Novo, Pandora, Vestas) and the ISS quotes sample in `data/raw/quotes_sample/`.

## Run order
Run the scripts in `analysis/` in number order from the project root, e.g. `python analysis/01_load_clean.py`.

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
