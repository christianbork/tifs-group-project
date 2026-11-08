# %% [markdown]
# # 01 Load and clean
# Reads every FactSet export in data/raw/<stock>/, stitches split days,
# drops auction indicative rows (IND), labels each print (lit / auction /
# dark / off-book / close) and saves one parquet per stock to data/processed/.
# Also writes a stock-day data-quality table for the report.

# %% Setup
import pandas as pd

from tifs_group_project import config
from tifs_group_project.io import load_stock

config.PROCESSED.mkdir(parents=True, exist_ok=True)
config.TABLES.mkdir(parents=True, exist_ok=True)

# %% Load, clean and save each stock
qa_rows = []
for stock in config.STOCKS:
    df, qa = load_stock(stock)
    df.to_parquet(config.PROCESSED / f"trades_{stock}.parquet", index=False)
    qa_rows += qa
    print(f"{stock:8s} days={df['date'].nunique():2d}  rows kept={len(df):>8,}  continuous lit={df['continuous'].sum():>8,}")

qa = pd.DataFrame(qa_rows).fillna(0)
qa.to_csv(config.TABLES / "01_data_quality_stock_day.csv", index=False)

# %% Checks: every stock-day complete and untruncated
print("\nStock-days flagged as possibly truncated:")
print(qa.loc[qa["truncated_flag"], ["stock", "date", "first_ts", "last_ts", "max_file_rows"]].to_string(index=False) or "none")

# %% Summary table for the report (continuous session, lit trades; October only, the forward week is described in script 09)
qa_oct = qa[qa["sample"] != "FWD"]
summary = (
    qa_oct.groupby("stock", sort=False)
    .agg(
        days=("date", "count"),
        lit_trades_per_day=("n_continuous", "mean"),
        median_spread_bps=("median_spread_bps", "median"),
        turnover_dkk_m_per_day=("dkk_turnover_m", "mean"),
        pct_outside_quote=("pct_outside_quote", "mean"),
        pct_bad_quote=("pct_bad_quote", "mean"),
    )
    .round(1)
)
kinds = [c for c in qa.columns if c.startswith("n_") and c not in ("n_continuous",)]
mix = qa_oct.groupby("stock", sort=False)[kinds].sum()
mix = (100 * mix.div(mix.sum(axis=1), axis=0)).round(1).add_prefix("pct_")
summary = summary.join(mix)
summary.to_csv(config.TABLES / "01_data_summary.csv")
print("\n", summary.to_string())
