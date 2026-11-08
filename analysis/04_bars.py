# %% [markdown]
# # 04 Bars: 1-, 5- and 15-min intervals per stock
# Builds the interval data that scripts 05-08 read (spec: docs/PLAN.md §4c).
# 1. Exact 1-min bars per stock-day from the signed trades (quote carried forward).
# 2. Aggregate to 5-min (main) and 15-min (robustness); the last 15-min bar is
#    16:45-16:55 (10 min, see `bar_minutes`).
# 3. Checks: bar counts, coverage, and agreement with script 03's daily spreads.

# %% Setup
import numpy as np
import pandas as pd

from tifs_group_project import config
from tifs_group_project.bars import aggregate, minute_bars, tick_size

# %% 1-min bars
frames = []
for stock in config.STOCKS:
    d = pd.read_parquet(config.PROCESSED / f"signed_{stock}.parquet").sort_values(
        "ts", kind="stable"
    )
    tick = tick_size(d["bid"], d["ask"])
    for day, g in d.groupby("date"):
        frames.append(minute_bars(g, day, tick).assign(stock=stock, date=day))
m1 = pd.concat(frames, ignore_index=True)

bars = {f: aggregate(m1, k) for f, k in [("1min", 1), ("5min", 5), ("15min", 15)]}
for f, b in bars.items():
    b.to_parquet(config.PROCESSED / f"bars_{f}.parquet", index=False)

# %% Checks: shape, coverage, missing values
rows = []
for f, b in bars.items():
    rows.append(
        {
            "freq": f,
            "rows": len(b),
            "stock_days": b.groupby(["stock", "date"]).ngroups,
            "bars_per_day": b.groupby(["stock", "date"]).size().unique().tolist(),
            "pct_no_quote": round(100 * b["quoted_hs_bps"].isna().mean(), 2),
            "pct_no_trades": round(100 * (b["trades"] == 0).mean(), 2),
            "pct_thin": round(100 * b["thin"].mean(), 2),
            "pct_ret_missing": round(100 * b["ret_bps"].isna().mean(), 2),
            "trades_total": int(b["trades"].sum()),
        }
    )
check = pd.DataFrame(rows).set_index("freq")
check.to_csv(config.TABLES / "04_bars_check.csv")
print("Bar checks:\n", check.to_string(), "\n")

# trade counts and day returns must be identical across frequencies
tot = {
    f: b.groupby(["stock", "date"]).agg(
        trades=("trades", "sum"), ret=("ret_bps", "sum")
    )
    for f, b in bars.items()
}
assert (tot["1min"]["trades"] == tot["5min"]["trades"]).all() and (
    tot["1min"]["trades"] == tot["15min"]["trades"]
).all()
assert np.allclose(tot["1min"]["ret"], tot["5min"]["ret"]) and np.allclose(
    tot["1min"]["ret"], tot["15min"]["ret"]
)

# daily time-weighted full spread from bars should match script 03 (same carried-forward quote)
b5 = bars["5min"]
day = b5.groupby(["stock", "date"]).apply(
    lambda x: 2 * np.average(x["quoted_hs_bps"].fillna(0), weights=x["q_sec"]),
    include_groups=False,
)
d03 = pd.read_csv(
    config.TABLES / "03_daily_stock_day.csv", parse_dates=["date"]
).set_index(["stock", "date"])
diff = (day - d03["spread_bps_tw"]).abs()
print(
    f"Daily spread vs script 03: max abs diff {diff.max():.3f} bps, median {diff.median():.4f} bps\n"
)

# %% Summary per stock (5-min, in-sample)
s = (
    b5[b5["sample"] == "IS"]
    .groupby("stock")
    .agg(
        group=("group", "first"),
        quoted_hs_bps=("quoted_hs_bps", "median"),
        eff_hs_bps=("eff_hs_bps", "median"),
        pct_time_1tick=("pct_time_1tick", "mean"),
        trades_per_bar=("trades", "median"),
        pct_thin=("thin", "mean"),
        rv_bps=("rv_bps", "median"),
        ofi_sd=("ofi", "std"),
    )
    .reindex(list(config.STOCKS))
)
s["pct_thin"] *= 100
s["eff_over_quoted"] = s["eff_hs_bps"] / s["quoted_hs_bps"]
s.round(2).to_csv(config.TABLES / "04_bars_summary_5min.csv")
print("5-min bars, in-sample medians:\n", s.round(2).to_string(), "\n")

# US-open alignment: bars around the US open, per date regime
us = b5.assign(
    us_open_cph=b5["date"]
    .dt.strftime("%Y-%m-%d")
    .map(config.US_OPEN_CPH)
    .fillna(config.US_OPEN_CPH["default"])
    + " CPH"
)
us = (
    us[us["min_since_us_open"].between(-30, 25)]
    .groupby(["us_open_cph", "min_since_us_open"])["rv_bps"]
    .median()
    .unstack(0)
)
us.round(1).to_csv(config.TABLES / "04_rv_around_us_open.csv")
print("Median 5-min RV (bps) around the US open:\n", us.round(1).to_string())
