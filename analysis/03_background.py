# %% [markdown]
# # 03 Background: intraday patterns, daily series, event days, tick constraint
# 1. Intraday U-shapes by minute: quoted spread, volume, volatility.
# 2. Daily series per stock, with the OOS period shaded and event days flagged.
# 3. Tick constraint: how often is the spread one tick wide?
# 4. Inputs to the design choices: interval length and open/close treatment.
#
# Shapes and design checks use in-sample days only, so the choices made here
# (interval length, open/close buffers) do not peek at the OOS period.
# Spreads are sampled at trades (attached quote); "time-weighted" assumes the
# attached quote prevails until the next trade (~5% bias, see script 02).

# %% Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tifs_group_project import config

config.FIGURES.mkdir(parents=True, exist_ok=True)
COLORS = dict(
    zip(
        config.STOCKS,
        ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    )
)
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e5e4e0",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
    }
)

# %% Load signed continuous trades
frames = []
for stock in config.STOCKS:
    d = pd.read_parquet(config.PROCESSED / f"signed_{stock}.parquet").sort_values(
        "ts", kind="stable"
    )
    steps = np.diff(np.unique(np.r_[d["bid"], d["ask"]]).round(4))
    d["tick"] = steps[steps > 0].min()  # smallest observed price step = tick size
    frames.append(d)
d = pd.concat(frames, ignore_index=True)
d["spread_bps"] = (d["ask"] - d["bid"]) / d["mid"] * 1e4
d["spread_ticks"] = ((d["ask"] - d["bid"]) / d["tick"]).round().astype(int)
d["dkk"] = d["price"] * d["vol"]
d["minute"] = d["ts"].dt.floor("min")
d["mod"] = (d["ts"].dt.hour - 9) * 60 + d["ts"].dt.minute  # minute of day, 0 = 09:00
# time the attached quote is assumed to prevail: until the next trade (or 17:00)
nxt = (
    d.groupby(["stock", "date"])["ts"]
    .shift(-1)
    .fillna(d["date"] + pd.Timedelta(config.LAST_CONT_TRADE))
)
d["dur"] = (nxt - d["ts"]).dt.total_seconds()
print(
    d.groupby("stock")[["tick", "spread_bps", "dur"]]
    .agg(["min", "median", "max"])
    .round(3),
    "\n",
)


# %% 1-min mid grid per stock-day: last mid in each minute, forward-filled
def minute_grid(g: pd.DataFrame) -> pd.DataFrame:
    day = g.name[1]  # (stock, date) group key
    idx = pd.date_range(
        day + pd.Timedelta(config.CONT_START),
        day + pd.Timedelta(config.LAST_CONT_TRADE),
        freq="min",
        inclusive="left",
    )
    m = g.groupby("minute").agg(
        mid=("mid", "last"), dkk=("dkk", "sum"), trades=("price", "size")
    )
    m = m.reindex(idx)
    m["mid"] = m["mid"].ffill()
    m[["dkk", "trades"]] = m[["dkk", "trades"]].fillna(0)
    m["ret_bps"] = (
        np.log(m["mid"]).diff() * 1e4
    )  # first minute NaN: overnight/auction move excluded
    m["mod"] = np.arange(len(m))
    return m


grid = (
    d.groupby(["stock", "date"], group_keys=True)
    .apply(minute_grid, include_groups=False)
    .reset_index(level=[0, 1])
    .rename_axis("minute")
    .reset_index()
)
grid["sample"] = config.sample_label(grid["date"])

# %% Intraday U-shapes (in-sample), per minute
d_is, g_is = d[d["sample"] == "IS"], grid[grid["sample"] == "IS"]
spread_min = d_is.groupby(["stock", "mod"]).apply(
    lambda x: np.average(x["spread_bps"], weights=x["dur"]), include_groups=False
)
volu_min = g_is.groupby(["stock", "mod"])["dkk"].mean()
vol_min = g_is.groupby(["stock", "mod"])["ret_bps"].apply(
    lambda r: np.sqrt((r**2).mean())
)
intraday = pd.DataFrame(
    {"spread_bps_tw": spread_min, "dkk_turnover": volu_min, "vol_1min_bps": vol_min}
)
intraday.round(3).to_csv(config.TABLES / "03_intraday_by_minute.csv")

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
titles = [
    "Quoted spread (time-weighted)",
    "Turnover (DKK)",
    "Volatility (RMS 1-min mid return)",
]
for ax, col, title in zip(axes, intraday.columns, titles):
    for stock in config.STOCKS:
        s = intraday.loc[stock, col]
        s = (
            (s / s.mean()).rolling(5, center=True, min_periods=1).mean()
        )  # index to own mean, 5-min smooth
        ax.plot(s.index, s.values, color=COLORS[stock], label=stock)
    ax.axhline(1, color="#8a8984", linewidth=0.8)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel("× stock's daily mean")
axes[0].legend(ncol=6, frameon=False, loc="upper center", fontsize=9)
ticks = np.arange(0, 481, 60)
axes[-1].set_xticks(ticks, [f"{9 + t // 60:02d}:00" for t in ticks])
axes[-1].set_xlabel("Copenhagen time (in-sample days, 5-min centred average)")
fig.tight_layout()
fig.savefig(config.FIGURES / "03_intraday_ushapes.png", dpi=150)
plt.close(fig)

# %% Daily series per stock-day (all days)
r5 = grid.groupby(["stock", "date"]).apply(
    lambda g: np.log(g["mid"].iloc[::5]).diff().mul(1e4), include_groups=False
)
rv5 = (r5**2).groupby(
    level=[0, 1]
).sum() ** 0.5  # realised vol from 5-min mid returns, bps
daily = d.groupby(["stock", "date"]).apply(
    lambda x: pd.Series(
        {
            "spread_bps_tw": np.average(x["spread_bps"], weights=x["dur"]),
            "spread_bps_median": x["spread_bps"].median(),
            "pct_time_1tick": 100
            * x.loc[x["spread_ticks"] == 1, "dur"].sum()
            / x["dur"].sum(),
            "trades": len(x),
            "dkk_turnover_m": x["dkk"].sum() / 1e6,
        }
    ),
    include_groups=False,
)
daily["rv_5min_bps"] = rv5
daily = daily.reset_index()
daily["sample"] = config.sample_label(daily["date"])
daily["event"] = [
    config.EVENT_DAYS.get((s, str(t.date())), "")
    for s, t in zip(daily["stock"], daily["date"])
]

# robust z-score vs the stock's in-sample days: flags unusual days beyond the known event list
for col in ["spread_bps_tw", "rv_5min_bps", "dkk_turnover_m"]:
    ref = daily[daily["sample"] == "IS"].groupby("stock")[col]
    med, mad = (
        daily["stock"].map(ref.median()),
        daily["stock"].map(ref.apply(lambda x: (x - x.median()).abs().median())),
    )
    daily[f"z_{col}"] = (daily[col] - med) / (1.4826 * mad)
# descriptive tables and figures use October only; the forward week's daily stats (z-scored
# against the same in-sample reference) go to a separate file for the event-day discussion in 09
daily_all = daily
daily = daily_all[daily_all["sample"] != "FWD"].copy()
daily.to_csv(config.TABLES / "03_daily_stock_day.csv", index=False, float_format="%.3f")
daily_all[daily_all["sample"] == "FWD"].to_csv(
    config.TABLES / "03_daily_stock_day_fwd.csv", index=False, float_format="%.3f"
)

zcols = [c for c in daily if c.startswith("z_")]
unusual = daily[(daily[zcols].abs() > 3).any(axis=1) | (daily["event"] != "")]
unusual.to_csv(config.TABLES / "03_unusual_days.csv", index=False, float_format="%.2f")
print(
    "Unusual stock-days (|robust z| > 3 vs in-sample, or known event):\n",
    unusual[
        [
            "stock",
            "date",
            "sample",
            "event",
            "spread_bps_tw",
            "rv_5min_bps",
            "dkk_turnover_m",
            *zcols,
        ]
    ].to_string(index=False, float_format="%.1f"),
    "\n",
)

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
for ax, col, title in zip(
    axes,
    ["spread_bps_tw", "rv_5min_bps", "dkk_turnover_m"],
    [
        "Quoted spread, time-weighted (bps)",
        "Realised volatility, 5-min mid returns (bps)",
        "Turnover (DKK m, log scale)",
    ],
):
    for stock in config.STOCKS:
        s = daily[daily["stock"] == stock]
        ax.plot(
            s["date"],
            s[col],
            color=COLORS[stock],
            marker="o",
            markersize=3,
            label=stock,
        )
    ax.axvspan(
        pd.Timestamp(config.IS_END) + pd.Timedelta(hours=12),
        daily["date"].max() + pd.Timedelta(hours=12),
        color="#8a8984",
        alpha=0.12,
        linewidth=0,
    )
    ax.set_title(title, loc="left", fontsize=11)
for (s, t), label in config.EVENT_DAYS.items():
    for ax in axes:
        ax.axvline(pd.Timestamp(t), color="#52514e", linewidth=0.8, linestyle="--")
    axes[2].text(
        pd.Timestamp(t) - pd.Timedelta(hours=6),
        0.03,
        f"{s} {label}",
        transform=axes[2].get_xaxis_transform(),
        fontsize=8,
        ha="right",
        color="#52514e",
    )
axes[2].set_yscale("log")
axes[0].text(
    0.99,
    0.97,
    "Out-of-sample",
    transform=axes[0].transAxes,
    ha="right",
    va="top",
    fontsize=8,
    color="#52514e",
)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=6, frameon=False, loc="upper center", fontsize=9)
fig.autofmt_xdate()
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(config.FIGURES / "03_daily_series.png", dpi=150)
plt.close(fig)

# %% Event day: DSV earnings intraday vs DSV typical in-sample day
ev_stock, ev_day = next(iter(config.EVENT_DAYS))
dsv = d[d["stock"] == ev_stock].copy()
dsv["bucket"] = dsv["mod"] // 15 * 15
is_ev = dsv["date"] == ev_day
typ = (
    dsv[(dsv["sample"] == "IS") & ~is_ev]
    .groupby(["date", "bucket"])
    .apply(
        lambda x: np.average(x["spread_bps"], weights=x["dur"]), include_groups=False
    )
    .groupby("bucket")
    .median()
)
evs = (
    dsv[is_ev]
    .groupby("bucket")
    .apply(
        lambda x: np.average(x["spread_bps"], weights=x["dur"]), include_groups=False
    )
)
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(typ.index, typ.values, color="#8a8984", label="Typical in-sample day (median)")
ax.plot(
    evs.index,
    evs.values,
    color=COLORS[ev_stock],
    label=f"{ev_day} ({config.EVENT_DAYS[(ev_stock, ev_day)]})",
)
ax.set_xticks(ticks, [f"{9 + t // 60:02d}:00" for t in ticks])
ax.set_title(
    f"{ev_stock} quoted spread, 15-min buckets (bps, time-weighted)",
    loc="left",
    fontsize=11,
)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(config.FIGURES / "03_event_dsv_earnings.png", dpi=150)
plt.close(fig)

# %% Tick constraint
tick = (
    d_is.groupby("stock")
    .apply(
        lambda x: pd.Series(
            {
                "tick_dkk": x["tick"].iloc[0],
                "median_price": x["mid"].median(),
                "tick_bps": 1e4 * x["tick"].iloc[0] / x["mid"].median(),
                "median_spread_bps": x["spread_bps"].median(),
                "median_spread_ticks": x["spread_ticks"].median(),
                "pct_trades_1tick": 100 * (x["spread_ticks"] == 1).mean(),
                "pct_time_1tick": 100
                * x.loc[x["spread_ticks"] == 1, "dur"].sum()
                / x["dur"].sum(),
                "pct_time_2tick": 100
                * x.loc[x["spread_ticks"] == 2, "dur"].sum()
                / x["dur"].sum(),
                "pct_time_3plus": 100
                * x.loc[x["spread_ticks"] >= 3, "dur"].sum()
                / x["dur"].sum(),
            }
        ),
        include_groups=False,
    )
    .reindex(list(config.STOCKS))
)
tick.round(2).to_csv(config.TABLES / "03_tick_constraint.csv")
print("Tick constraint (in-sample):\n", tick.round(2).to_string(), "\n")

hr = (
    d_is.assign(hour=d_is["mod"] // 30 * 30)
    .groupby(["stock", "hour"])
    .apply(
        lambda x: 100 * x.loc[x["spread_ticks"] == 1, "dur"].sum() / x["dur"].sum(),
        include_groups=False,
    )
)
fig, ax = plt.subplots(figsize=(9, 3.6))
for stock in config.STOCKS:
    ax.plot(
        hr.loc[stock].index,
        hr.loc[stock].values,
        color=COLORS[stock],
        marker="o",
        markersize=3,
        label=stock,
    )
ax.set_xticks(ticks, [f"{9 + t // 60:02d}:00" for t in ticks])
ax.set_ylim(0, 100)
ax.set_title(
    "Share of time the spread is one tick (%, 30-min buckets, in-sample)",
    loc="left",
    fontsize=11,
)
ax.legend(ncol=6, frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(config.FIGURES / "03_tick_share_intraday.png", dpi=150)
plt.close(fig)

# %% Design inputs: interval length and open/close treatment
rows = []
for stock in config.STOCKS:
    gs = g_is[g_is["stock"] == stock]
    for k in [1, 5, 15, 30]:
        n = gs.groupby(["date", gs["mod"] // k])["trades"].sum()
        rows.append(
            {
                "stock": stock,
                "interval_min": k,
                "median_trades": n.median(),
                "pct_empty": 100 * (n == 0).mean(),
                "pct_under_5_trades": 100 * (n < 5).mean(),
            }
        )
interval = (
    pd.DataFrame(rows)
    .pivot(index="stock", columns="interval_min")
    .reindex(list(config.STOCKS))
)
interval.round(1).to_csv(config.TABLES / "03_interval_activity.csv")
print("Trades per interval (in-sample):\n", interval.round(1).to_string(), "\n")

# spread/vol in the first and last 15/30 minutes relative to the 10:00-16:00 median
ref = intraday.reset_index()
mid_day = (
    ref[(ref["mod"] >= 60) & (ref["mod"] < 420)]
    .groupby("stock")[["spread_bps_tw", "vol_1min_bps"]]
    .median()
)
oc = {}
for label, lo, hi in [
    ("first_15", 0, 15),
    ("first_30", 0, 30),
    ("last_30", 445, 475),
    ("last_15", 460, 475),
]:
    w = (
        ref[(ref["mod"] >= lo) & (ref["mod"] < hi)]
        .groupby("stock")[["spread_bps_tw", "vol_1min_bps"]]
        .mean()
    )
    for col in w:
        oc[f"{col}_{label}_x_midday"] = w[col] / mid_day[col]
open_close = pd.DataFrame(oc).reindex(list(config.STOCKS))
open_close.round(2).to_csv(config.TABLES / "03_open_close_ratios.csv")
print(
    "Open/close relative to 10:00-16:00 median (in-sample):\n",
    open_close.round(2).T.to_string(),
)
