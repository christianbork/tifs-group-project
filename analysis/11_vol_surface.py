# %% [markdown]
# # 11 Intraday volatility surface: sampling horizon x time of day
# A volatility estimate from intraday mid returns depends on how often you sample. At
# very short horizons the bid-ask bounce and discrete ticks add "transitory" variance that
# is not there at longer horizons; the realised variance per minute falls as the sampling
# interval grows and then flattens at the "fundamental" level (the volatility signature
# plot of Andersen, Bollerslev, Diebold and Labys, 2000). Here the signature is computed
# for every in-sample day and every hour of the day, which gives a surface
# sigma(interval, hour) per stock. Two uses in the report:
#
# - how much of the 1-min volatility that feeds the HAR / GARCH models is microstructure
#   noise, by stock (expected: more in the tick-constrained stocks, where the tick is a
#   large share of the price), and at which interval the signature flattens;
# - whether the time-of-day pattern is the same at all horizons (it is not at the open).
#
# Mid prices are the trade-attached mids (last mid before each grid point, within the day),
# in-sample days only, continuous session 09:00-16:55. Outputs: 11_vol_signature.csv,
# 11_vol_surface_by_hour.csv, 11_vol_surface.png.

# %% Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tifs_group_project import config

INTERVALS_S = [5, 15, 30, 60, 120, 300, 600, 900, 1800]  # sampling intervals in seconds
HOURS = list(range(9, 17))  # 09:00-09:59 ... 16:00-16:54
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def day_grid_rv(mid: pd.Series, day: pd.Timestamp, step_s: int) -> pd.DataFrame:
    """Realised variance of log-mid returns per hour bucket on a step_s grid (previous tick)."""
    start = day + pd.Timedelta(config.CONT_START)
    end = day + pd.Timedelta(config.LAST_CONT_TRADE)
    grid = pd.date_range(start, end, freq=f"{step_s}s", inclusive="both")
    m = mid[~mid.index.duplicated(keep="last")].sort_index()
    sampled = m.reindex(m.index.union(grid)).ffill().reindex(grid)
    r = np.log(sampled).diff().dropna() * 1e4  # bps
    hour = r.index.hour
    out = pd.DataFrame({"hour": hour, "r2": r.to_numpy() ** 2})
    agg = out.groupby("hour")["r2"].agg(["sum", "size"])
    agg["rv_per_min"] = agg["sum"] / (agg["size"] * step_s / 60)  # variance per minute (bps^2)
    return agg


# %% Compute the surface per stock (in-sample days)
rows = []
for stock in config.STOCKS:
    d = pd.read_parquet(config.PROCESSED / f"signed_{stock}.parquet", columns=["ts", "date", "sample", "mid"])
    d = d[(d["sample"] == "IS") & d["mid"].notna()]
    for day, g in d.groupby("date"):
        mid = pd.Series(g["mid"].to_numpy(), index=pd.DatetimeIndex(g["ts"]))
        for step in INTERVALS_S:
            agg = day_grid_rv(mid, pd.Timestamp(day), step)
            for h, r_ in agg.iterrows():
                rows.append({"stock": stock, "date": day, "interval_s": step, "hour": int(h), "rv_per_min": r_["rv_per_min"]})
surf = pd.DataFrame(rows)

# average over days -> sigma per sqrt(minute) in bps, by stock x interval x hour, and whole day
by_hour = (
    surf.groupby(["stock", "interval_s", "hour"])["rv_per_min"].mean().pipe(np.sqrt).rename("sigma_bps_per_sqrt_min").reset_index()
)
by_hour.round(3).to_csv(config.TABLES / "11_vol_surface_by_hour.csv", index=False)

full = surf.groupby(["stock", "interval_s"])["rv_per_min"].mean().pipe(np.sqrt).unstack("interval_s")
full.columns = [f"{c}s" if c < 60 else f"{c // 60}min" for c in full.columns]
full = full.reindex(list(config.STOCKS))
ref = full["15min"]
sig = full.copy()
# noise share = part of the variance at a short interval that is not there at 15 min: 1 - sigma_15^2 / sigma_h^2
sig["noise_share_5s_pct"] = 100 * (1 - (full["15min"] ** 2) / (full["5s"] ** 2))
sig["noise_share_1min_pct"] = 100 * (1 - (full["15min"] ** 2) / (full["1min"] ** 2))
sig["ratio_5s_to_15min"] = full["5s"] / full["15min"]
# first interval at which sigma is within 5% of the 15-min level (where the signature flattens)
flat = []
for stock in full.index:
    lvl = full.loc[stock, "15min"]
    ok = [c for c in full.columns if full.loc[stock, c] <= 1.05 * lvl]
    flat.append(ok[0] if ok else "none")
sig["flattens_at"] = flat
tick = pd.read_csv(config.TABLES / "03_tick_constraint.csv").set_index("stock")
sig["tick_bps"] = tick["tick_bps"]
sig["half_spread_bps"] = tick["median_spread_bps"] / 2
sig["pct_time_1tick"] = tick["pct_time_1tick"]
# excess variance per 5-s sample above the 15-min level, as an RMS in bps: a pure bid-ask bounce between
# the two sides of a one-tick spread would give about half a spread per switch, so compare with half_spread_bps
sig["excess_rms_5s_bps"] = np.sqrt(np.clip((full["5s"] ** 2 - full["15min"] ** 2) * (5 / 60), 0, None))
sig.round(3).to_csv(config.TABLES / "11_vol_signature.csv")
print("Volatility signature (bps per sqrt-minute, in-sample mean over days):\n", full.round(2).to_string(), "\n")
print(
    "Microstructure noise share of variance vs the 15-min level, and tick size:\n",
    sig[["noise_share_5s_pct", "noise_share_1min_pct", "flattens_at", "excess_rms_5s_bps", "half_spread_bps", "tick_bps", "pct_time_1tick"]].round(2).to_string(),
    "\n",
)

# %% Figure: surface per stock, sigma relative to the stock's 15-min full-day level
fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
labels = [f"{s}s" if s < 60 else f"{s // 60}m" for s in INTERVALS_S]
vals = []
for ax, stock in zip(axes.flat, config.STOCKS):
    m = by_hour[by_hour["stock"] == stock].pivot(index="interval_s", columns="hour", values="sigma_bps_per_sqrt_min")
    m = m.reindex(index=INTERVALS_S, columns=HOURS) / ref[stock]
    vals.append(m.to_numpy())
vmax = np.nanpercentile(np.concatenate([v.ravel() for v in vals]), 98)
for ax, stock, m in zip(axes.flat, config.STOCKS, vals):
    im = ax.imshow(np.log2(m), aspect="auto", origin="lower", cmap="viridis", vmin=-1, vmax=np.log2(vmax))
    ax.set_yticks(range(len(INTERVALS_S)), labels)
    ax.set_xticks(range(len(HOURS)), [f"{h:02d}" for h in HOURS])
    grp = "C" if stock in config.CONSTRAINED else "U"
    ax.set_title(f"{stock} ({grp}), 15-min level = {ref[stock]:.1f} bps/√min", loc="left", fontsize=9)
    ax.grid(False)
for ax in axes[1]:
    ax.set_xlabel("hour of day (CPH)")
for ax in axes[:, 0]:
    ax.set_ylabel("sampling interval")
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("log2( σ / σ_15min, whole day )")
cb.set_ticks([-1, 0, 1, 2, 3])
cb.set_ticklabels(["½×", "1×", "2×", "4×", "8×"])
fig.suptitle(
    "Intraday volatility surface: realised volatility per √minute by sampling interval and hour (in-sample days)",
    x=0.01,
    ha="left",
    fontsize=11,
)
fig.savefig(config.FIGURES / "11_vol_surface.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 11_vol_surface.png")
