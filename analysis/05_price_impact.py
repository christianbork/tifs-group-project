# %% [markdown]
# # 05 Informed trading: price impact per stock-day / rolling window
# 1. Events = aggressive orders (order_id). Mid return for order k runs from the
#    mid at order k's first fill to the mid at order k+1's first fill (same day).
# 2. Hasbrouck (1991) VAR: mid return + order flow (sign; signed volume as a
#    robustness check), order flow ordered first, lags within the day only.
#    Lag length chosen by BIC in 5..10 on in-sample days, then fixed.
# 3. Permanent impact = cumulative mid response to a one-unit buy shock after
#    20 orders (bps). Per stock-day for VAR_DAILY stocks; rolling ROLL_DAYS-day
#    window (ending on day t) for VAR_ROLLING stocks.
# 4. Cross-check: effective = realised + price impact half-spread at +1 / +5 min,
#    averaged over the same windows; correlation of the two impact measures.
# Output for script 07: data/processed/impact_daily.parquet, where `impact_pred`
# on day t uses only data up to day t-1.

# %% Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tifs_group_project import config
from tifs_group_project.impact import (
    cum_irf,
    decomposition,
    fit_var,
    order_events,
    rolling_impact,
    windows,
)

H = 20  # IRF horizon in orders
LAGS = range(5, 11)
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

# %% Events, lag choice (in-sample only), VAR per window
rows, irfs, lag_rows = [], [], []
for stock in config.STOCKS:
    d = pd.read_parquet(config.PROCESSED / f"signed_{stock}.parquet")
    ev = order_events(d)
    is_days = ev["date"] <= config.IS_END
    # signed volume in units of the stock's median order size (in-sample), so x = +1 is a typical buy
    ev["svol"] = ev["signed_vol"] / ev.loc[is_days, "vol"].median()

    bic = {p: fit_var(ev[is_days], "sign", p)["bic"] for p in LAGS}
    p = min(bic, key=bic.get)
    lag_rows.append(
        {
            "stock": stock,
            "chosen_p": p,
            **{f"bic_p{k}": round(v, 1) for k, v in bic.items()},
        }
    )

    dec = decomposition(d)
    days = sorted(ev["date"].unique())
    n_days = 1 if stock in config.VAR_DAILY else config.ROLL_DAYS
    for end, win in windows(days, n_days):
        e = ev[ev["date"].isin(win)]
        v_sign, v_svol = fit_var(e, "sign", p), fit_var(e, "svol", p)
        irf = cum_irf(v_sign, H)
        irfs.append({"stock": stock, "date": end, **{h: irf[h] for h in range(H + 1)}})
        w = dec[dec["date"].isin(win)]
        rows.append(
            {
                "stock": stock,
                "date": end,
                "window_days": len(win),
                "n_orders": v_sign["n"],
                "lags": p,
                "var_impact_bps": irf[H],
                "var_impact_h10_bps": irf[10],
                "var_impact_svol_bps": cum_irf(v_svol, H)[H],
                "eff_hs_bps": w["eff_bps"].mean(),
                "impact_1m_bps": w["impact_1m_bps"].mean(),
                "impact_5m_bps": w["impact_5m_bps"].mean(),
                "realised_5m_bps": w["realised_5m_bps"].mean(),
            }
        )
    print(f"{stock}: {len(ev):,} orders, p={p}, {n_days}-day windows")

lags = pd.DataFrame(lag_rows).set_index("stock")
lags.to_csv(config.TABLES / "05_lag_choice.csv")
print("\nBIC lag choice (in-sample):\n", lags.to_string(), "\n")

# %% Impact table + predictor for day t+1 (no look-ahead)
imp = pd.DataFrame(rows)
imp["group"] = np.where(
    imp["stock"].isin(config.CONSTRAINED), "constrained", "unconstrained"
)
imp["sample"] = np.where(imp["date"] <= config.IS_END, "IS", "OOS")
# estimate through day t-1 -> predictor for day t: shift by one trading day within stock
all_days = pd.DataFrame(
    [(s, t) for s in config.STOCKS for t in sorted(imp["date"].unique())],
    columns=["stock", "date"],
)
imp = all_days.merge(imp, on=["stock", "date"], how="left")
for col in ["var_impact_bps", "var_impact_svol_bps", "impact_5m_bps", "eff_hs_bps"]:
    imp[col.replace("_bps", "_pred_bps")] = imp.groupby("stock")[col].shift(1)
imp.to_parquet(config.PROCESSED / "impact_daily.parquet", index=False)
imp.to_csv(config.TABLES / "05_impact_stock_day.csv", index=False, float_format="%.3f")

# %% Summary per stock
est = imp.dropna(subset=["var_impact_bps"])
summ = (
    est.groupby("stock")
    .agg(
        group=("group", "first"),
        windows=("date", "size"),
        lags=("lags", "first"),
        var_impact_med=("var_impact_bps", "median"),
        var_impact_sd=("var_impact_bps", "std"),
        var_h10_med=("var_impact_h10_bps", "median"),
        var_svol_med=("var_impact_svol_bps", "median"),
        impact_1m_med=("impact_1m_bps", "median"),
        impact_5m_med=("impact_5m_bps", "median"),
        eff_hs_med=("eff_hs_bps", "median"),
        realised_5m_med=("realised_5m_bps", "median"),
    )
    .reindex(list(config.STOCKS))
)
summ["pct_eff_spread_adverse_sel"] = 100 * summ["impact_5m_med"] / summ["eff_hs_med"]
summ["var_impact_ar1"] = est.groupby("stock").apply(
    lambda x: x["var_impact_bps"].autocorr(), include_groups=False
)
summ.round(3).to_csv(config.TABLES / "05_impact_summary.csv")
print(
    "Impact summary (bps, medians across windows):\n", summ.round(2).to_string(), "\n"
)

# %% Correlation between the VAR impact and the decomposition impact
corr_rows = []
for label, sub in [("all", est), *[(s, est[est["stock"] == s]) for s in config.STOCKS]]:
    for h in ["impact_1m_bps", "impact_5m_bps"]:
        corr_rows.append(
            {
                "sample": label,
                "decomp": h,
                "n": len(sub),
                "pearson": sub["var_impact_bps"].corr(sub[h]),
                "spearman": sub["var_impact_bps"].corr(sub[h], method="spearman"),
            }
        )
# within-stock: demean both by stock, so the correlation is not just the cross-stock level difference
dm = est[["var_impact_bps", "impact_5m_bps"]] - est.groupby("stock")[
    ["var_impact_bps", "impact_5m_bps"]
].transform("mean")
corr_rows.append(
    {
        "sample": "within-stock (demeaned)",
        "decomp": "impact_5m_bps",
        "n": len(dm),
        "pearson": dm["var_impact_bps"].corr(dm["impact_5m_bps"]),
        "spearman": dm["var_impact_bps"].corr(dm["impact_5m_bps"], method="spearman"),
    }
)
corr = pd.DataFrame(corr_rows)
corr.round(3).to_csv(config.TABLES / "05_impact_correlation.csv", index=False)
print(
    "Correlation, VAR impact vs decomposition impact:\n",
    corr.round(2).to_string(index=False),
    "\n",
)
print(
    "Rolling-window stocks (Pandora, GN, ISS) use overlapping windows, so their n overstates independent obs."
)

# %% Figures
irf = pd.DataFrame(irfs).set_index(["stock", "date"])
fig, ax = plt.subplots(figsize=(9, 4))
for stock in config.STOCKS:
    med = irf.loc[stock].median()
    ax.plot(
        med.index,
        med.values,
        color=COLORS[stock],
        marker="o",
        markersize=3,
        label=stock,
    )
ax.set_xlabel("Orders after a one-unit buy shock")
ax.set_xticks(range(0, H + 1, 2))
ax.set_ylabel("Cumulative mid response (bps)")
ax.set_title(
    "Hasbrouck VAR impulse response, median across windows", loc="left", fontsize=11
)
ax.legend(ncol=6, frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(config.FIGURES / "05_irf_cumulative.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
for stock in config.STOCKS:
    s = est[est["stock"] == stock]
    ax.plot(
        s["date"],
        s["var_impact_bps"],
        color=COLORS[stock],
        marker="o",
        markersize=3,
        label=stock,
    )
ax.axvspan(
    pd.Timestamp(config.IS_END) + pd.Timedelta(hours=12),
    est["date"].max() + pd.Timedelta(hours=12),
    color="#8a8984",
    alpha=0.12,
    linewidth=0,
)
ax.text(
    0.99,
    0.97,
    "Out-of-sample",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    color="#52514e",
)
ax.set_title(
    f"Permanent impact of a buy order (bps, {H}-order cumulative IRF); "
    f"rolling {config.ROLL_DAYS}-day for Pandora/GN/ISS",
    loc="left",
    fontsize=10,
)
ax.legend(
    ncol=6, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22)
)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(config.FIGURES / "05_impact_by_day.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
for stock in config.STOCKS:
    s = est[est["stock"] == stock]
    ax.scatter(
        s["impact_5m_bps"],
        s["var_impact_bps"],
        s=18,
        color=COLORS[stock],
        label=stock,
        edgecolor="white",
        linewidth=0.8,
    )
lim = [0, max(est["impact_5m_bps"].max(), est["var_impact_bps"].max()) * 1.05]
ax.plot(lim, lim, color="#8a8984", linewidth=0.8, linestyle="--")
ax.set_xlabel("Decomposition price impact at +5 min (bps)")
ax.set_ylabel("VAR permanent impact (bps)")
ax.set_title("Two impact measures per window (dashed = 45°)", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(config.FIGURES / "05_var_vs_decomposition.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Intraday rolling impact (joinable to bars_5min on stock + ts)
# At each 5-min bar start tau, re-estimated from orders in [tau - 60 min, tau)
# whose mid return is complete before tau (same day only; the 09:00 bar is NaN).
# - reg: mid return on sign_t..sign_{t-5}, impact = sum of coefficients (>= 50 orders)
# - var: Hasbrouck VAR(5), 20-order cumulative IRF (>= 100 orders)

# %% Rolling estimates
bars5 = pd.read_parquet(
    config.PROCESSED / "bars_5min.parquet", columns=["stock", "date", "ts", "sample"]
)
roll = []
for stock in config.STOCKS:
    ev = order_events(pd.read_parquet(config.PROCESSED / f"signed_{stock}.parquet"))
    starts = bars5.loc[bars5["stock"] == stock, "ts"]
    roll.append(rolling_impact(ev, starts).assign(stock=stock))
roll = bars5.merge(pd.concat(roll, ignore_index=True), on=["stock", "ts"], how="left")
roll[
    ["stock", "ts", "impact_roll_reg_bps", "impact_roll_var_bps", "roll_n_orders"]
].to_parquet(config.PROCESSED / "impact_intraday_5min.parquet", index=False)
cov = roll.groupby("stock")[["impact_roll_reg_bps", "impact_roll_var_bps"]].apply(
    lambda x: 100 * x.notna().mean()
)
print(
    "Coverage, % of 5-min bars with an estimate:\n",
    cov.reindex(list(config.STOCKS)).round(1).to_string(),
    "\n",
)

# %% Within-stock variation and persistence vs the daily VAR (in-sample)
r_is = roll[roll["sample"] == "IS"].copy()
daily_var = imp.set_index(["stock", "date"])["var_impact_bps"]
rows = []
for stock in config.STOCKS:
    s = r_is[r_is["stock"] == stock]
    dv = imp[(imp["stock"] == stock) & (imp["sample"] == "IS")][
        "var_impact_bps"
    ].dropna()
    row = {
        "stock": stock,
        "daily_var_mean": dv.mean(),
        "daily_var_sd_across_days": dv.std(),
        "daily_var_ar1": dv.autocorr(),
    }
    for m in ["impact_roll_reg_bps", "impact_roll_var_bps"]:
        x = s[m]
        within = x - s.groupby("date")[m].transform("mean")
        g = s.groupby("date")[m]
        day_mean = g.mean()
        tag = "reg" if "reg" in m else "var"
        row |= {
            f"{tag}_mean": x.mean(),
            f"{tag}_sd": x.std(),
            f"{tag}_pct_var_within_day": 100 * within.var() / x.var(),
            f"{tag}_ac_5min": x.corr(g.shift(1)),
            f"{tag}_ac_60min": x.corr(g.shift(12)),  # non-overlapping windows
            f"{tag}_daymean_ar1": day_mean.autocorr(),
            f"{tag}_corr_daymean_vs_daily_var": day_mean.corr(
                daily_var.loc[stock].reindex(day_mean.index)
            ),
        }
    rows.append(row)
persist = pd.DataFrame(rows).set_index("stock")
persist.round(3).to_csv(config.TABLES / "05_intraday_impact_persistence.csv")
print(
    "Intraday rolling impact vs daily VAR (in-sample):\n",
    persist.round(2).T.to_string(),
    "\n",
)

# %% Figure: intraday profile of the rolling impact (median by bar, in-sample)
prof = r_is.assign(minute=(r_is["ts"] - r_is["date"]).dt.total_seconds() / 60 - 540)
prof = prof.groupby(["stock", "minute"])["impact_roll_reg_bps"].median()
fig, ax = plt.subplots(figsize=(9, 4))
for stock in config.STOCKS:
    ax.plot(
        prof.loc[stock].index, prof.loc[stock].values, color=COLORS[stock], label=stock
    )
ticks = np.arange(0, 481, 60)
ax.set_xticks(ticks, [f"{9 + t // 60:02d}:00" for t in ticks])
ax.set_ylabel("bps")
ax.set_title(
    "Rolling 60-min impact at each 5-min bar (median across in-sample days)",
    loc="left",
    fontsize=11,
)
ax.legend(
    ncol=6, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12)
)
fig.tight_layout()
fig.savefig(config.FIGURES / "05_intraday_impact_profile.png", dpi=150)
plt.close(fig)
