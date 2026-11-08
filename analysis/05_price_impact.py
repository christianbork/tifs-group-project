# %% [markdown]
# # 05 Informed trading: price impact per stock-day / rolling window
# 1. Events = aggressive orders (order_id). Mid return for order k runs from the
#    mid at order k's first fill to the mid at order k+1's first fill (same day).
# 2. Hasbrouck (1991) VAR: mid return + order flow (sign; signed volume as a
#    robustness check), order flow ordered first, lags within the day only.
#    Lag length (v2): v1 chose p by BIC in 5..10 and every stock hit the upper bound.
#    BIC keeps falling up to p = 40 (n ~ 10^4-10^5 orders makes BIC favour ever longer
#    lags), so p is fixed at 20, where the in-sample impact has stabilised: moving to
#    p = 40 changes it by < 5% for every stock. The BIC path and this sensitivity are
#    saved in 05_lag_choice.csv (in-sample days only).
# 3. Permanent impact = cumulative mid response to a one-unit buy shock after
#    20 orders (bps). Per stock-day for VAR_DAILY stocks; rolling ROLL_DAYS-day
#    window (ending on day t) for VAR_ROLLING stocks.
# 3b. v2: day-block bootstrap 95% CI for each stock's in-sample permanent impact
#    (AE blueprint: irf(..., boot = TRUE)); 500 resamples of whole days.
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
    bootstrap_impact,
    cum_irf,
    decomposition,
    fit_var,
    order_events,
    rolling_impact,
    windows,
)

H = 20  # IRF horizon in orders
LAGS = [5, 10, 15, 20, 30, 40]  # lag-sensitivity grid
P_FIXED = 20
N_BOOT = 500
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
rows, irfs, lag_rows, boot_rows = [], [], [], []
for stock in config.STOCKS:
    d = pd.read_parquet(config.PROCESSED / f"signed_{stock}.parquet")
    ev = order_events(d)
    is_days = ev["date"] <= config.IS_END
    # signed volume in units of the stock's median order size (in-sample), so x = +1 is a typical buy
    ev["svol"] = ev["signed_vol"] / ev.loc[is_days, "vol"].median()

    fits = {k: fit_var(ev[is_days], "sign", k) for k in LAGS}
    p = P_FIXED
    lag_rows.append(
        {
            "stock": stock,
            "chosen_p": p,
            "bic_min_at": min(fits, key=lambda k: fits[k]["bic"]),
            **{f"bic_p{k}": round(v["bic"], 1) for k, v in fits.items()},
            **{f"impact_p{k}": round(cum_irf(v, max(H, k))[H], 3) for k, v in fits.items()},
        }
    )

    # bootstrap CI for the pooled in-sample permanent impact (sign VAR, chosen p)
    boot_rows.append(
        {"stock": stock, "lags": p, **bootstrap_impact(ev[is_days], "sign", p, H, n_boot=N_BOOT, seed=0)}
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
imp_cols = [f"impact_p{k}" for k in LAGS]
print("\nLag choice (in-sample): BIC minimum and impact (bps) by lag length:\n",
      lags[["bic_min_at", *imp_cols]].to_string(), "\n")
rel = (lags[f"impact_p{P_FIXED}"] / lags[f"impact_p{max(LAGS)}"] - 1).abs()
print(f"Max relative change in impact from p={P_FIXED} to p={max(LAGS)}: {100 * rel.max():.1f}%\n")
assert rel.max() < 0.05, "impact not yet stable at P_FIXED"


boot = pd.DataFrame(boot_rows).set_index("stock")
boot.round(3).to_csv(config.TABLES / "05_impact_bootstrap.csv")
print(
    f"Pooled in-sample permanent impact with day-block bootstrap 95% CI ({N_BOOT} resamples, bps):\n",
    boot.round(2).to_string(),
    "\n",
)

# %% Impact table + predictor for day t+1 (no look-ahead)
imp = pd.DataFrame(rows)
# estimate through day t-1 -> predictor for day t: shift by one trading day within stock
all_days = pd.DataFrame(
    [(s, t) for s in config.STOCKS for t in sorted(imp["date"].unique())],
    columns=["stock", "date"],
)
imp = all_days.merge(imp, on=["stock", "date"], how="left")
# labels after the merge, so the first days without a rolling estimate are labelled too
imp["group"] = np.where(
    imp["stock"].isin(config.CONSTRAINED), "constrained", "unconstrained"
)
imp["sample"] = config.sample_label(imp["date"])
for col in ["var_impact_bps", "var_impact_svol_bps", "impact_5m_bps", "eff_hs_bps"]:
    imp[col.replace("_bps", "_pred_bps")] = imp.groupby("stock")[col].shift(1)
imp.to_parquet(config.PROCESSED / "impact_daily.parquet", index=False)
imp.to_csv(config.TABLES / "05_impact_stock_day.csv", index=False, float_format="%.3f")

# %% Summary per stock (October only; the forward week is kept out of every descriptive table)
est = imp[imp["sample"] != "FWD"].dropna(subset=["var_impact_bps"])
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

fig, ax = plt.subplots(figsize=(7, 3.4))
y = np.arange(len(boot))[::-1]
ax.errorbar(
    boot["impact"],
    y,
    xerr=[boot["impact"] - boot["ci_low"], boot["ci_high"] - boot["impact"]],
    fmt="D",
    color="#2a78d6",
    ecolor="#8a8984",
    capsize=3,
)
ax.set_yticks(y, boot.index)
ax.set_xlabel("Permanent impact of a buy order (bps), 95% day-block bootstrap CI")
ax.set_xlim(0, None)
ax.set_title("Hasbrouck VAR impact, pooled in-sample days", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(config.FIGURES / "05_impact_bootstrap.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Intraday rolling impact (joinable to bars_5min on stock + ts)
# At each 5-min bar start tau: Hasbrouck VAR(5) on the last N_ROLL orders whose
# mid return is complete before tau (window may span the previous day; returns
# and lags never cross a day), 20-order cumulative IRF. Smoothed with an EWMA
# over bars (half-life EWM_HALFLIFE bars, fixed a priori), which uses only
# current and past estimates.

# %% Rolling estimates
N_ROLL, EWM_HALFLIFE = 300, 6
bars5 = pd.read_parquet(
    config.PROCESSED / "bars_5min.parquet", columns=["stock", "date", "ts", "sample"]
)
roll = []
for stock in config.STOCKS:
    ev = order_events(pd.read_parquet(config.PROCESSED / f"signed_{stock}.parquet"))
    r = rolling_impact(ev, bars5.loc[bars5["stock"] == stock, "ts"], n_orders=N_ROLL)
    r["impact_roll_ewm_bps"] = (
        r["impact_roll_var_bps"].ewm(halflife=EWM_HALFLIFE, ignore_na=True).mean()
    )
    roll.append(r.assign(stock=stock))
roll = bars5.merge(pd.concat(roll, ignore_index=True), on=["stock", "ts"], how="left")
roll["roll_window_min"] = roll["roll_window_span"].dt.total_seconds() / 60
roll[
    ["stock", "ts", "impact_roll_var_bps", "impact_roll_ewm_bps", "roll_window_min"]
].to_parquet(config.PROCESSED / "impact_intraday_5min.parquet", index=False)
roll_oct = roll[roll["sample"] != "FWD"]
cov = (
    roll_oct.groupby("stock")
    .agg(
        pct_bars_raw=("impact_roll_var_bps", lambda x: 100 * x.notna().mean()),
        pct_bars_ewm=("impact_roll_ewm_bps", lambda x: 100 * x.notna().mean()),
        first_estimate=(
            "ts",
            lambda x: x[roll_oct.loc[x.index, "impact_roll_var_bps"].notna()].min(),
        ),
        median_window_min=("roll_window_min", "median"),
        pct_windows_span_prev_day=("roll_window_min", lambda x: 100 * (x > 475).mean()),
    )
    .reindex(list(config.STOCKS))
)
cov.to_csv(config.TABLES / "05_intraday_impact_coverage.csv", float_format="%.1f")
print(f"Coverage of the last-{N_ROLL}-orders impact:\n", cov.round(1).to_string(), "\n")

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
    for m, tag in [("impact_roll_var_bps", "raw"), ("impact_roll_ewm_bps", "ewm")]:
        x = s[m]
        within = x - s.groupby("date")[m].transform("mean")
        day_mean = s.groupby("date")[m].mean()
        row |= {
            f"{tag}_mean": x.mean(),
            f"{tag}_sd": x.std(),
            f"{tag}_pct_var_within_day": 100 * within.var() / x.var(),
            f"{tag}_ac_1bar": x.autocorr(1),
            f"{tag}_ac_12bars": x.autocorr(12),
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

# %% Figure: intraday profile of the smoothed rolling impact (median by bar, in-sample)
prof = r_is.assign(minute=(r_is["ts"] - r_is["date"]).dt.total_seconds() / 60 - 540)
prof = prof.groupby(["stock", "minute"])["impact_roll_ewm_bps"].median()
fig, ax = plt.subplots(figsize=(9, 4))
for stock in config.STOCKS:
    ax.plot(
        prof.loc[stock].index, prof.loc[stock].values, color=COLORS[stock], label=stock
    )
ticks = np.arange(0, 481, 60)
ax.set_xticks(ticks, [f"{9 + t // 60:02d}:00" for t in ticks])
ax.set_ylabel("bps")
ax.set_title(
    f"Rolling impact, last {N_ROLL} orders, EWMA-smoothed (median across in-sample days)",
    loc="left",
    fontsize=11,
)
ax.legend(
    ncol=6, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12)
)
fig.tight_layout()
fig.savefig(config.FIGURES / "05_intraday_impact_profile.png", dpi=150)
plt.close(fig)
