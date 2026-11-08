# %% [markdown]
# # 06 Volatility forecasts per 5-min bar
# 1. 1-min mid returns (first minute of each day excluded = no overnight move),
#    deseasonalised by an in-sample time-of-day factor that follows the US open.
# 2. GARCH(1,1) and GJR-GARCH on the deseasonalised returns, fitted on in-sample
#    days, then run with fixed parameters over all days.
# 3. Benchmarks: rolling 30-min std, ATR(14 bars), intraday HAR-RV (bar, hour, day).
# 4. Every forecast for bar t is made at the end of bar t-1. OOS evaluation vs
#    realised 5-min volatility: RMSE (vol), QLIKE (variance), Mincer-Zarnowitz.
# 5. v2 (AE Lab 8 steps): before fitting, Ljung-Box on returns and squared returns and
#    Engle's ARCH-LM test; after fitting, Ljung-Box / ARCH-LM on the standardised residuals
#    of GARCH and GJR, with normal and Student-t errors (in-sample days only).
# Output for script 07: data/processed/vol_forecast_5min.parquet (joins to bars_5min on stock + ts).

# %% Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tifs_group_project import config
from tifs_group_project.volatility import (
    arch_lm,
    bar_forecast,
    garch_diagnostics,
    ljung_box,
    garch_variance_forecasts,
    mincer_zarnowitz,
    qlike,
    seasonal_variance,
)

MODELS = ["garch", "gjr", "roll30", "atr", "har"]
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
b1 = pd.read_parquet(config.PROCESSED / "bars_1min.parquet")

# %% Forecasts per stock
out, params, seas, diag = [], [], [], []
for stock in config.STOCKS:
    m1 = b1[b1["stock"] == stock].sort_values("ts").reset_index(drop=True)
    m1["e"] = seasonal_variance(m1)
    m1["bar_ts"] = m1["ts"].dt.floor("5min")
    seas.append(m1[["stock", "date", "minute_of_day", "e", "sample"]])

    v = m1[m1["ret_bps"].notna()].reset_index(drop=True)
    z = (v["ret_bps"] / np.sqrt(v["e"])).to_numpy()
    n_is = int((v["sample"] == "IS").sum())

    g = v.groupby("bar_ts")
    bars = pd.DataFrame(
        {
            "rv2": g["ret_bps"].apply(lambda r: (r**2).sum()),
            "E": g["e"].sum(),
            "date": g["date"].first(),
            "sample": g["sample"].first(),
        }
    )

    # AE Lab 8 diagnostics on in-sample deseasonalised returns
    z_is = z[:n_is]
    lb_q, lb_p = ljung_box(z_is, 10)
    lb2_q, lb2_p = ljung_box(z_is**2, 10)
    lm, lm_p = arch_lm(z_is, 10)
    diag.append(
        {"stock": stock, "model": "returns (before)", "pct_zero_returns": 100 * (z_is == 0).mean(), "lb_std_q": lb_q, "lb_std_p": lb_p,
         "lb_sq_std_q": lb2_q, "lb_sq_std_p": lb2_p, "arch_lm_std": lm, "arch_lm_std_p": lm_p}
    )
    for name, gjr in [("garch", False), ("gjr", True)]:
        for dist in ["normal", "t"]:
            diag.append(
                {"stock": stock, "model": f"{name} ({dist})", **garch_diagnostics(z, n_is, gjr, dist)}
            )

    # GARCH / GJR: fixed in-sample parameters, forecasts from every origin
    for name, gjr in [("garch", False), ("gjr", True)]:
        f, p = garch_variance_forecasts(z, n_is, gjr)
        bars[f"fc_{name}"] = bar_forecast(v, f)
        params.append(
            {
                "stock": stock,
                "model": name,
                **p.to_dict(),
                "persistence": p["alpha[1]"]
                + p["beta[1]"]
                + 0.5 * p.get("gamma[1]", 0.0),
            }
        )

    # rolling 30-min variance of deseasonalised returns (crosses days: yesterday's close is known)
    s30 = pd.Series(z**2).rolling(30, min_periods=10).mean().to_numpy()
    bars["fc_roll30"] = bar_forecast(v, np.repeat(s30[:, None], 5, axis=1))

    # HAR on deseasonalised bar RV: previous bar, previous hour (12 bars), previous day (95 bars)
    rvz = bars["rv2"] / bars["E"]
    har = pd.DataFrame(
        {
            "y": rvz,
            "b": rvz.shift(1),
            "h": rvz.shift(1).rolling(12).mean(),
            "d": rvz.shift(1).rolling(95).mean(),
        }
    )
    fit_rows = har.dropna().index.intersection(bars.index[bars["sample"] == "IS"])
    ols = sm.OLS(
        har.loc[fit_rows, "y"], sm.add_constant(har.loc[fit_rows, ["b", "h", "d"]])
    ).fit()
    pred = ols.predict(sm.add_constant(har[["b", "h", "d"]], has_constant="add"))
    bars["fc_har"] = bars["E"] * pred.clip(lower=0.05 * rvz[fit_rows].mean())
    params.append(
        {
            "stock": stock,
            "model": "har",
            **{f"har_{k}": val for k, val in ols.params.items()},
        }
    )

    # ATR: true range of the mid per bar (from 1-min closing mids; first bar of a day uses its own range only)
    gm = m1.groupby("bar_ts")["mid_end"]
    hi, lo, close = gm.max(), gm.min(), gm.last()
    prev = close.shift(1).where(
        pd.Series(close.index.date, index=close.index)
        == pd.Series(close.index.date, index=close.index).shift(1)
    )
    tr = (
        (np.maximum(hi, prev.fillna(hi)) - np.minimum(lo, prev.fillna(lo)))
        / close.shift(1).fillna(close)
        * 1e4
    )
    trz = (tr / np.sqrt(bars["E"].reindex(tr.index))).reindex(bars.index)
    raw = np.sqrt(bars["E"]) * trz.rolling(14, min_periods=5).mean().shift(1)
    isr = (bars["sample"] == "IS") & raw.notna()
    c = (np.sqrt(bars.loc[isr, "rv2"]) * raw[isr]).sum() / (
        raw[isr] ** 2
    ).sum()  # scale set in-sample
    bars["fc_atr"] = (c * raw) ** 2
    params.append({"stock": stock, "model": "atr", "atr_scale": c})

    out.append(bars.rename_axis("ts").reset_index().assign(stock=stock))
    print(f"{stock}: {len(v):,} 1-min returns ({n_is:,} in-sample), {len(bars):,} bars")

fc = pd.concat(out, ignore_index=True)
params = pd.DataFrame(params)
params.round(4).to_csv(config.TABLES / "06_model_params.csv", index=False)
print(
    "\nGARCH / GJR parameters (in-sample):\n",
    params[params["model"].isin(["garch", "gjr"])]
    .drop(columns=[c for c in params if c.startswith(("har_", "atr"))])
    .round(3)
    .to_string(index=False),
    "\n",
)

# %% AE diagnostics table
diag = pd.DataFrame(diag)
# Student-t fits with nu at the lower bound are degenerate: thin, tick-constrained stocks
# have a large mass of zero 1-min mid returns, which the t density 'explains' with nu -> 2
diag["degenerate_t"] = diag["nu"] <= 2.1
diag.round(4).to_csv(config.TABLES / "06_diagnostics.csv", index=False)
print(
    "AE Lab 8 diagnostics, in-sample (Ljung-Box Q(10) and ARCH-LM(10); 'before' = deseasonalised returns,\n"
    "otherwise standardised residuals; p < 0.05 = structure left):\n",
    diag[["stock", "model", "pct_zero_returns", "lb_std_p", "lb_sq_std_p", "arch_lm_std_p", "bic", "nu", "degenerate_t"]].round(4).to_string(index=False),
    "\n",
)

# %% Save forecasts for script 07 (bps; *_ratio = forecast / routine seasonal level = "unusual" vol)
save = fc[["stock", "ts"]].copy()
save["vol_seasonal_bps"] = np.sqrt(fc["E"])
for mdl in MODELS:
    save[f"vol_fc_{mdl}_bps"] = np.sqrt(fc[f"fc_{mdl}"])
    save[f"vol_fc_{mdl}_ratio"] = save[f"vol_fc_{mdl}_bps"] / save["vol_seasonal_bps"]
save.to_parquet(config.PROCESSED / "vol_forecast_5min.parquet", index=False)

# check: realised RV here equals rv_bps in bars_5min
b5 = pd.read_parquet(
    config.PROCESSED / "bars_5min.parquet", columns=["stock", "ts", "rv_bps"]
)
chk = fc.merge(b5, on=["stock", "ts"])
print(
    f"RV check vs bars_5min: max abs diff {np.abs(np.sqrt(chk['rv2']) - chk['rv_bps']).max():.2e} bps\n"
)

# %% OOS evaluation (common sample: every model has a forecast)
ev = fc[(fc["sample"] == "OOS")].dropna(subset=[f"fc_{m}" for m in MODELS])
is_event = pd.Series(
    [(s, str(d.date())) in config.EVENT_DAYS for s, d in zip(ev["stock"], ev["date"])],
    index=ev.index,
)
rows = []
in_s = fc[fc["sample"] == "IS"].dropna(subset=[f"fc_{m}" for m in MODELS])
for subset, sub in [
    ("OOS all days", ev),
    ("OOS ex event days", ev[~is_event]),
    ("IS (fit period, for reference)", in_s),
]:
    for stock in ["All", *config.STOCKS]:
        s = sub if stock == "All" else sub[sub["stock"] == stock]
        for mdl in MODELS:
            f2 = s[f"fc_{mdl}"]
            rows.append(
                {
                    "subset": subset,
                    "stock": stock,
                    "model": mdl,
                    "n": len(s),
                    "rmse_vol_bps": np.sqrt(
                        ((np.sqrt(s["rv2"]) - np.sqrt(f2)) ** 2).mean()
                    ),
                    "qlike": qlike(s["rv2"], f2),
                    **mincer_zarnowitz(s["rv2"], f2),
                }
            )
evaluation = pd.DataFrame(rows)
evaluation.round(4).to_csv(config.TABLES / "06_evaluation.csv", index=False)
print(
    "OOS evaluation (5-min bars, 22-30 Oct):\n",
    evaluation.round(3).to_string(index=False),
    "\n",
)
for col, label in [("qlike", "QLIKE"), ("rmse_vol_bps", "RMSE (bps)")]:
    pivot = evaluation.pivot_table(
        index=["subset", "stock"], columns="model", values=col
    )[MODELS]
    pivot["best"] = pivot.idxmin(axis=1)
    pivot.round(3).to_csv(config.TABLES / f"06_eval_{col}.csv")
    print(
        f"OOS {label} by model (lower is better):\n", pivot.round(3).to_string(), "\n"
    )

# %% Figures
seas = pd.concat(seas, ignore_index=True)
fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
for ax, stock in zip(axes.flat, config.STOCKS):
    s = seas[seas["stock"] == stock]
    for day, ls, label in [
        ("2026-10-20", "-", "In-sample day (US open 15:30)"),
        ("2026-10-27", "--", "OOS day (US open 14:30)"),
    ]:
        dd = s[s["date"] == day]
        ax.plot(
            dd["minute_of_day"],
            np.sqrt(dd["e"]),
            color=COLORS[stock],
            linestyle=ls,
            label=label,
        )
    ax.set_title(stock, loc="left", fontsize=10)
    ax.set_xticks([0, 120, 240, 360, 480], ["09", "11", "13", "15", "17"])
axes[0, 0].legend(frameon=False, fontsize=8)
fig.supylabel("Expected 1-min vol (bps)")
fig.suptitle(
    "Seasonal volatility factor: the US-open bump moves with the US open",
    x=0.01,
    ha="left",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(config.FIGURES / "06_seasonal_factor.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
for ax, (stock, day) in zip(axes, [("Novo", "2026-10-27"), ("ISS", "2026-10-29")]):
    s = fc[(fc["stock"] == stock) & (fc["date"] == day)]
    x = (s["ts"] - s["ts"].dt.normalize()).dt.total_seconds() / 60 - 540
    ax.bar(x, np.sqrt(s["rv2"]), width=4, color="#c3c2b7", label="Realised 5-min vol")
    ax.plot(
        x, np.sqrt(s["fc_gjr"]), color=COLORS[stock], label="GJR forecast (made at t-1)"
    )
    ax.plot(
        x,
        np.sqrt(s["fc_har"]),
        color="#52514e",
        linestyle="--",
        linewidth=1.2,
        label="HAR forecast",
    )
    ax.set_title(f"{stock}, {day} (out-of-sample)", loc="left", fontsize=10)
    ax.set_ylabel("bps")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
ticks = np.arange(0, 481, 60)
axes[-1].set_xticks(ticks, [f"{9 + t // 60:02d}:00" for t in ticks])
fig.tight_layout()
fig.savefig(config.FIGURES / "06_forecast_examples.png", dpi=150)
plt.close(fig)
