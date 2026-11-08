# %% [markdown]
# # 09 Forward test: 2-6 November 2026, evaluated once
# The out-of-sample window (22-30 Oct) was looked at three times (script 07, then the two
# refinements in 08). The November exports were pulled on 8 Nov, after every model and
# hyper-parameter was fixed, so they are the one genuinely unseen week (Student Guide A.5
# "forward testing"; Lecture 7: out-of-sample data should be used once and never iterated on).
#
# Rules: every model is re-fitted on in-sample days only (1-21 Oct) with the specification
# and hyper-parameters chosen in 07/08; nothing is tuned here. The FWD rows enter only as
# prediction targets. Do not edit models after reading this script's output.
#
# Outputs: 09_fwd_rmse.csv, 09_fwd_diebold_mariano.csv, 09_fwd_direction.csv,
# 09_fwd_vs_oos.csv, 09_fwd_by_day.csv, 09_fwd_rmse.png.

# %% Setup
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tifs_group_project import config
from tifs_group_project.spread_model import (
    build_panel,
    coarse_bucket,
    dm_test,
    fit_ols,
    nw_lags,
    predict,
)

SEED = 42
warnings.filterwarnings("ignore", message=".*eval_set.*")
IMPACT = ["impact_roll_ewm_bps", "impact_prevday_bps"]
VOL = ["log_vol_har"]
CONTROLS = ["lag_log_trades", "lag_log_turnover", "lag_abs_ofi"]
KEY = IMPACT + VOL
GROUPS = {"constrained": config.CONSTRAINED, "unconstrained": config.UNCONSTRAINED}
GROUP_COLORS = {"constrained": "#2a78d6", "unconstrained": "#eb6834"}
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e5e4e0",
        "grid.linewidth": 0.6,
    }
)

panel = build_panel("5min")
g = panel.groupby("stock")
panel["lag_quoted_hs"] = g["quoted_hs_bps"].shift(1)
panel["lag_ofi"] = g["ofi"].shift(1).fillna(0.0)
panel["bucket"] = coarse_bucket(panel["minute_of_day"], panel["min_since_us_open"])
is_ = panel[panel["sample"] == "IS"]
L = nw_lags(int(is_.groupby("stock").size().median()))

fwd = (
    panel[panel["sample"] == "FWD"]
    .dropna(subset=["quoted_hs_bps", "bench_yday", "lag_quoted_hs", *KEY, *CONTROLS])
    .copy()
)
assert len(fwd), (
    "no forward-test rows: run 01-06 with the November exports in data/raw first"
)
assert fwd["date"].min() >= pd.Timestamp(config.FWD_START)
is_event = pd.Series(
    [
        (s, str(d.date())) in config.EVENT_DAYS
        for s, d in zip(fwd["stock"], fwd["date"])
    ],
    index=fwd.index,
)
print("=" * 100)
print(
    "FORWARD TEST (2-6 Nov 2026). Models fitted on 1-21 Oct only; these rows are used once."
)
print(
    f"  {len(fwd):,} stock-bars, {fwd['date'].nunique()} days, "
    f"{int(is_event.sum()):,} bars on event days: "
    + ", ".join(
        f"{s} {d}" for (s, d), lab in config.EVENT_DAYS.items() if d >= config.FWD_START
    )
)
print("=" * 100, "\n")

# %% Benchmarks
fwd["pred_IS average"] = fwd["bench_is_avg"]
fwd["pred_yesterday"] = fwd["bench_yday"]
fwd["pred_last bar"] = fwd["lag_quoted_hs"]

# %% Linear models, same specifications as 07 / 08, fitted on in-sample days
SPECS = {
    "original: combined": ("quoted_hs_bps", KEY + CONTROLS, False),
    "refined: combined": ("quoted_hs_bps", KEY + CONTROLS, True),
    "log shared: combined": ("log_quoted_hs", KEY + CONTROLS, False),
    "refined + lagged spread": (
        "quoted_hs_bps",
        KEY + CONTROLS + ["lag_quoted_hs"],
        True,
    ),
}
for grp, stocks in GROUPS.items():
    g_is, idx = is_[is_["stock"].isin(stocks)], fwd["stock"].isin(stocks)
    for name, (y, xv, stock_tod) in SPECS.items():
        res, _ = fit_ols(g_is, y, xv, L, stock_tod=stock_tod)
        pred = predict(res, fwd[idx], xv, stock_tod=stock_tod)
        if y == "log_quoted_hs":  # Duan smearing, as in 08
            pred = np.exp(pred) * np.exp(res.resid).mean()
        fwd.loc[idx, f"pred_{name}"] = pred

# %% LightGBM with the hyper-parameters chosen in 08 (forward-chaining CV on in-sample days)
FEATS = [
    "impact_roll_ewm_bps",
    "impact_prevday_bps",
    "log_vol_har",
    "minute_of_day",
    "min_since_us_open",
    "bucket",
    "stock",
    "lag_quoted_hs",
    "lag_log_trades",
    "lag_log_turnover",
    "lag_ofi",
    "lag_abs_ofi",
]
CATS = ["bucket", "stock"]


def ml_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATS].copy()
    X["stock"] = pd.Categorical(X["stock"], categories=list(config.STOCKS))
    X["bucket"] = pd.Categorical(
        X["bucket"], categories=sorted(panel["bucket"].unique())
    )
    return X


cv = pd.read_csv(config.TABLES / "08_r2_lgbm_cv.csv").sort_values("cv_rmse")
best = cv.iloc[0]
lgbm_params = {
    "objective": "regression",
    "learning_rate": 0.03,
    "n_estimators": round(best["best_iter"]),
    "num_leaves": int(best["num_leaves"]),
    "min_child_samples": int(best["min_child_samples"]),
    "colsample_bytree": float(best["colsample_bytree"]),
    "subsample": 1.0,
    "random_state": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
print(
    "LightGBM parameters frozen from 08:",
    {
        k: lgbm_params[k]
        for k in ["n_estimators", "num_leaves", "min_child_samples", "colsample_bytree"]
    },
    "\n",
)
train = is_.dropna(subset=["quoted_hs_bps"])
lgbm = lgb.LGBMRegressor(**lgbm_params).fit(
    ml_frame(train), train["quoted_hs_bps"], categorical_feature=CATS
)
fwd["pred_LightGBM"] = lgbm.predict(ml_frame(fwd))

MODELS = ["IS average", "yesterday", "last bar", *SPECS, "LightGBM"]
for m in MODELS:
    fwd[f"se_{m}"] = (fwd["quoted_hs_bps"] - fwd[f"pred_{m}"]) ** 2

# %% RMSE by group and stock, all days and ex event days
rows = []
for subset, sub in [("all FWD days", fwd), ("ex event days", fwd[~is_event])]:
    for unit in [*GROUPS, *config.STOCKS]:
        s = sub[sub["group"] == unit] if unit in GROUPS else sub[sub["stock"] == unit]
        rows.append(
            {
                "subset": subset,
                "unit": unit,
                "n": len(s),
                **{m: np.sqrt(s[f"se_{m}"].mean()) for m in MODELS},
            }
        )
rmse = pd.DataFrame(rows).set_index(["subset", "unit"])
rmse.to_csv(config.TABLES / "09_fwd_rmse.csv", float_format="%.4f")
print(
    "Forward-test RMSE of the quoted half-spread (bps):\n",
    rmse.round(3).to_string(),
    "\n",
)

# %% Diebold-Mariano, same pairs as the headline table in 08
L_dm = nw_lags(fwd["ts"].nunique())
pairs = [
    ("original: combined", "last bar"),
    ("refined: combined", "last bar"),
    ("log shared: combined", "last bar"),
    ("refined + lagged spread", "last bar"),
    ("LightGBM", "last bar"),
    ("refined: combined", "yesterday"),
    ("LightGBM", "yesterday"),
    ("LightGBM", "refined + lagged spread"),
]
dm_rows = []
for subset, sub in [("all FWD days", fwd), ("ex event days", fwd[~is_event])]:
    for unit in [*GROUPS, *config.STOCKS]:
        s = sub[sub["group"] == unit] if unit in GROUPS else sub[sub["stock"] == unit]
        loss = s.groupby("ts")[[f"se_{m}" for m in MODELS]].mean()
        for a, b in pairs:
            dm_rows.append(
                {
                    "subset": subset,
                    "unit": unit,
                    "model": a,
                    "vs": b,
                    **dm_test(loss[f"se_{a}"], loss[f"se_{b}"], L_dm),
                }
            )
dmt = pd.DataFrame(dm_rows)
dmt.to_csv(
    config.TABLES / "09_fwd_diebold_mariano.csv", index=False, float_format="%.4f"
)
print(
    f"Diebold-Mariano, all FWD days (NW lags {L_dm}; negative = model beats 'vs'):\n",
    dmt[dmt["subset"] == "all FWD days"]
    .pivot_table(index=["model", "vs"], columns="unit", values="dm_stat", sort=False)[
        [*GROUPS, *config.STOCKS]
    ]
    .round(2)
    .to_string(),
    "\n",
)

# %% Direction of change vs the last bar: does the model call widening / narrowing correctly?
# A market maker acts on direction (widen or tighten the quote); a point forecast that is close
# in RMSE but wrong in direction is of little use. Bars where the spread did not change are dropped.
d_act = np.sign(fwd["quoted_hs_bps"] - fwd["lag_quoted_hs"])
moved = d_act != 0
dir_rows = []
for m in [m for m in MODELS if m != "last bar"]:
    d_pred = np.sign(fwd[f"pred_{m}"] - fwd["lag_quoted_hs"])
    for unit in [*GROUPS, *config.STOCKS]:
        sel = (fwd["group"] == unit) if unit in GROUPS else (fwd["stock"] == unit)
        sel = sel & moved
        hit = (d_pred[sel] == d_act[sel]).mean()
        dir_rows.append(
            {"model": m, "unit": unit, "n_moved": int(sel.sum()), "hit_rate": hit}
        )
direction = pd.DataFrame(dir_rows).pivot(
    index="model", columns="unit", values="hit_rate"
)[[*GROUPS, *config.STOCKS]]
direction.to_csv(config.TABLES / "09_fwd_direction.csv", float_format="%.4f")
print(
    "Direction-of-change hit rate vs last bar (share of bars with a spread change; 0.5 = coin flip):\n",
    direction.round(3).to_string(),
    "\n",
)

# %% Forward test vs the (three-times-used) out-of-sample window
oos = (
    pd.read_csv(config.TABLES / "08_oos_rmse.csv")
    .query("subset == 'all OOS days'")
    .set_index("unit")
)
cmp = pd.DataFrame(
    {
        "OOS 22-30 Oct": oos.loc[list(GROUPS), MODELS].stack(),
        "FWD 2-6 Nov": rmse.loc["all FWD days"].loc[list(GROUPS), MODELS].stack(),
    }
)
cmp.index.names = ["group", "model"]
cmp["FWD / OOS"] = cmp["FWD 2-6 Nov"] / cmp["OOS 22-30 Oct"]
cmp.to_csv(config.TABLES / "09_fwd_vs_oos.csv", float_format="%.4f")
print(
    "RMSE, out-of-sample window vs forward test (bps):\n",
    cmp.round(3).to_string(),
    "\n",
)

# %% Per stock-day RMSE (event days stand out)
by_day = fwd.groupby(["stock", "date"]).apply(
    lambda s: pd.Series(
        {
            m: np.sqrt(s[f"se_{m}"].mean())
            for m in ["last bar", "refined: combined", "LightGBM"]
        }
        | {"mean_hs_bps": s["quoted_hs_bps"].mean(), "bars": len(s)}
    ),
    include_groups=False,
)
by_day["event"] = [
    config.EVENT_DAYS.get((s, str(d.date())), "") for s, d in by_day.index
]
by_day.round(3).to_csv(config.TABLES / "09_fwd_by_day.csv")
print("Per stock-day RMSE (bps):\n", by_day.round(3).to_string(), "\n")

# %% Figure: OOS vs FWD RMSE by group
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
y = np.arange(len(MODELS))[::-1]
for ax, grp in zip(axes, GROUPS):
    ax.barh(
        y + 0.19,
        oos.loc[grp, MODELS].to_numpy(dtype=float),
        height=0.36,
        color="#c3c2b7",
        label="OOS 22-30 Oct (used 3x)",
    )
    ax.barh(
        y - 0.19,
        rmse.loc[("all FWD days", grp), MODELS].to_numpy(dtype=float),
        height=0.36,
        color=GROUP_COLORS[grp],
        label="Forward 2-6 Nov (used once)",
    )
    ax.set_yticks(y, MODELS)
    ax.set_xlabel("RMSE (bps)")
    ax.set_title(f"{grp.capitalize()} group", loc="left", fontsize=10)
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle(
    "Quoted half-spread forecast error: out-of-sample window vs forward test",
    x=0.01,
    ha="left",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(config.FIGURES / "09_fwd_rmse.png", dpi=150)
plt.close(fig)
print(
    "Saved 09_fwd_rmse.png. Reminder: the forward-test week is now used; do not tune on it."
)
