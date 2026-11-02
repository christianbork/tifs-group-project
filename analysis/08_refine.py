# %% [markdown]
# # 08 Refinement: stock x time controls, LightGBM + SHAP, HMM regimes
# Script 07 stays as the original model. Here:
# 1. Refinement 1: replace the shared time-of-day dummies + stock FE with stock x
#    coarse time-bucket dummies (keyed to the actual US open). Motivated by the
#    in-sample misfit of 07 (Novo in-sample RMSE 0.49 bps vs 0.37 around its own mean).
#    Alternative: log spread with the shared dummies. NOTE: the out-of-sample data
#    is evaluated a SECOND time for this refinement (07 was the first look).
# 2. Refinement 2: LightGBM on the same no-look-ahead features + lagged spread,
#    tuned by forward-chaining whole-day CV on in-sample days only; SHAP.
# 3. Robustness: 2-state HMM (calm vs stressed) with forward-filtered regimes;
#    do the drivers change by regime?

# %% Setup
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from hmmlearn.hmm import GaussianHMM

from tifs_group_project import config
from tifs_group_project.regimes import forward_filter
from tifs_group_project.spread_model import (
    build_panel,
    coarse_bucket,
    dm_test,
    fit_ols,
    key_table,
    nw_lags,
    predict,
)

SEED = 42
warnings.filterwarnings(
    "ignore", message=".*eval_set.*"
)  # lightgbm 4.7 deprecation notice
IMPACT = ["impact_roll_ewm_bps", "impact_prevday_bps"]
VOL = ["log_vol_har"]
CONTROLS = ["lag_log_trades", "lag_log_turnover", "lag_abs_ofi"]
KEY = IMPACT + VOL
GROUPS = {"constrained": config.CONSTRAINED, "unconstrained": config.UNCONSTRAINED}
RACE = {
    "controls only": CONTROLS,
    "vol only": VOL + CONTROLS,
    "impact only": IMPACT + CONTROLS,
    "combined": KEY + CONTROLS,
}
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
    }
)

panel = build_panel("5min")
g = panel.groupby("stock")
panel["lag_quoted_hs"] = g["quoted_hs_bps"].shift(1)
panel["lag_ofi"] = g["ofi"].shift(1).fillna(0.0)
panel["bucket"] = coarse_bucket(panel["minute_of_day"], panel["min_since_us_open"])
is_ = panel[panel["sample"] == "IS"]
L = nw_lags(int(is_.groupby("stock").size().median()))
# same OOS evaluation sample as script 07
oos = (
    panel[panel["sample"] == "OOS"]
    .dropna(subset=["quoted_hs_bps", "bench_yday", *KEY, *CONTROLS])
    .copy()
)
oos["pred_IS average"], oos["pred_yesterday"] = oos["bench_is_avg"], oos["bench_yday"]
oos["pred_last bar"] = oos["lag_quoted_hs"]  # naive benchmark: the spread in bar t-1
is_event = pd.Series(
    [
        (s, str(d.date())) in config.EVENT_DAYS
        for s, d in zip(oos["stock"], oos["date"])
    ],
    index=oos.index,
)
print(
    f"Newey-West lags L = {L}; OOS evaluation sample: {len(oos):,} stock-bars (same as script 07)\n"
)

# %% Refinement 1: fit original / refined / log-shared on in-sample days
SPECS = {
    "original": ("quoted_hs_bps", False),
    "refined": ("quoted_hs_bps", True),
    "log shared": ("log_quoted_hs", False),
}
fit_rows, coef_rows = [], []
for grp, stocks in GROUPS.items():
    g_is, idx = is_[is_["stock"].isin(stocks)], oos["stock"].isin(stocks)
    for spec, (y, stock_tod) in SPECS.items():
        for name, xv in RACE.items():
            res, d = fit_ols(g_is, y, xv, L, stock_tod=stock_tod)
            fitted, pred = (
                res.fittedvalues,
                predict(res, oos[idx], xv, stock_tod=stock_tod),
            )
            if (
                y == "log_quoted_hs"
            ):  # back to bps with Duan's smearing factor (in-sample residuals)
                smear = np.exp(res.resid).mean()
                fitted, pred = np.exp(fitted) * smear, np.exp(pred) * smear
            oos.loc[idx, f"pred_{spec}: {name}"] = pred
            if name == "combined":
                e = d["quoted_hs_bps"] - fitted
                for stock in stocks:
                    s = d["stock"] == stock
                    own = d.loc[s, "quoted_hs_bps"] - d.loc[s, "quoted_hs_bps"].mean()
                    fit_rows.append(
                        {
                            "group": grp,
                            "stock": stock,
                            "spec": spec,
                            "is_rmse_own_mean": np.sqrt((own**2).mean()),
                            "is_rmse_model": np.sqrt((e[s] ** 2).mean()),
                        }
                    )
                coef_rows.append(key_table(res, d, spec, KEY).assign(group=grp))

is_fit = pd.DataFrame(fit_rows).pivot_table(
    index=["group", "stock"], columns="spec", values="is_rmse_model", sort=False
)[list(SPECS)]
is_fit.insert(
    0,
    "own mean",
    pd.DataFrame(fit_rows)
    .groupby(["group", "stock"], sort=False)["is_rmse_own_mean"]
    .first(),
)
is_fit.round(4).to_csv(config.TABLES / "08_r1_insample_fit.csv")
print(
    "Refinement 1 - in-sample RMSE of the combined model (bps); 'own mean' = RMSE around the stock's mean:\n",
    is_fit.round(3).to_string(),
    "\n",
)
coefs = pd.concat(coef_rows, ignore_index=True)
coefs.to_csv(config.TABLES / "08_r1_coefficients.csv", index=False, float_format="%.4f")
print(
    "Driver coefficients (combined model) by spec; log-spread coefs are in log points:\n",
    coefs.pivot_table(
        index=["group", "spec"], columns="var", values=["effect_1sd", "t"], sort=False
    )
    .round(3)
    .to_string(),
    "\n",
)

# %% Refinement 2: LightGBM, forward-chaining whole-day CV on in-sample days
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


train = is_.dropna(subset=["quoted_hs_bps"])
X_tr, y_tr = ml_frame(train), train["quoted_hs_bps"]
days = sorted(train["date"].unique())
val_days = days[
    -5:
]  # each fold: train on all earlier in-sample days, validate on this one day
base = dict(
    objective="regression",
    learning_rate=0.03,
    n_estimators=3000,
    subsample=1.0,
    random_state=SEED,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)
grid = [
    dict(num_leaves=nl, min_child_samples=mc, colsample_bytree=ff)
    for nl in [7, 15, 31]
    for mc in [20, 100]
    for ff in [0.7, 1.0]
]
cv_rows = []
for params in grid:
    for vd in val_days:
        tr, va = train["date"] < vd, train["date"] == vd
        m = lgb.LGBMRegressor(**base, **params).fit(
            X_tr[tr],
            y_tr[tr],
            eval_set=[(X_tr[va], y_tr[va])],
            categorical_feature=CATS,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        cv_rows.append(
            {
                **params,
                "val_day": vd,
                "rmse": m.best_score_["valid_0"]["l2"] ** 0.5,
                "best_iter": m.best_iteration_,
            }
        )
cv = pd.DataFrame(cv_rows)
cv_sum = (
    cv.groupby(["num_leaves", "min_child_samples", "colsample_bytree"])
    .agg(cv_rmse=("rmse", "mean"), best_iter=("best_iter", "mean"))
    .sort_values("cv_rmse")
)
cv_sum.round(4).to_csv(config.TABLES / "08_r2_lgbm_cv.csv")
best = cv_sum.index[0]
best_params = dict(
    num_leaves=int(best[0]),
    min_child_samples=int(best[1]),
    colsample_bytree=float(best[2]),
)
n_iter = int(round(cv_sum.iloc[0]["best_iter"]))
print(
    "LightGBM forward-chaining CV (validation = last 5 in-sample days), top 5:\n",
    cv_sum.head().round(4).to_string(),
    f"\nChosen: {best_params}, n_estimators = {n_iter}\n",
)
lgbm = lgb.LGBMRegressor(**{**base, "n_estimators": n_iter}, **best_params).fit(
    X_tr, y_tr, categorical_feature=CATS
)
oos["pred_LightGBM"] = lgbm.predict(ml_frame(oos))  # the one OOS look for the ML model

# linear counterpart with the same lagged-spread information, so the ML gain is not just the lag
for grp, stocks in GROUPS.items():
    idx = oos["stock"].isin(stocks)
    xv = KEY + CONTROLS + ["lag_quoted_hs"]
    res, _ = fit_ols(
        is_[is_["stock"].isin(stocks)], "quoted_hs_bps", xv, L, stock_tod=True
    )
    oos.loc[idx, "pred_refined + lagged spread"] = predict(
        res, oos[idx], xv, stock_tod=True
    )

# %% OOS race: original vs refined vs alternatives
print("=" * 100)
print("NOTE on out-of-sample (22-30 Oct) use:")
print("  1st evaluation: script 07 (original model).")
print(
    "  2nd evaluation: refinement 1 here (stock x time buckets; log spread with shared dummies)."
)
print(
    "  3rd evaluation: refinement 2 here (LightGBM, and the linear refined + lagged spread model)."
)
print("=" * 100, "\n")
MODELS = [
    "IS average",
    "yesterday",
    "last bar",
    *[f"{s}: {m}" for s in SPECS for m in RACE],
    "refined + lagged spread",
    "LightGBM",
]
for m in MODELS:
    oos[f"se_{m}"] = (oos["quoted_hs_bps"] - oos[f"pred_{m}"]) ** 2

rmse = []
for subset, sub in [("all OOS days", oos), ("ex event days", oos[~is_event])]:
    for unit in [*GROUPS, *config.STOCKS]:
        s = sub[sub["group"] == unit] if unit in GROUPS else sub[sub["stock"] == unit]
        rmse.append(
            {
                "subset": subset,
                "unit": unit,
                "n": len(s),
                **{m: np.sqrt(s[f"se_{m}"].mean()) for m in MODELS},
            }
        )
rmse = pd.DataFrame(rmse).set_index(["subset", "unit"])
rmse.to_csv(config.TABLES / "08_oos_rmse.csv", float_format="%.4f")
show = [
    "IS average",
    "yesterday",
    "last bar",
    "original: combined",
    "refined: combined",
    "log shared: combined",
    "refined: vol only",
    "refined: impact only",
    "refined + lagged spread",
    "LightGBM",
]
print(
    "OOS RMSE (bps), selected models (full table in 08_oos_rmse.csv):\n",
    rmse[show].round(3).to_string(),
    "\n",
)

L_dm = nw_lags(oos["ts"].nunique())
pairs = [
    ("refined: combined", "original: combined"),
    ("log shared: combined", "original: combined"),
    ("refined: combined", "IS average"),
    ("refined: combined", "yesterday"),
    ("refined: combined", "refined: controls only"),
    ("refined: combined", "refined: vol only"),
    ("refined: combined", "refined: impact only"),
    ("LightGBM", "refined: combined"),
    ("LightGBM", "refined + lagged spread"),
    ("LightGBM", "yesterday"),
    ("original: combined", "last bar"),
    ("refined: combined", "last bar"),
    ("log shared: combined", "last bar"),
    ("refined + lagged spread", "last bar"),
    ("LightGBM", "last bar"),
]
dm_rows = []
for subset, sub in [("all OOS days", oos), ("ex event days", oos[~is_event])]:
    for unit in [*GROUPS, *config.STOCKS]:
        s = sub[sub["group"] == unit] if unit in GROUPS else sub[sub["stock"] == unit]
        loss = s.groupby("ts")[
            [f"se_{m}" for m in MODELS]
        ].mean()  # group: average across stocks per bar
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
    config.TABLES / "08_oos_diebold_mariano.csv", index=False, float_format="%.4f"
)
print(
    f"Diebold-Mariano, all OOS days (NW lags {L_dm}; negative = model beats 'vs'):\n",
    dmt[dmt["subset"] == "all OOS days"]
    .pivot_table(index=["model", "vs"], columns="unit", values="dm_stat", sort=False)[
        [*GROUPS, *config.STOCKS]
    ]
    .round(2)
    .to_string(),
    "\n",
)

# %% SHAP (explains the model on in-sample training rows; OOS is used only in the race)
explainer = shap.TreeExplainer(lgbm)
sv = explainer.shap_values(X_tr)
imp_by_group = (
    pd.DataFrame(np.abs(sv), columns=FEATS, index=train.index)
    .groupby(train["group"])
    .mean()
    .T
)
imp_by_group = imp_by_group.sort_values("unconstrained", ascending=False)
imp_by_group.round(4).to_csv(config.TABLES / "08_r2_shap_importance.csv")
print(
    "Mean |SHAP| by group (bps of predicted half-spread):\n",
    imp_by_group.round(3).to_string(),
    "\n",
)


def shap_slope(feat: str, by: pd.Series) -> pd.Series:
    """Slope of a feature's SHAP value on the feature, within each level of `by` (bps per unit)."""
    j = FEATS.index(feat)
    df = pd.DataFrame(
        {"x": X_tr[feat].to_numpy(), "s": sv[:, j], "by": by.to_numpy()}
    ).dropna()
    return df.groupby("by").apply(
        lambda d: np.polyfit(d["x"], d["s"], 1)[0], include_groups=False
    )


# the plan's interactions: vol x open (slope of the vol effect by time bucket),
# impact x thin stock (slope of the rolling-impact effect by stock)
inter = pd.concat(
    {
        "vol slope by bucket": shap_slope("log_vol_har", train["bucket"]),
        "impact slope by stock": shap_slope("impact_roll_ewm_bps", train["stock"]),
    }
)
inter.round(4).to_frame("shap_slope").to_csv(
    config.TABLES / "08_r2_shap_interactions.csv"
)
print(
    "SHAP interaction check, slope of SHAP on the feature within each group "
    "(vol: bps per log point; impact: bps per bps):\n",
    inter.round(3).to_string(),
    "\n",
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
y = np.arange(len(imp_by_group))[::-1]
for k, (grp, color) in enumerate(zip(GROUPS, ["#2a78d6", "#eb6834"])):
    axes[0].barh(
        y + (0.5 - k) * 0.38, imp_by_group[grp], height=0.36, color=color, label=grp
    )
axes[0].set_yticks(y, imp_by_group.index, fontsize=8)
axes[0].set_xlabel("Mean |SHAP| (bps)")
axes[0].set_title("Feature reliance by group", loc="left", fontsize=10)
axes[0].legend(frameon=False, fontsize=8)
for ax, feat, title in [
    (axes[1], "log_vol_har", "Vol forecast (log ratio)"),
    (axes[2], "impact_roll_ewm_bps", "Impact, rolling (bps)"),
]:
    j = FEATS.index(feat)
    for grp, color in zip(GROUPS, ["#2a78d6", "#eb6834"]):
        msk = (train["group"] == grp).to_numpy()
        ax.scatter(
            X_tr[feat].to_numpy()[msk],
            sv[msk, j],
            s=4,
            alpha=0.35,
            color=color,
            label=grp,
            linewidths=0,
        )
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xlabel(title)
    ax.set_ylabel("SHAP (bps)")
    ax.set_title(f"SHAP dependence: {title}", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(config.FIGURES / "08_shap.png", dpi=150)
plt.close(fig)

# %% Robustness: 2-state HMM regimes (fit on in-sample days; forward-filtered, inputs lagged one bar)
# Inputs are deseasonalised (minus the in-sample stock x time-bucket mean) and scaled by the
# stock's in-sample SD, so states pick up unusual activity, not the routine intraday pattern.
# Main = volatility-only regime (calm vs stressed). The 3-input version (vol, turnover, spread)
# is reported as an alternative: in-sample it splits "quiet and wide" from "active and tight"
# bars rather than stress, and using the lagged spread to define regimes for a spread
# regression is partly circular.
HMM_SPECS = {
    "vol only (main)": ["hmm_vol"],
    "vol, turnover, spread": ["hmm_vol", "hmm_turnover", "hmm_spread"],
}
raw = pd.DataFrame(
    {
        "hmm_vol": np.log1p(panel["rv_bps"]),
        "hmm_turnover": np.log1p(panel["dkk_turnover"]),
        "hmm_spread": np.log(panel["quoted_hs_bps"]),
    }
)
in_s = panel["sample"] == "IS"
cell_mean = raw[in_s].groupby([panel["stock"], panel["bucket"]]).mean()
dev = (
    raw
    - cell_mean.reindex(
        pd.MultiIndex.from_arrays([panel["stock"], panel["bucket"]])
    ).to_numpy()
)
dev = dev / dev[in_s].groupby(panel["stock"]).std().reindex(panel["stock"]).to_numpy()
obs = dev.groupby(panel["stock"]).shift(1)  # bar t sees bar t-1

state_rows, share_rows, reg_rows = [], [], []
for spec, cols in HMM_SPECS.items():
    seqs = {s: obs.loc[(panel["stock"] == s), cols].dropna() for s in config.STOCKS}
    X_is = [x[panel.loc[x.index, "sample"] == "IS"].to_numpy() for x in seqs.values()]
    hmm = GaussianHMM(
        n_components=2, covariance_type="full", n_iter=300, random_state=SEED
    ).fit(np.vstack(X_is), lengths=[len(x) for x in X_is])
    hi = int(np.argmax(hmm.means_[:, 0]))  # stressed = the state with higher volatility
    for k in range(2):
        state_rows.append(
            {
                "hmm": spec,
                "state": "stressed" if k == hi else "calm",
                **dict(zip(cols, hmm.means_[k])),
                "stay_prob": hmm.transmat_[k, k],
            }
        )
    p = pd.Series(np.nan, index=panel.index)
    for s, x in seqs.items():
        p.loc[x.index] = forward_filter(hmm, x.to_numpy())[:, hi]
    stressed = (p > 0.5).astype(float).where(p.notna())
    if spec == "vol only (main)":
        panel["p_stressed"], panel["stressed"] = p, stressed

    share = (
        stressed.groupby([panel["stock"], panel["sample"]]).mean().unstack().mul(100)
    )
    for stock in config.STOCKS:
        share_rows.append(
            {
                "hmm": spec,
                "stock": stock,
                "pct_stressed_IS": share.loc[stock, "IS"],
                "pct_stressed_OOS": share.loc[stock, "OOS"],
            }
        )
    for (s, t), lab in config.EVENT_DAYS.items():
        normal = (panel["stock"] == s) & (panel["date"] != t) & in_s
        share_rows.append(
            {
                "hmm": spec,
                "stock": f"{s} {t} ({lab})",
                "pct_stressed_event_day": 100
                * stressed[(panel["stock"] == s) & (panel["date"] == t)].mean(),
                "pct_stressed_IS_other_days": 100 * stressed[normal].mean(),
            }
        )

    # drivers by regime: refined spec, drivers interacted with the stressed dummy (in-sample)
    is_r = panel[in_s].assign(stressed=stressed[in_s])
    for k in KEY:
        is_r[f"{k}_x_stress"] = is_r[k] * is_r["stressed"]
    for grp, stocks in GROUPS.items():
        xv = KEY + [f"{k}_x_stress" for k in KEY] + ["stressed"] + CONTROLS
        res, d = fit_ols(
            is_r[is_r["stock"].isin(stocks)], "quoted_hs_bps", xv, L, stock_tod=True
        )
        for k in KEY:
            calm, strs, diff = (
                res.t_test(f"{k} = 0"),
                res.t_test(f"{k} + {k}_x_stress = 0"),
                res.t_test(f"{k}_x_stress = 0"),
            )
            reg_rows.append(
                {
                    "hmm": spec,
                    "group": grp,
                    "var": k,
                    "calm_coef": float(calm.effect[0]),
                    "calm_t": float(calm.tvalue[0][0]),
                    "stressed_coef": float(strs.effect[0]),
                    "stressed_t": float(strs.tvalue[0][0]),
                    "diff_t": float(diff.tvalue[0][0]),
                    "pct_bars_stressed": 100 * d["stressed"].mean(),
                }
            )

states = pd.DataFrame(state_rows)
shares = pd.DataFrame(share_rows)
by_regime = pd.DataFrame(reg_rows)
states.to_csv(config.TABLES / "08_hmm_states.csv", index=False, float_format="%.4f")
shares.to_csv(
    config.TABLES / "08_hmm_stressed_share.csv", index=False, float_format="%.2f"
)
by_regime.to_csv(
    config.TABLES / "08_hmm_drivers_by_regime.csv", index=False, float_format="%.4f"
)
print(
    "HMM states (inputs standardised, deseasonalised):\n",
    states.round(2).to_string(index=False),
    "\n",
)
print(
    "% of bars in the stressed regime:\n", shares.round(1).to_string(index=False), "\n"
)
print(
    "Drivers by regime (refined spec, in-sample; diff_t tests stressed = calm):\n",
    by_regime.round(3).to_string(index=False),
    "\n",
)

daily = panel.groupby(["stock", "date"])["stressed"].mean().mul(100).reset_index()
fig, ax = plt.subplots(figsize=(9, 3.8))
for stock in config.STOCKS:
    s = daily[daily["stock"] == stock]
    ax.plot(
        s["date"],
        s["stressed"],
        color=COLORS[stock],
        marker="o",
        markersize=3,
        linewidth=1.4,
        label=stock,
    )
for (s, t), lab in config.EVENT_DAYS.items():
    ax.axvline(pd.Timestamp(t), color="#52514e", linewidth=0.8, linestyle="--")
    ax.text(pd.Timestamp(t), 98, f" {s} {lab}", fontsize=8, color="#52514e", va="top")
ax.axvspan(
    pd.Timestamp(config.IS_END) + pd.Timedelta(hours=12),
    daily["date"].max() + pd.Timedelta(hours=12),
    color="#8a8984",
    alpha=0.12,
    linewidth=0,
)
ax.set_ylim(0, 100)
ax.set_ylabel("% of bars stressed")
ax.set_title(
    "HMM (volatility) stressed share per stock-day, forward-filtered; shaded = OOS",
    loc="left",
    fontsize=11,
)
ax.legend(
    ncol=6, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32)
)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(config.FIGURES / "08_hmm_regimes.png", dpi=150)
plt.close(fig)
