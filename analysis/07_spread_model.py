# %% [markdown]
# # 07 Spread model: informed trading vs volatility
# Half-spread(bps, bar t) = a + b1 impact_roll(before t) + b2 impact(day t-1)
#   + c log(HAR vol forecast / seasonal level) + time-of-day (clock + US-open)
#   + lagged activity controls + stock fixed effects
# 1. Estimated on in-sample days, separately for the constrained (Novo, DSV, ISS)
#    and unconstrained (Vestas, Pandora, GN) groups; panel Newey-West SEs.
# 2. Dependent variables: quoted half-spread (main), effective half-spread, and
#    for the constrained group the share of time the spread is > 1 tick (fractional logit).
# 3. OOS race (22-30 Oct, touched once): in-sample average and yesterday's spread
#    benchmarks vs controls-only / vol-only / impact-only / combined; RMSE + Diebold-Mariano.
# 4. Robustness: log spread, 15-min bars, drop 09:00-09:15, event flag, GJR vol, NW lags.
# 5. v2 (after the review): standard-error robustness (NW 7/24/95 lags, Driscoll-Kraay,
#    stock-day and day clusters) with residual autocorrelation; a dynamic ADL(1,0) spec
#    with the last observed spread and long-run effects; joint Wald tests (AE
#    linearHypothesis). Activity controls are now lagged within the day (build_panel).

# %% Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tifs_group_project import config
from tifs_group_project.spread_model import (
    SE_TYPES,
    build_panel,
    dm_test,
    fit_frac_logit,
    fit_ols,
    long_run,
    nw_lags,
    predict,
    residual_acf,
    wald,
)

IMPACT = ["impact_roll_ewm_bps", "impact_prevday_bps"]
VOL = ["log_vol_har"]
CONTROLS = ["lag_log_trades", "lag_log_turnover", "lag_abs_ofi"]
KEY = IMPACT + VOL
GROUPS = {"constrained": config.CONSTRAINED, "unconstrained": config.UNCONSTRAINED}
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
is_ = panel[panel["sample"] == "IS"]
T = int(is_.groupby("stock").size().median())
L = nw_lags(T)
print(
    f"Newey-West lags: rule floor(4 (T/100)^(2/9)) with T = {T} bars per stock -> L = {L}; robustness L = 24 (2 hours)\n"
    f"(same as the AE blueprint's floor(0.75 T^(1/3)) - 1 = {int(np.floor(0.75 * T ** (1 / 3))) - 1})\n"
)

# %% Sample and no-look-ahead checks
need = ["quoted_hs_bps", *KEY, *CONTROLS]
cover = (
    panel.groupby(["stock", "sample"])
    .apply(
        lambda x: pd.Series(
            {"bars": len(x), "usable": int(x[need].notna().all(axis=1).sum())}
        ),
        include_groups=False,
    )
    .unstack("sample")
)
print(
    "Bars and usable bars (all regressors present):\n",
    cover.reindex(list(config.STOCKS)).to_string(),
    "\n",
)
assert (
    panel.loc[panel["date"] == panel["date"].min(), "impact_prevday_bps"].isna().all()
)  # no day before day 1
assert panel.loc[panel["date"] == panel["date"].min(), "bench_yday"].isna().all()

# correlation of the drivers (in-sample, within stock)
dm = is_[KEY].sub(is_.groupby("stock")[KEY].transform("mean"))
print(
    "Within-stock correlation of the drivers (in-sample):\n",
    dm.corr().round(2).to_string(),
    "\n",
)


def key_table(res, d, label: str, xvars=KEY, logit: bool = False) -> pd.DataFrame:
    """Key coefficients, NW t-stats, and the effect of a one within-stock-SD move (bps, or prob. for logit)."""
    sd = d[xvars].sub(d.groupby("stock")[xvars].transform("mean")).std()
    scale = (
        (res.fittedvalues * (1 - res.fittedvalues)).mean() if logit else 1.0
    )  # average marginal effect
    return pd.DataFrame(
        {
            "spec": label,
            "var": xvars,
            "coef": res.params[xvars].values,
            "t": res.tvalues[xvars].values,
            "p": res.pvalues[xvars].values,
            "effect_1sd": (res.params[xvars] * sd * scale).values,
            "n": int(res.nobs),
            "r2": getattr(res, "rsquared", np.nan),
        }
    )


# %% Main regressions (in-sample) by group
main, models = [], {}
for grp, stocks in GROUPS.items():
    g = is_[is_["stock"].isin(stocks)]
    res, d = fit_ols(g, "quoted_hs_bps", KEY + CONTROLS, L)
    models[grp] = res
    main.append(key_table(res, d, "quoted (main)").assign(group=grp))
    res, d = fit_ols(g, "eff_hs_bps", KEY + CONTROLS, L)
    main.append(key_table(res, d, "effective").assign(group=grp))
    if grp == "constrained":
        res, d = fit_frac_logit(g, "p_gt_1tick", KEY + CONTROLS, L)
        main.append(
            key_table(res, d, "P(spread > 1 tick), logit AME", logit=True).assign(
                group=grp
            )
        )
main = pd.concat(main, ignore_index=True)[
    ["group", "spec", "var", "coef", "t", "p", "effect_1sd", "n", "r2"]
]
main.to_csv(
    config.TABLES / "07_main_coefficients.csv", index=False, float_format="%.4f"
)
print(
    "Main in-sample estimates (effect_1sd = bps per within-stock SD; probability for the logit):\n",
    main.round(3).to_string(index=False),
    "\n",
)
full = pd.concat(
    {g: pd.DataFrame({"coef": r.params, "t": r.tvalues}) for g, r in models.items()},
    axis=1,
)
full.round(4).to_csv(config.TABLES / "07_main_full_quoted.csv")

# %% Robustness (in-sample)
panel15 = build_panel("15min")
is15 = panel15[panel15["sample"] == "IS"]
L15 = nw_lags(int(is15.groupby("stock").size().median()))
rob = []
for grp, stocks in GROUPS.items():
    g, g15 = is_[is_["stock"].isin(stocks)], is15[is15["stock"].isin(stocks)]
    specs = [
        ("main", g, "quoted_hs_bps", KEY, L),
        ("NW L = 24", g, "quoted_hs_bps", KEY, 24),
        ("log spread", g, "log_quoted_hs", KEY, L),
        ("15-min bars", g15, "quoted_hs_bps", KEY, L15),
        ("drop 09:00-09:15", g[g["minute_of_day"] >= 15], "quoted_hs_bps", KEY, L),
        ("GJR vol", g, "quoted_hs_bps", IMPACT + ["log_vol_gjr"], L),
    ]
    if g["event_day"].std() > 0:  # only DSV 21-Oct is an in-sample event
        specs.append(("+ event-day dummy", g, "quoted_hs_bps", KEY + ["event_day"], L))
    for label, data, y, xv, lags in specs:
        res, d = fit_ols(data, y, xv + CONTROLS, lags)
        rob.append(key_table(res, d, label, xvars=xv).assign(group=grp))
rob = pd.concat(rob, ignore_index=True)
rob.to_csv(config.TABLES / "07_robustness.csv", index=False, float_format="%.4f")
rob_view = rob.pivot_table(
    index=["group", "spec"], columns="var", values="t", sort=False
)
print(
    f"Robustness, NW t-stats of the drivers (15-min bars use L = {L15}):\n",
    rob_view.round(2).to_string(),
    "\n",
)

# %% v2: standard-error robustness and residual autocorrelation (in-sample, main spec)
SE_SPECS = [
    ("NW L=7 (main)", "nw", L),
    ("NW L=24", "nw", 24),
    ("NW L=95 (1 day)", "nw", 95),
    ("Driscoll-Kraay L=24", "dk", 24),
    ("cluster stock-day", "cluster_stockday", None),
    ("cluster day", "cluster_day", None),
]
se_rows, acf_rows = [], []
for grp, stocks in GROUPS.items():
    g = is_[is_["stock"].isin(stocks)]
    for label, cov, lags in SE_SPECS:
        res, d = fit_ols(g, "quoted_hs_bps", KEY + CONTROLS, lags or L, cov=cov)
        se_rows += [
            {"group": grp, "se": label, "var": v, "coef": res.params[v], "se_value": res.bse[v], "t": res.tvalues[v]}
            for v in KEY
        ]
        if cov == "nw" and lags == L:
            acf_rows.append(residual_acf(res, d).assign(group=grp, spec="static (main)"))
se_tab = pd.DataFrame(se_rows)
se_tab.to_csv(config.TABLES / "07_se_robustness.csv", index=False, float_format="%.4f")
se_view = se_tab.pivot_table(index=["group", "var"], columns="se", values="t", sort=False)
se_view["min |t|"] = se_view.abs().min(axis=1)
se_view["robust at 5%"] = se_view["min |t|"] > 1.96
print("Standard-error robustness, t-stats of the drivers (coefficients identical across columns):\n",
      se_view.round(2).to_string(), "\n")

# %% v2: dynamic spec ADL(1,0): add the last observed spread; long-run effects b/(1-rho)
dyn_rows, lr_rows = [], []
for grp, stocks in GROUPS.items():
    g = is_[is_["stock"].isin(stocks)]
    for label, cov, lags in [("NW L=7", "nw", L), ("Driscoll-Kraay L=24", "dk", 24)]:
        res, d = fit_ols(g, "quoted_hs_bps", KEY + CONTROLS + ["lag_quoted_hs"], lags, cov=cov)
        dyn_rows.append(key_table(res, d, f"dynamic, {label}", xvars=KEY + ["lag_quoted_hs"]).assign(group=grp))
        lr_rows.append(long_run(res, KEY).assign(group=grp, se=label))
        if cov == "nw":
            acf_rows.append(residual_acf(res, d).assign(group=grp, spec="dynamic"))
dyn = pd.concat(dyn_rows, ignore_index=True)
lr = pd.concat(lr_rows, ignore_index=True)
dyn.to_csv(config.TABLES / "07_dynamic.csv", index=False, float_format="%.4f")
lr.to_csv(config.TABLES / "07_dynamic_long_run.csv", index=False, float_format="%.4f")
print("Dynamic spec (last observed spread added), short-run coefficients:\n",
      dyn.pivot_table(index=["group", "var"], columns="spec", values="t", sort=False).round(2).to_string(), "\n")
print("Long-run effects b/(1-rho), delta-method SEs:\n", lr.round(3).to_string(index=False), "\n")

acf_tab = pd.concat(acf_rows, ignore_index=True)
acf_tab.to_csv(config.TABLES / "07_residual_acf.csv", index=False, float_format="%.3f")
print("Residual autocorrelation by stock (lag 95 = same bar previous day):\n",
      acf_tab.round(2).to_string(index=False), "\n")

# %% v2: joint Wald tests (AE linearHypothesis): do the two impact measures matter jointly?
wald_rows = []
for grp, stocks in GROUPS.items():
    g = is_[is_["stock"].isin(stocks)]
    for spec, xv in [("static", KEY + CONTROLS), ("dynamic", KEY + CONTROLS + ["lag_quoted_hs"])]:
        for label, cov, lags in [("NW L=7", "nw", L), ("Driscoll-Kraay L=24", "dk", 24)]:
            res, _ = fit_ols(g, "quoted_hs_bps", xv, lags, cov=cov)
            for test, terms in [("impact (both) = 0", IMPACT), ("all drivers = 0", KEY)]:
                wald_rows.append({"group": grp, "spec": spec, "se": label, "test": test, **wald(res, terms)})
wald_tab = pd.DataFrame(wald_rows)
wald_tab.to_csv(config.TABLES / "07_wald.csv", index=False, float_format="%.4f")
print("Joint Wald tests (chi2, p):\n", wald_tab.round(4).to_string(index=False), "\n")

# %% OOS race (touch out-of-sample once)
RACE = {
    "controls only": CONTROLS,
    "vol only": VOL + CONTROLS,
    "impact only": IMPACT + CONTROLS,
    "combined": KEY + CONTROLS,
}
oos = (
    panel[(panel["sample"] == "OOS")]
    .dropna(subset=["quoted_hs_bps", "bench_yday", *KEY, *CONTROLS])
    .copy()
)
for grp, stocks in GROUPS.items():
    g_is, idx = is_[is_["stock"].isin(stocks)], oos["stock"].isin(stocks)
    for name, xv in RACE.items():
        res, _ = fit_ols(g_is, "quoted_hs_bps", xv, L)
        oos.loc[idx, f"pred_{name}"] = predict(res, oos[idx], xv)
oos["pred_IS average"], oos["pred_yesterday"] = oos["bench_is_avg"], oos["bench_yday"]
MODELS = ["IS average", "yesterday", *RACE]
for m in MODELS:
    oos[f"se_{m}"] = (oos["quoted_hs_bps"] - oos[f"pred_{m}"]) ** 2
is_event = pd.Series(
    [
        (s, str(d.date())) in config.EVENT_DAYS
        for s, d in zip(oos["stock"], oos["date"])
    ],
    index=oos.index,
)

rmse = []
for subset, sub in [("all OOS days", oos), ("ex event days", oos[~is_event])]:
    for unit in ["constrained", "unconstrained", *config.STOCKS]:
        s = sub[sub["group"] == unit] if unit in GROUPS else sub[sub["stock"] == unit]
        rmse.append(
            {
                "subset": subset,
                "unit": unit,
                "n": len(s),
                **{m: np.sqrt(s[f"se_{m}"].mean()) for m in MODELS},
            }
        )
rmse = pd.DataFrame(rmse)
rmse.to_csv(config.TABLES / "07_oos_rmse.csv", index=False, float_format="%.4f")
print(
    "OOS RMSE of the quoted half-spread (bps):\n",
    rmse.round(3).to_string(index=False),
    "\n",
)

L_dm = nw_lags(oos["ts"].nunique())
dm_rows = []
pairs = [(m, b) for m in RACE for b in ["IS average", "yesterday"]] + [
    ("combined", "controls only"),
    ("combined", "vol only"),
    ("combined", "impact only"),
]
for subset, sub in [("all OOS days", oos), ("ex event days", oos[~is_event])]:
    for grp in GROUPS:
        loss = (
            sub[sub["group"] == grp].groupby("ts")[[f"se_{m}" for m in MODELS]].mean()
        )  # average across stocks
        for a, b in pairs:
            dm_rows.append(
                {
                    "subset": subset,
                    "group": grp,
                    "model": a,
                    "vs": b,
                    **dm_test(loss[f"se_{a}"], loss[f"se_{b}"], L_dm),
                }
            )
dmt = pd.DataFrame(dm_rows)
dmt.to_csv(
    config.TABLES / "07_oos_diebold_mariano.csv", index=False, float_format="%.4f"
)
print(
    f"Diebold-Mariano (squared error, group loss averaged across stocks per bar; NW lags {L_dm}; "
    "negative = model beats 'vs'):\n",
    dmt.round(3).to_string(index=False),
    "\n",
)

# %% Figures
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
labels = {
    "impact_roll_ewm_bps": "Impact, rolling",
    "impact_prevday_bps": "Impact, previous day",
    "log_vol_har": "Vol forecast (HAR)",
}
for ax, grp in zip(axes, GROUPS):
    r = (
        main[(main["group"] == grp) & (main["spec"] == "quoted (main)")]
        .set_index("var")
        .loc[KEY]
    )
    se = r["effect_1sd"] / r["t"]
    t_dk = se_tab[(se_tab["group"] == grp) & (se_tab["se"] == "Driscoll-Kraay L=24")].set_index("var").loc[KEY, "t"]
    se_dk = r["effect_1sd"] / t_dk
    y = np.arange(len(KEY))[::-1]
    ax.errorbar(
        r["effect_1sd"], y + 0.1, xerr=1.96 * se.abs(), fmt="o", color="#2a78d6",
        ecolor="#2a78d6", capsize=3, markersize=6, label="Newey-West L=7",
    )
    ax.errorbar(
        r["effect_1sd"], y - 0.1, xerr=1.96 * se_dk.abs(), fmt="s", color="#52514e",
        ecolor="#52514e", capsize=3, markersize=5, label="Driscoll-Kraay L=24",
    )
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.set_yticks(y, [labels[k] for k in KEY])
    ax.set_title(f"{grp.capitalize()} group", loc="left", fontsize=10)
    ax.set_xlabel("bps per +1 within-stock SD")
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle(
    "Drivers of the quoted half-spread (in-sample, 95% CI under two SE choices)",
    x=0.01,
    ha="left",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(config.FIGURES / "07_driver_effects.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.8))
r = rmse[(rmse["subset"] == "all OOS days") & rmse["unit"].isin(GROUPS)].set_index(
    "unit"
)[MODELS]
x = np.arange(len(MODELS))
for k, (grp, color) in enumerate(zip(GROUPS, ["#2a78d6", "#eb6834"])):
    ax.bar(x + (k - 0.5) * 0.38, r.loc[grp], width=0.36, color=color, label=grp)
ax.set_xticks(x, MODELS)
ax.set_ylabel("RMSE (bps)")
ax.set_title(
    "Out-of-sample RMSE of the quoted half-spread, 22-30 Oct", loc="left", fontsize=11
)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(config.FIGURES / "07_oos_rmse.png", dpi=150)
plt.close(fig)
