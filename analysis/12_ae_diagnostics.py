# %% [markdown]
# # 12 Applied Econometrics diagnostics (report appendix)
# The checks from the Applied Econometrics blueprint (AE Blueprint Final.R) that the
# main pipeline did not run in v1, implemented in Python:
# 1. ADF unit-root tests (urca::ur.df, drift, IC lag choice) on the spread and the drivers:
#    are levels regressions valid?
# 2. Granger-style causality (linearHypothesis on lag blocks): do the drivers add to an
#    AR(3) of the spread (with time-of-day, stock FE and activity controls)?
# 3. QLR structural-break test (sup-F over trimmed break dates) on each stock's daily spread:
#    is the out-of-sample drift a break? Ex-post diagnostic on the 22 October days (not used
#    to choose any model; the 2-6 Nov forward week is left out).
# 4. One appendix table that also collects the AE checks run elsewhere: Newey-West lag rule
#    (07), VAR lag choice and bootstrap CIs (05), ARCH-LM / Ljung-Box (06).
# Reads the panel built by scripts 05-06 (via build_panel) and tables from 05-07.

# %% Setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from tifs_group_project import config
from tifs_group_project.spread_model import build_panel, fit_ols, lag_within_day, nw_lags, wald

IMPACT = ["impact_roll_ewm_bps", "impact_prevday_bps"]
VOL = ["log_vol_har"]
CONTROLS = ["lag_log_trades", "lag_log_turnover", "lag_abs_ofi"]
KEY = IMPACT + VOL
GROUPS = {"constrained": config.CONSTRAINED, "unconstrained": config.UNCONSTRAINED}
# Stock & Watson QLR critical values, 15% trimming, q = 1 restriction
QLR_CV = {"10%": 7.12, "5%": 8.68, "1%": 12.16}
plt.rcParams.update(
    {"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e4e0", "grid.linewidth": 0.6}
)

panel = build_panel("5min")
is_ = panel[panel["sample"] == "IS"]
L = nw_lags(int(is_.groupby("stock").size().median()))

# %% 1. ADF tests (in-sample, each stock's series in time order; drift; AIC lags up to 12)
adf_rows = []
for s in config.STOCKS:
    d = is_[is_["stock"] == s]
    for col, label in [
        ("quoted_hs_bps", "quoted half-spread"),
        ("log_vol_har", "log vol forecast ratio"),
        ("impact_roll_ewm_bps", "rolling impact"),
    ]:
        x = d[col].dropna().to_numpy()
        stat, p, lags, nobs, cv, _ = adfuller(x, maxlag=12, regression="c", autolag="AIC", result_object=False)
        adf_rows.append(
            {"stock": s, "series": label, "tau": stat, "cv_5pct": cv["5%"], "p_value": p, "lags": lags, "n": nobs,
             "decision_5pct": "reject unit root" if stat < cv["5%"] else "fail to reject"}
        )
adf = pd.DataFrame(adf_rows)
adf.round(4).to_csv(config.TABLES / "12_adf.csv", index=False)
print("ADF tests (H0: unit root):\n", adf.round(3).to_string(index=False), "\n")

# %% 2. Granger-style tests: drivers added to an AR(3) of the spread (lags within the day)
g_rows = []
p3 = panel.copy()
for k in [1, 2, 3]:
    p3[f"hs_lag{k}"] = lag_within_day(p3, "quoted_hs_bps", k)
p3 = p3[p3["sample"] == "IS"]
AR = ["hs_lag1", "hs_lag2", "hs_lag3"]
for grp, stocks in GROUPS.items():
    g = p3[p3["stock"].isin(stocks)]
    for se_label, cov, lags in [("NW L=7", "nw", L), ("Driscoll-Kraay L=24", "dk", 24)]:
        res, _ = fit_ols(g, "quoted_hs_bps", AR + KEY + CONTROLS, lags, cov=cov)
        for test, terms in [("impact (both) -> spread", IMPACT), ("rolling impact -> spread", ["impact_roll_ewm_bps"]),
                            ("vol forecast -> spread", VOL), ("all drivers -> spread", KEY)]:
            g_rows.append({"group": grp, "se": se_label, "test": test, "n": int(res.nobs), **wald(res, terms)})
granger = pd.DataFrame(g_rows)
granger.round(4).to_csv(config.TABLES / "12_granger.csv", index=False)
print("Granger-style Wald tests (AR(3) + time-of-day + stock FE + activity controls):\n",
      granger.round(4).to_string(index=False), "\n")

# %% 3. QLR break test on each stock's daily time-weighted half-spread (22 days)
day = (
    panel[panel["sample"] != "FWD"]
    .groupby(["stock", "date"])
    .apply(lambda x: np.average(x["quoted_hs_bps"].fillna(0), weights=x["q_sec"]), include_groups=False)
    .rename("hs")
    .reset_index()
)
q_rows, q_paths = [], []
for s in config.STOCKS:
    y = day[day["stock"] == s].sort_values("date").reset_index(drop=True)
    y["lag"] = y["hs"].shift(1)
    y = y.dropna().reset_index(drop=True)
    n = len(y)
    lo, hi = int(np.floor(0.15 * n)), int(np.ceil(0.85 * n))
    fstats = []
    for tau in range(lo, hi):
        D = (np.arange(n) >= tau).astype(float)  # break: mean shift from day tau on
        X = sm.add_constant(np.column_stack([y["lag"], D]))
        r = sm.OLS(y["hs"], X).fit()
        f = float(r.tvalues.iloc[2] ** 2)  # q = 1: F = t^2
        fstats.append(f)
        q_paths.append({"stock": s, "break_date": y.loc[tau, "date"], "F": f})
    k = int(np.argmax(fstats))
    tau_hat = y.loc[lo + k, "date"]
    pre, post = y.loc[: lo + k - 1, "hs"].mean(), y.loc[lo + k :, "hs"].mean()
    q_rows.append(
        {"stock": s, "QLR": fstats[k], "break_date": tau_hat.date(), "mean_before": pre, "mean_after": post,
         "sig_10pct": fstats[k] > QLR_CV["10%"], "sig_5pct": fstats[k] > QLR_CV["5%"], "sig_1pct": fstats[k] > QLR_CV["1%"],
         "break_in_oos": tau_hat > pd.Timestamp(config.IS_END)}
    )
qlr = pd.DataFrame(q_rows)
qlr.round(3).to_csv(config.TABLES / "12_qlr.csv", index=False)
print("QLR test for a mean shift in the daily half-spread (AR(1) model, 15% trimming; "
      f"critical values {QLR_CV}):\n", qlr.round(2).to_string(index=False), "\n")

paths = pd.DataFrame(q_paths)
fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True)
for ax, s in zip(axes.flat, config.STOCKS):
    pth = paths[paths["stock"] == s]
    ax.plot(pth["break_date"], pth["F"], color="#2a78d6")
    for lab, cv in QLR_CV.items():
        if lab != "10%":
            ax.axhline(cv, color="#8a8984", linestyle="--", linewidth=0.8)
            ax.text(pth["break_date"].iloc[0], cv, f" {lab}", fontsize=7, color="#52514e", va="bottom")
    ax.axvline(pd.Timestamp(config.IS_END) + pd.Timedelta(hours=12), color="#52514e", linewidth=0.8)
    ax.set_title(s, loc="left", fontsize=10)
fig.supylabel("F-statistic (candidate break)")
fig.suptitle("QLR test for a break in the daily half-spread (vertical line = start of OOS)", x=0.01, ha="left", fontsize=11)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(config.FIGURES / "12_qlr.png", dpi=150)
plt.close(fig)

# %% 4. Appendix summary: every AE check in one table
T = config.TABLES
lag_choice = pd.read_csv(T / "05_lag_choice.csv", index_col="stock")
boot = pd.read_csv(T / "05_impact_bootstrap.csv", index_col="stock")
diag6 = pd.read_csv(T / "06_diagnostics.csv")
se7 = pd.read_csv(T / "07_se_robustness.csv")
before = diag6[diag6["model"] == "returns (before)"]
after = diag6[diag6["model"] == "garch (normal)"]
min_t = se7.assign(at=se7["t"].abs()).groupby(["group", "var"])["at"].min()
rows = [
    ("Newey-West HAC lag rule (Lab 7)", "07", f"L = {L} = floor(0.75 T^(1/3)) - 1 with T = {int(is_.groupby('stock').size().median())} bars per stock",
     "Residual ACF at lag 1 is 0.3-0.7, so L = 7 is short; 07 reports NW 24/95, Driscoll-Kraay and clusters"),
    ("Driver significance across SE choices", "07",
     "; ".join(f"{g[:5]} {v.replace('_bps', '').replace('impact_', 'imp_')}: min|t| {t:.2f}" for (g, v), t in min_t.items()),
     "Volatility robust in both groups; rolling impact robust only in the dynamic spec (07_dynamic.csv)"),
    ("ADF unit root (ur.df, drift, AIC)", "12",
     f"{(adf['decision_5pct'] == 'reject unit root').sum()} of {len(adf)} series reject a unit root at 5%",
     "Levels regressions are valid"),
    ("Granger causality (linearHypothesis)", "12",
     "; ".join(f"{r.group[:5]} {r.test.split(' ->')[0]} p={r.p_value:.3f}" for r in granger[granger["se"] == "Driscoll-Kraay L=24"].itertuples()
               if r.test in ("impact (both) -> spread", "vol forecast -> spread")),
     "Drivers add information beyond the spread's own lags in the unconstrained group"),
    ("QLR structural break", "12",
     "; ".join(f"{r.stock} {r.QLR:.1f}{'*' if r.sig_5pct else ''} ({r.break_date})" for r in qlr.itertuples()),
     "* = significant at 5%; the OOS drift is tested ex post, not used for model choice"),
    ("VAR lag selection (VARselect / BIC)", "05",
     f"BIC minimum at p = {int(lag_choice['bic_min_at'].max())} (largest tried) for every stock",
     "BIC over-selects at n ~ 1e4-1e5; p = 20 fixed where impact is stable (< 5% change to p = 40)"),
    ("IRF bootstrap CIs (irf boot = TRUE)", "05",
     "; ".join(f"{s} {r.impact:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]" for s, r in boot.iterrows()),
     "Day-block bootstrap, 500 resamples; all impacts clearly above zero"),
    ("ARCH effects before GARCH (Lab 8)", "06",
     f"ARCH-LM p < 0.05 for {(before['arch_lm_std_p'] < 0.05).sum()} of {len(before)} stocks",
     "Volatility clustering present, so GARCH-type models are appropriate"),
    ("Standardised GARCH residuals (Lab 8)", "06",
     f"ARCH-LM on standardised residuals p > 0.05 for {(after['arch_lm_std_p'] > 0.05).sum()} of {len(after)} stocks "
     f"(left: {', '.join(after.loc[after['arch_lm_std_p'] <= 0.05, 'stock']) or 'none'}); Ljung-Box (mean) p < 0.05 for "
     f"{', '.join(after.loc[after['lb_std_p'] < 0.05, 'stock']) or 'none'}",
     "GARCH(1,1) removes most clustering; Student-t fits degenerate (nu at 2.05) for "
     + (", ".join(diag6.loc[diag6["degenerate_t"].astype(bool), "stock"].unique()) or "none")
     + " because of the mass of zero 1-min mid returns"),
]
appendix = pd.DataFrame(rows, columns=["AE check", "script", "result", "implication"])
appendix.to_csv(T / "12_ae_appendix.csv", index=False)
with pd.option_context("display.max_colwidth", 200):
    print("Applied Econometrics appendix:\n", appendix.to_string(index=False))
