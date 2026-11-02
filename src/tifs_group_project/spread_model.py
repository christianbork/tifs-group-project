"""Spread model panel, time-of-day controls, Newey-West fits and Diebold-Mariano tests.

Every regressor for bar t is known before bar t starts: impact (rolling, up to
the bar start; daily, up to day t-1), the volatility forecast (made at the end
of bar t-1), and activity controls lagged one bar. Time-of-day buckets are a
calendar fact, so they are known in advance.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import config

US_WINDOW = (-30, 60)  # minutes around the US open handled by US-relative buckets
CLOCK_BUCKETS = [
    (0, 15, "09:00"),
    (15, 30, "09:15"),
    (30, 60, "09:30"),
    (60, 120, "10:00"),
    (120, 180, "11:00"),
    (180, 240, "12:00"),
    (240, 300, "13:00"),
    (300, 450, "14:00-16:30"),
    (450, 475, "16:30"),
]
US_BUCKETS = [(-30, 0, "US-30"), (0, 15, "US+0"), (15, 30, "US+15"), (30, 60, "US+30")]


def tod_bucket(minute_of_day: pd.Series, min_since_us_open: pd.Series) -> pd.Series:
    """Clock-time bucket, overridden by a US-relative bucket within US_WINDOW of the US open.

    In-sample the US open is always 15:30, so clock and US time are collinear there;
    this labelling lets the US buckets move to 14:30 on 26-30 Oct while the clock
    buckets keep their meaning (e.g. the open and the close).
    """
    m, u = minute_of_day.to_numpy(), min_since_us_open.to_numpy()
    out = np.full(len(m), "", dtype=object)
    for lo, hi, label in CLOCK_BUCKETS:
        out[(m >= lo) & (m < hi)] = label
    for lo, hi, label in US_BUCKETS:
        out[(u >= lo) & (u < hi)] = label
    return pd.Series(out, index=minute_of_day.index)


def build_panel(freq: str = "5min") -> pd.DataFrame:
    """Bars + impact + volatility forecasts + lagged controls + benchmarks, one row per stock-bar."""
    p = config.PROCESSED
    b = (
        pd.read_parquet(p / f"bars_{freq}.parquet")
        .sort_values(["stock", "ts"])
        .reset_index(drop=True)
    )
    b = b.merge(
        pd.read_parquet(p / "impact_intraday_5min.parquet")[
            ["stock", "ts", "impact_roll_ewm_bps"]
        ],
        on=["stock", "ts"],
        how="left",
    )
    daily = pd.read_parquet(p / "impact_daily.parquet")[
        ["stock", "date", "var_impact_pred_bps"]
    ]
    b = b.merge(
        daily.rename(columns={"var_impact_pred_bps": "impact_prevday_bps"}),
        on=["stock", "date"],
        how="left",
    )
    vol = pd.read_parquet(p / "vol_forecast_5min.parquet")[
        ["stock", "ts", "vol_fc_har_ratio", "vol_fc_gjr_ratio"]
    ]
    b = b.merge(
        vol, on=["stock", "ts"], how="left"
    )  # 15-min bars take the forecast for their first 5-min bar
    b["log_vol_har"] = np.log(b["vol_fc_har_ratio"])
    b["log_vol_gjr"] = np.log(b["vol_fc_gjr_ratio"])

    g = b.groupby("stock")
    b["lag_log_trades"] = np.log1p(g["trades"].shift(1))
    b["lag_log_turnover"] = np.log1p(g["dkk_turnover"].shift(1))
    b["lag_abs_ofi"] = (
        g["ofi"].shift(1).abs().fillna(0.0)
    )  # no trades last bar -> no imbalance
    b["tod"] = tod_bucket(b["minute_of_day"], b["min_since_us_open"])
    b["event_day"] = (b["event"] != "").astype(int)
    b["p_gt_1tick"] = 1 - b["pct_time_1tick"] / 100
    b["log_quoted_hs"] = np.log(b["quoted_hs_bps"])

    # benchmarks: stock's in-sample average, and yesterday's average (time-weighted over the day)
    is_mean = b[b["sample"] == "IS"].groupby("stock")["quoted_hs_bps"].mean()
    b["bench_is_avg"] = b["stock"].map(is_mean)
    day = (
        b.groupby(["stock", "date"])
        .apply(
            lambda x: np.average(x["quoted_hs_bps"].fillna(0), weights=x["q_sec"]),
            include_groups=False,
        )
        .rename("day_mean")
    )
    yday = day.groupby(level=0).shift(1).rename("bench_yday").reset_index()
    return b.merge(yday, on=["stock", "date"], how="left")


def coarse_bucket(minute_of_day: pd.Series, min_since_us_open: pd.Series) -> pd.Series:
    """Coarse time buckets for the stock x time refinement (script 08), keyed to the actual US open.

    09:00-09:15, 09:15-09:30, 09:30-10:00, 10:00-12:00, 12:00-US open, US open +30 min,
    rest of the afternoon, last 15 min of continuous trading (16:40-16:55).
    """
    m, u = minute_of_day.to_numpy(), min_since_us_open.to_numpy()
    conds = [m < 15, m < 30, m < 60, m < 180, u < 0, u < 30, m < 460]
    labels = [
        "09:00",
        "09:15",
        "09:30",
        "10:00-12:00",
        "12:00-US",
        "US+0-30",
        "afternoon",
    ]
    return pd.Series(
        np.select(conds, labels, default="last15"), index=minute_of_day.index
    )


def design(df: pd.DataFrame, xvars: list[str], stock_tod: bool = False) -> pd.DataFrame:
    """Regressors + time-of-day and stock dummies.

    Default (script 07): shared time-of-day dummies (reference 12:00) + stock fixed effects.
    stock_tod=True (script 08): stock x coarse-bucket dummies, which also absorb the stock
    fixed effects (reference: the first stock-bucket column).
    """
    if stock_tod:
        cell = (
            df["stock"].astype(str)
            + "|"
            + coarse_bucket(df["minute_of_day"], df["min_since_us_open"])
        )
        dummies = pd.get_dummies(cell, prefix="sxt", dtype=float).iloc[:, 1:]
        return sm.add_constant(
            pd.concat([df[xvars], dummies], axis=1), has_constant="add"
        )
    tod = pd.get_dummies(df["tod"], prefix="tod", dtype=float).drop(
        columns="tod_12:00", errors="ignore"
    )
    fe = pd.get_dummies(df["stock"], prefix="fe", dtype=float).iloc[:, 1:]
    return sm.add_constant(pd.concat([df[xvars], tod, fe], axis=1), has_constant="add")


def nw_lags(t: int) -> int:
    """Newey-West (1994) rule of thumb: floor(4 (T/100)^(2/9))."""
    return int(np.floor(4 * (t / 100) ** (2 / 9)))


def fit_ols(
    df: pd.DataFrame, y: str, xvars: list[str], lags: int, stock_tod: bool = False
):
    """OLS with panel Newey-West SEs (HAC within stock, rows sorted by stock then time)."""
    d = df.dropna(subset=[y, *xvars]).sort_values(["stock", "ts"])
    X = design(d, xvars, stock_tod)
    groups = pd.factorize(d["stock"])[0]
    res = sm.OLS(d[y], X).fit(
        cov_type="hac-panel", cov_kwds={"groups": groups, "maxlags": lags}
    )
    return res, d


def fit_frac_logit(df: pd.DataFrame, y: str, xvars: list[str], lags: int):
    """Fractional logit (Papke-Wooldridge) for a share in [0, 1], panel Newey-West SEs."""
    d = df.dropna(subset=[y, *xvars]).sort_values(["stock", "ts"])
    X = design(d, xvars)
    res = sm.GLM(d[y], X, family=sm.families.Binomial()).fit(
        cov_type="hac-panel",
        cov_kwds={"groups": pd.factorize(d["stock"])[0], "maxlags": lags},
    )
    return res, d


def predict(
    res, df: pd.DataFrame, xvars: list[str], stock_tod: bool = False
) -> pd.Series:
    """Predict with the in-sample design columns (missing dummy columns filled with 0)."""
    X = design(df, xvars, stock_tod).reindex(
        columns=res.model.exog_names, fill_value=0.0
    )
    return pd.Series(np.asarray(res.predict(X)), index=df.index)


def dm_test(loss_a: pd.Series, loss_b: pd.Series, lags: int) -> dict:
    """Diebold-Mariano on d = loss_a - loss_b (negative mean: model a better), HLN small-sample correction."""
    d = (loss_a - loss_b).dropna().to_numpy()
    t = len(d)
    u = d - d.mean()
    lrv = u @ u / t + 2 * sum(
        (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / t for k in range(1, lags + 1)
    )
    stat = (
        d.mean() / np.sqrt(lrv / t) * np.sqrt((t - 1) / t)
    )  # HLN correction for h = 1
    return {
        "mean_loss_diff": d.mean(),
        "dm_stat": stat,
        "p_value": 2 * stats.t.sf(abs(stat), df=t - 1),
        "n": t,
    }


def key_table(
    res, d: pd.DataFrame, label: str, xvars: list[str], logit: bool = False
) -> pd.DataFrame:
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
