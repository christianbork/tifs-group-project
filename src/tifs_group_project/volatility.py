"""Volatility forecasts for 5-min bars from 1-min mid returns.

Returns are deseasonalised by an in-sample time-of-day factor that follows the
US open (15:30 CPH in-sample, 14:30 CPH on 26-30 Oct). Every forecast for a bar
is made at the end of the previous bar, so it uses returns up to t-1 only.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def seasonal_variance(
    m1: pd.DataFrame, smooth: int = 5, us_window: int = 60
) -> pd.Series:
    """Expected squared 1-min return (bps^2) per row of one stock's 1-min bars.

    v(m): in-sample mean of r^2 by minute of day. The US-open bump = v in the first
    `us_window` minutes after the in-sample US open relative to the mean of the hour
    before it; outside that window the pattern is clock time (e.g. the close). The
    bump is removed from the raw clock pattern *before* smoothing (centred
    `smooth`-min window), the two pieces are smoothed separately, and the bump is
    re-applied at each day's own US open. Removing the bump first matters: smoothing
    the raw curve first spread the 15:30 spike into 15:28-15:29, which then stayed in
    the clock pattern and was re-applied on the 14:30-open days (26-30 Oct) as a
    spurious spike one bar before the real one.
    """
    m1 = m1.assign(us_min=m1["minute_of_day"] - m1["min_since_us_open"])
    is_ = m1[m1["sample"] == "IS"]
    v_raw = (
        (is_["ret_bps"] ** 2)
        .groupby(is_["minute_of_day"])
        .mean()
        .reindex(range(m1["minute_of_day"].max() + 1))
        .ffill()
        .bfill()
    )
    us_is = int(is_["us_min"].mode().iloc[0])
    assert (is_["us_min"] == us_is).all(), "in-sample days must share one US-open time"
    base = v_raw.iloc[us_is - 60 : us_is].mean()
    bump_raw = v_raw.iloc[us_is : us_is + us_window].to_numpy() / base
    bump_s = pd.Series(bump_raw).rolling(smooth, center=True, min_periods=1).mean().to_numpy()

    def bump(k: np.ndarray, vals: np.ndarray) -> np.ndarray:
        k = np.asarray(k)
        inside = (k >= 0) & (k < len(vals))
        return np.where(inside, vals[np.clip(k, 0, len(vals) - 1)], 1.0)

    idx = np.arange(len(v_raw))
    clock = pd.Series(v_raw.to_numpy() / bump(idx - us_is, bump_raw))
    clock = clock.rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    m = m1["minute_of_day"].to_numpy()
    return pd.Series(clock[m] * bump(m - m1["us_min"].to_numpy(), bump_s), index=m1.index)


def garch_variance_forecasts(
    z: np.ndarray, n_is: int, gjr: bool, horizon: int = 5
) -> tuple[np.ndarray, pd.Series]:
    """Fit (GJR-)GARCH(1,1) on the first n_is obs; forecast 1..horizon steps from every origin.

    Row t of the result = variance forecasts for t+1..t+horizon using data up to t,
    with the parameters fixed at their in-sample estimates.
    """
    am = arch_model(
        z, mean="Zero", vol="GARCH", p=1, o=int(gjr), q=1, dist="normal", rescale=False
    )
    res = am.fit(last_obs=n_is, disp="off")
    fc = res.forecast(horizon=horizon, start=0, reindex=True, align="origin")
    return fc.variance.to_numpy(), res.params


def bar_forecast(m1: pd.DataFrame, f: np.ndarray) -> pd.Series:
    """Sum minute-level variance forecasts into 5-min bar forecasts (bps^2).

    m1 must hold only valid-return minutes in time order, with columns `bar_ts`
    and `e` (seasonal variance). Origin = the last valid minute before the bar;
    the bar's j-th valid minute uses horizon j from that origin.
    """
    pos = np.arange(len(m1))
    origin = (
        pd.Series(pos, index=m1.index).groupby(m1["bar_ts"]).transform("min").to_numpy()
        - 1
    )
    j = pos - origin  # 1..5
    ok = (origin >= 0) & (j <= f.shape[1])
    contrib = np.full(len(m1), np.nan)
    contrib[ok] = m1["e"].to_numpy()[ok] * f[origin[ok], j[ok] - 1]
    return pd.Series(contrib, index=m1.index).groupby(m1["bar_ts"]).sum(min_count=1)


def qlike(rv2: pd.Series, f2: pd.Series) -> float:
    """QLIKE loss in the form log F + RV/F (defined when RV = 0; same ranking as Patton 2011)."""
    return float((np.log(f2) + rv2 / f2).mean())


def mincer_zarnowitz(rv2: pd.Series, f2: pd.Series, lags: int = 12) -> dict:
    """RV^2 = a + b F^2 with Newey-West errors; Wald p-value for a = 0, b = 1."""
    X = sm.add_constant(f2.to_numpy())
    res = sm.OLS(rv2.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    w = res.wald_test("const = 0, x1 = 1", scalar=True)
    return {
        "mz_a": res.params[0],
        "mz_b": res.params[1],
        "mz_r2": res.rsquared,
        "mz_p_unbiased": float(w.pvalue),
    }


def ljung_box(x: np.ndarray, lag: int = 10) -> tuple[float, float]:
    """Ljung-Box Q(lag) statistic and p-value (H0: no autocorrelation up to `lag`)."""
    r = acorr_ljungbox(np.asarray(x), lags=[lag])
    return float(r["lb_stat"].iloc[0]), float(r["lb_pvalue"].iloc[0])


def arch_lm(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Engle's ARCH-LM test (H0: no ARCH effects up to `lags`): LM statistic and p-value."""
    lm, lm_p, _, _ = het_arch(np.asarray(x), nlags=lags)
    return float(lm), float(lm_p)


def garch_diagnostics(z: np.ndarray, n_is: int, gjr: bool, dist: str = "normal", lag: int = 10) -> dict:
    """AE Lab 8 checks for a (GJR-)GARCH(1,1) fitted on the first n_is observations.

    Before: ARCH effects in the (deseasonalised) returns. After: the standardised
    residuals e/sigma and their squares should look like white noise; a remaining
    ARCH-LM rejection means the variance model has not captured all clustering.
    """
    am = arch_model(z[:n_is], mean="Zero", vol="GARCH", p=1, o=int(gjr), q=1, dist=dist, rescale=False)
    res = am.fit(disp="off")
    std = np.asarray(res.std_resid)
    std = std[np.isfinite(std)]
    lb_q, lb_p = ljung_box(std, lag)
    lb2_q, lb2_p = ljung_box(std**2, lag)
    lm, lm_p = arch_lm(std, lag)
    return {
        "loglik": float(res.loglikelihood),
        "bic": float(res.bic),
        "nu": float(res.params["nu"]) if "nu" in res.params else np.nan,
        "lb_std_q": lb_q,
        "lb_std_p": lb_p,
        "lb_sq_std_q": lb2_q,
        "lb_sq_std_p": lb2_p,
        "arch_lm_std": lm,
        "arch_lm_std_p": lm_p,
    }
