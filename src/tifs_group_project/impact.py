"""Price impact: Hasbrouck (1991) VAR in order time and the spread decomposition.

Event = one aggressive order (fills sharing a timestamp, `order_id`). The
trade-attached quote at an order's first fill approximates the pre-trade quote.
"""

import numpy as np
import pandas as pd

from . import config


def order_events(d: pd.DataFrame) -> pd.DataFrame:
    """One row per order: pre-trade mid, sign, signed volume, and the mid return to the next order.

    r_bps for order k = log mid at order k+1's first fill - log mid at order k's
    first fill (bps); NaN for the last order of the day, so nothing crosses days.
    """
    d = d.sort_values(["ts", "fill_pos"], kind="stable")
    ev = (
        d.groupby("order_id")
        .agg(
            date=("date", "first"),
            ts=("ts", "first"),
            mid=("mid", "first"),
            sign=("side", "first"),
            vol=("vol", "sum"),
        )
        .sort_values("ts", kind="stable")
    )
    ev["signed_vol"] = ev["sign"] * ev["vol"]
    ev["r_bps"] = np.log(ev.groupby("date")["mid"].shift(-1) / ev["mid"]) * 1e4
    return ev.reset_index()


def _lags(ev: pd.DataFrame, col: str, p: int) -> np.ndarray:
    g = ev.groupby("date")[col]
    return np.column_stack([g.shift(i).to_numpy() for i in range(1, p + 1)])


def var_design(ev: pd.DataFrame, x: str, p: int) -> tuple:
    """Event-level VAR regressors (lags within the day) and a mask of rows with complete data."""
    xl, rl = _lags(ev, x, p), _lags(ev, "r_bps", p)
    xt, rt = ev[x].to_numpy(float), ev["r_bps"].to_numpy(float)
    ok = ~np.isnan(np.column_stack([xl, rl, xt, rt])).any(axis=1)
    one = np.ones(len(ev))
    return (
        np.column_stack([one, xt, xl, rl]),
        rt,
        np.column_stack([one, xl, rl]),
        xt,
        ok,
    )


def fit_var(ev: pd.DataFrame, x: str, p: int) -> dict:
    """OLS of the two-equation VAR with order flow ordered first (x_t enters the r equation).

    r_t = c + sum_{i=0..p} b_i x_{t-i} + sum_{i=1..p} a_i r_{t-i}
    x_t = c + sum_{i=1..p} d_i x_{t-i} + sum_{i=1..p} g_i r_{t-i}
    Lags are built within the day, so the first p orders of each day are dropped.
    """
    Xr, rt, Xx, xt, ok = var_design(ev, x, p)
    return fit_design(Xr[ok], rt[ok], Xx[ok], xt[ok], p)


def fit_design(
    Xr: np.ndarray, rt: np.ndarray, Xx: np.ndarray, xt: np.ndarray, p: int
) -> dict:
    n = len(rt)
    br, *_ = np.linalg.lstsq(Xr, rt, rcond=None)
    bx, *_ = np.linalg.lstsq(Xx, xt, rcond=None)
    ssr_r, ssr_x = ((rt - Xr @ br) ** 2).sum(), ((xt - Xx @ bx) ** 2).sum()
    bic = (
        n * np.log(ssr_r / n)
        + Xr.shape[1] * np.log(n)
        + n * np.log(ssr_x / n)
        + Xx.shape[1] * np.log(n)
    )
    return {
        "n": int(n),
        "p": p,
        "b": br[1 : p + 2],
        "a": br[p + 2 :],
        "d": bx[1 : p + 1],
        "g": bx[p + 1 :],
        "bic": bic,
    }


def cum_irf(v: dict, h: int) -> np.ndarray:
    """Cumulative mid response (bps) to a one-unit buy shock in x at t=0, for t = 0..h."""
    p = v["p"]
    x, r = (
        np.zeros(h + 1 + p),
        np.zeros(h + 1 + p),
    )  # p leading zeros = pre-shock history
    for t in range(p, h + 1 + p):
        past_x, past_r = x[t - p : t][::-1], r[t - p : t][::-1]
        x[t] = 1.0 if t == p else v["d"] @ past_x + v["g"] @ past_r
        r[t] = v["b"][0] * x[t] + v["b"][1:] @ past_x + v["a"] @ past_r
    return np.cumsum(r[p:])


def decomposition(d: pd.DataFrame, horizons_min=(1, 5)) -> pd.DataFrame:
    """Per trade: effective = realised + price impact half-spreads (bps), mid at t+h within the day.

    The future mid is the last trade-attached mid at or before t+h; trades whose
    t+h passes the end of continuous trading get NaN.
    """
    d = d.sort_values("ts", kind="stable").reset_index(drop=True)
    out = pd.DataFrame({"date": d["date"], "ts": d["ts"]})
    sign = np.where(d["side"] != 0, d["side"], np.sign(d["price"] - d["mid"]))
    out["eff_bps"] = sign * (d["price"] - d["mid"]) / d["mid"] * 1e4
    quotes = d[["ts", "date", "mid"]].rename(columns={"ts": "ts_q", "mid": "mid_h"})
    for h in horizons_min:
        tgt = pd.DataFrame(
            {"ts_q": d["ts"] + pd.Timedelta(minutes=h), "date": d["date"], "i": d.index}
        )
        m = (
            pd.merge_asof(tgt.sort_values("ts_q"), quotes, on="ts_q", by="date")
            .set_index("i")
            .sort_index()
        )
        late = tgt["ts_q"] >= tgt["date"] + pd.Timedelta(config.LAST_CONT_TRADE)
        mid_h = m["mid_h"].where(~late)
        out[f"impact_{h}m_bps"] = sign * (mid_h - d["mid"]) / d["mid"] * 1e4
        out[f"realised_{h}m_bps"] = out["eff_bps"] - out[f"impact_{h}m_bps"]
    return out


def windows(dates: list, days: int) -> list[tuple]:
    """(end_date, [dates]) for rolling windows of `days` trading days; days=1 gives single days."""
    return [
        (dates[i], dates[i - days + 1 : i + 1]) for i in range(days - 1, len(dates))
    ]


def rolling_impact(
    ev: pd.DataFrame,
    bar_starts: pd.Series,
    p_var: int = 5,
    dl_lags: int = 5,
    window: str = "60min",
    h: int = 20,
    min_reg: int = 50,
    min_var: int = 100,
) -> pd.DataFrame:
    """Intraday impact at each bar start tau from orders in [tau - window, tau), same day.

    An order is used only if its mid return is complete before tau, i.e. the next
    order (whose mid ends the return) also arrived before tau. Lagged regressors
    may reach back before the window start (still pre-tau, same day).
    - reg: r_k = a + sum_{i=0..dl_lags} lambda_i sign_{k-i}; impact = sum of lambdas.
    - var: Hasbrouck VAR(p_var) on the window, h-order cumulative IRF.
    """
    ev = ev.sort_values("ts", kind="stable").reset_index(drop=True)
    next_ts = ev.groupby("date")["ts"].shift(-1)
    sl = np.column_stack([ev["sign"].to_numpy(float), _lags(ev, "sign", dl_lags)])
    Xg = np.column_stack([np.ones(len(ev)), sl])
    rt = ev["r_bps"].to_numpy(float)
    ok_reg = ~np.isnan(np.column_stack([Xg, rt])).any(axis=1)
    Xr, _, Xx, xt, ok_var = var_design(ev, "sign", p_var)
    ts, nts = ev["ts"].to_numpy(), next_ts.to_numpy()
    span = pd.Timedelta(window).to_timedelta64()

    out = []
    for tau in bar_starts.to_numpy():
        inw = (ts >= tau - span) & (
            nts < tau
        )  # NaT next_ts (last order of day) compares False
        m_reg, m_var = inw & ok_reg, inw & ok_var
        reg = var = np.nan
        if m_reg.sum() >= min_reg:
            coef, *_ = np.linalg.lstsq(Xg[m_reg], rt[m_reg], rcond=None)
            reg = coef[1:].sum()
        if m_var.sum() >= min_var:
            var = cum_irf(
                fit_design(Xr[m_var], rt[m_var], Xx[m_var], xt[m_var], p_var), h
            )[h]
        out.append((tau, reg, var, int(m_reg.sum())))
    return pd.DataFrame(
        out,
        columns=["ts", "impact_roll_reg_bps", "impact_roll_var_bps", "roll_n_orders"],
    )
