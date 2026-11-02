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


def fit_var(ev: pd.DataFrame, x: str, p: int) -> dict:
    """OLS of the two-equation VAR with order flow ordered first (x_t enters the r equation).

    r_t = c + sum_{i=0..p} b_i x_{t-i} + sum_{i=1..p} a_i r_{t-i}
    x_t = c + sum_{i=1..p} d_i x_{t-i} + sum_{i=1..p} g_i r_{t-i}
    Lags are built within the day, so the first p orders of each day are dropped.
    """
    xl, rl = _lags(ev, x, p), _lags(ev, "r_bps", p)
    xt, rt = ev[x].to_numpy(float), ev["r_bps"].to_numpy(float)
    ok = ~np.isnan(np.column_stack([xl, rl, xt, rt])).any(axis=1)
    n, one = ok.sum(), np.ones(ok.sum())
    Xr = np.column_stack([one, xt[ok], xl[ok], rl[ok]])
    Xx = np.column_stack([one, xl[ok], rl[ok]])
    br, *_ = np.linalg.lstsq(Xr, rt[ok], rcond=None)
    bx, *_ = np.linalg.lstsq(Xx, xt[ok], rcond=None)
    ssr_r, ssr_x = ((rt[ok] - Xr @ br) ** 2).sum(), ((xt[ok] - Xx @ bx) ** 2).sum()
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
