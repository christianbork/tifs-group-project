"""Market-maker quoting-rule simulation on the signed trade feed.

A hypothetical market maker posts a half-spread h_t (bps of the mid) around the
prevailing mid for every 5-min bar, chosen before the bar starts (a forecast or a
benchmark). Each aggressive lit trade in the bar is a potential fill: the aggressor
paid an effective half-spread e = side * (p - mid) / mid; if the market maker's quote
was at least as good (h_t <= e) it would have been first in the queue and takes the
trade, up to its quote size. Revenue per filled share is h_t; the adverse-selection
cost is the mid move over the next 5 minutes (the decomposition in `impact.py`);
realised = revenue - adverse selection. Quotes live on the tick grid: h is snapped to
the nearest multiple of half a tick (never below one), because a quote cannot sit inside
the grid and the effective half-spread paid by an aggressor is always a whole number of
half-ticks. Fills are counted per aggressive *order*, not per print: the fills that share
a timestamp are one order walking the book, and the market maker's quote is consumed
once by that order (see `orders`).

Rules compared: in-sample average, yesterday's average, last bar, the 07/08/09 model
forecasts, and the realised quoted half-spread of the bar as an oracle reference.
"""

import numpy as np
import pandas as pd


def fills(
    trades: pd.DataFrame, h_bps: pd.Series, quote_size: float, half_tick_bps: pd.Series
) -> pd.DataFrame:
    """Per-trade fill outcome for one quoting rule.

    trades: signed lit trades with columns side (+1/-1), vol, mid, eff_bps, impact_5m_bps.
    h_bps: the rule's half-spread for each trade's bar (aligned to trades.index).
    half_tick_bps: half a tick in bps of the mid, per trade (floor for h).
    """
    ht = half_tick_bps.to_numpy(float)
    k = np.maximum(1.0, np.round(h_bps.to_numpy(float) / ht))  # snap to the tick grid, floor at half a tick
    h = k * ht
    e = trades["eff_bps"].to_numpy(float)
    filled = (h <= e) & np.isfinite(h) & np.isfinite(e)
    size = np.minimum(trades["vol"].to_numpy(float), quote_size)
    out = pd.DataFrame(index=trades.index)
    out["h_bps"] = h
    out["filled"] = filled
    out["size"] = np.where(filled, size, 0.0)
    out["revenue_bps"] = np.where(filled, h, np.nan)
    out["impact_bps"] = np.where(
        filled, trades["impact_5m_bps"].to_numpy(float), np.nan
    )
    out["realised_bps"] = out["revenue_bps"] - out["impact_bps"]
    notional = out["size"] * trades["mid"].to_numpy(float)
    out["pnl_dkk"] = np.where(filled, notional * out["realised_bps"] / 1e4, 0.0)
    # the market maker buys when the aggressor sells (side = -1)
    out["signed_size"] = -trades["side"].to_numpy(float) * out["size"]
    return out


def orders(trades: pd.DataFrame) -> pd.DataFrame:
    """Collapse the fills of each aggressive order (same stock and order_id) to one row.

    The quote, mid, effective half-spread and 5-min impact are those of the first fill
    (the one that hit the touch, where the market maker's quote would have sat); the
    volume is the order's total. Columns other than vol are taken from the first fill.
    """
    t = trades.sort_values(["stock", "order_id", "fill_pos"], kind="stable")
    first = t.groupby(["stock", "order_id"], sort=False).head(1).copy()
    vol = t.groupby(["stock", "order_id"], sort=False)["vol"].sum()
    first["vol"] = vol.reindex(pd.MultiIndex.from_frame(first[["stock", "order_id"]])).to_numpy()
    first["fills_in_order"] = t.groupby(["stock", "order_id"], sort=False).size().to_numpy()
    return first.sort_values("ts", kind="stable").reset_index(drop=True)


def summarise(trades: pd.DataFrame, f: pd.DataFrame) -> dict:
    """Fill rate, spread capture, adverse selection, P&L and inventory for one rule."""
    filled = f["filled"].to_numpy()
    n = int(filled.sum())
    if n == 0:
        return {"fills": 0, "fill_rate_pct": 0.0}
    w = f.loc[filled, "size"]
    day = trades["date"]
    pnl_day = f["pnl_dkk"].groupby(day).sum()
    # inventory in DKK, accumulated in time order within the day (rows must be sorted by ts)
    inv = (f["signed_size"] * trades["mid"]).groupby(day).cumsum()
    return {
        "fills": n,
        "fill_rate_pct": 100 * n / len(f),
        "volume_share_pct": 100 * w.sum() / trades["vol"].sum(),
        "avg_half_spread_bps": float(
            np.average(f.loc[filled, "revenue_bps"], weights=w)
        ),
        "adverse_selection_bps": float(
            np.average(f.loc[filled, "impact_bps"], weights=w)
        ),
        "realised_bps": float(np.average(f.loc[filled, "realised_bps"], weights=w)),
        "pnl_dkk_per_day": float(pnl_day.mean()),
        "pnl_dkk_sd_day": float(pnl_day.std(ddof=0)),
        "pnl_sharpe_daily": float(pnl_day.mean() / pnl_day.std(ddof=0))
        if pnl_day.std(ddof=0) > 0
        else np.nan,
        "pct_days_positive": 100 * float((pnl_day > 0).mean()),
        "max_abs_inventory_dkk": float(inv.abs().max()),
        "days": int(pnl_day.size),
    }
