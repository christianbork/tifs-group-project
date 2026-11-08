"""Quote-widening rule: turn spread forecasts into quotes and score them by markout P&L.

Stylised market maker (Week 6 "risk adjustment function"):
- At the start of each 5-min bar it quotes bid/ask at mid -/+ h_t (bps), with h_t set by a
  rule that uses only information before the bar (a spread forecast from script 08).
- It is filled once by an incoming aggressive order whenever that order reached at least
  h_t away from its pre-trade mid (the aggressor reached its price level); it trades at its
  own price, mid +/- h_t. One quote, so at most one fill per order, however many levels
  the order walks.
- Each fill is marked out at the mid H minutes later (same day), as in the realised-spread
  decomposition of script 05: P&L (bps) = h_t - signed mid move = h_t - price impact.

Assumptions (state them in the report): optimistic queue priority at its own price level,
its quotes do not change other traders' behaviour, a fixed participation share of each
fill's size, no fees/rebates, inventory flattened at the markout (no inventory carry).
"""

import numpy as np
import pandas as pd


def fills(trades: pd.DataFrame, quotes: pd.DataFrame, rule: str, participation: float = 0.1) -> pd.DataFrame:
    """Fills of one quoting rule.

    trades: one row per aggressive order (or trade) with stock, date, ts, bar_ts, price, vol,
        eff_bps (deepest distance from the pre-trade mid in the aggressor's direction) and
        impact_bps (signed mid move to the markout horizon), NaN where the markout passes
        the end of trading.
    quotes: one row per stock-bar with stock, bar_ts and the column `rule` = quoted half-spread h_t (bps).
    """
    t = trades.merge(quotes[["stock", "bar_ts", rule]], on=["stock", "bar_ts"], how="inner")
    t = t[t[rule].notna() & t["impact_bps"].notna()]
    hit = t["eff_bps"] >= t[rule] - 1e-9
    f = t.loc[hit].copy()
    f["h_bps"] = f[rule]
    f["pnl_bps"] = f["h_bps"] - f["impact_bps"]
    f["notional"] = participation * f["price"] * f["vol"]
    f["pnl_dkk"] = f["pnl_bps"] / 1e4 * f["notional"]
    f["rule"] = rule
    return f[["stock", "date", "ts", "bar_ts", "rule", "h_bps", "impact_bps", "pnl_bps", "notional", "pnl_dkk"]]


def summarise(f: pd.DataFrame, quotes: pd.DataFrame, rule: str) -> dict:
    """Headline metrics for one rule (and one stock or group): fills, width, markout, P&L."""
    n_days = f["date"].nunique() if len(f) else np.nan
    w = f["notional"]
    daily = f.groupby(["stock", "date"])["pnl_dkk"].sum()
    return {
        "rule": rule,
        "avg_quote_bps": quotes[rule].mean(),
        "fills": len(f),
        "notional_dkk_m": w.sum() / 1e6,
        "spread_capture_bps": np.average(f["h_bps"], weights=w) if len(f) else np.nan,
        "adverse_sel_bps": np.average(f["impact_bps"], weights=w) if len(f) else np.nan,
        "markout_pnl_bps": np.average(f["pnl_bps"], weights=w) if len(f) else np.nan,
        "pnl_dkk": f["pnl_dkk"].sum(),
        "pnl_dkk_per_stock_day": daily.mean() if len(daily) else np.nan,
        "pnl_sd_per_stock_day": daily.std() if len(daily) > 1 else np.nan,
        "days": n_days,
    }


def paired_daily_test(f_a: pd.DataFrame, f_b: pd.DataFrame, keys: pd.MultiIndex) -> dict:
    """Paired t-test across stock-days of the daily P&L difference (rule a minus rule b)."""
    a = f_a.groupby(["stock", "date"])["pnl_dkk"].sum().reindex(keys, fill_value=0.0)
    b = f_b.groupby(["stock", "date"])["pnl_dkk"].sum().reindex(keys, fill_value=0.0)
    d = (a - b).to_numpy()
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"mean_diff_dkk": d.mean(), "t": d.mean() / se if se else np.nan, "n_stock_days": n, "share_days_better": (d > 0).mean()}


def scale_to_is_width(pred_is: pd.Series, target_is: pd.Series) -> float:
    """Scale factor so a rule's in-sample average quote equals the in-sample average spread.

    Uses in-sample rows only, so out-of-sample quotes stay free of look-ahead. With equal
    average width, any P&L difference between rules comes from WHEN they widen, not how much.
    """
    ok = pred_is.notna() & target_is.notna()
    return float(target_is[ok].mean() / pred_is[ok].mean())
