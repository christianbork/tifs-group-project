"""Trade signing (buyer- vs seller-initiated) and quote-timing diagnostics."""

import numpy as np
import pandas as pd


def add_order_groups(c: pd.DataFrame) -> pd.DataFrame:
    """Fills sharing a timestamp are treated as one aggressive order walking the book."""
    c = c.sort_values("ts", kind="stable").copy()
    grp = c.groupby(["date", "ts"], sort=False)
    c["order_id"] = grp.ngroup()
    c["fill_pos"] = grp.cumcount()
    c["fills_in_order"] = grp["price"].transform("size")
    return c


def tick_test(price: pd.Series, date: pd.Series) -> pd.Series:
    """+1 if price above the last different price (same day), -1 if below, 0 if no prior change."""
    # compare each price with the most recent price that differs from it (same day)
    out = np.zeros(len(price))
    p, d = price.to_numpy(), date.to_numpy()
    last_diff, cur_day, prev_p = np.nan, None, np.nan
    for i in range(len(p)):
        if d[i] != cur_day:
            cur_day, last_diff, prev_p = d[i], np.nan, np.nan
        if not np.isnan(prev_p) and p[i] != prev_p:
            last_diff = prev_p
        if not np.isnan(last_diff):
            out[i] = 1 if p[i] > last_diff else (-1 if p[i] < last_diff else 0)
        prev_p = p[i]
    return pd.Series(out.astype(int), index=price.index)


def sign_trades(c: pd.DataFrame) -> pd.DataFrame:
    """Lee-Ready style: quote rule against the trade-attached mid, tick test for midpoint trades.

    The attached quote is used because it fits better than lagged quotes (see 02 tables)
    and matches the prevailing quote ~93-95% of the time in the ISS quotes sample.
    """
    c = c.copy()
    q = np.select([c["price"] > c["mid"], c["price"] < c["mid"]], [1, -1], 0)
    t = tick_test(c["price"], c["date"]).to_numpy()
    c["side"] = np.where(q != 0, q, t)
    c["sign_rule"] = np.select([q != 0, t != 0], ["quote", "tick"], "unsigned")
    return c


def quote_fit(price, bid, ask) -> dict:
    ok = bid.notna() & ask.notna() & (ask > bid)
    p, b, a = price[ok], bid[ok], ask[ok]
    return {
        "pct_outside": round(100 * ((p > a) | (p < b)).mean(), 1),
        "pct_at_quote": round(100 * ((p == a) | (p == b)).mean(), 1),
        "pct_inside": round(100 * ((p < a) & (p > b)).mean(), 1),
    }


def rebuild_quotes(q: pd.DataFrame) -> pd.DataFrame:
    """From a FactSet quotes export (one side per update row), rebuild the L1 quote stream."""
    q = q.iloc[::-1].reset_index(drop=True)
    q["ts"] = pd.to_datetime(q["Date"] + " " + q["Time (ms)"], format="%Y/%m/%d %H:%M:%S.%f")
    is_trade = q["Exch Time"].notna()
    u = q.loc[~is_trade, ["ts", "Bid", "Ask"]].copy()
    u["bid"] = u["Bid"].replace(0, np.nan).ffill()
    u["ask"] = u["Ask"].replace(0, np.nan).ffill()
    return u[["ts", "bid", "ask"]].dropna()
