"""Time bars (1/5/15-min) built from signed continuous trades.

Quotes are only observed at trades, so the trade-attached quote is carried
forward until the next trade within the day. Bars hold contemporaneous values
only; lagged features are built later so the no-look-ahead rule lives in one place.
"""

import numpy as np
import pandas as pd

from . import config


def tick_size(bid: pd.Series, ask: pd.Series) -> float:
    """Smallest observed step between distinct quote prices."""
    steps = np.diff(np.unique(np.r_[bid, ask]).round(4))
    return float(steps[steps > 0].min())


def us_open(dates: pd.Series) -> pd.Series:
    """US cash open (09:30 New York) in Copenhagen time, per date."""
    hhmm = (
        dates.dt.strftime("%Y-%m-%d")
        .map(config.US_OPEN_CPH)
        .fillna(config.US_OPEN_CPH["default"])
    )
    return dates.dt.normalize() + pd.to_timedelta(hhmm + ":00")


def minute_bars(g: pd.DataFrame, day: pd.Timestamp, tick: float) -> pd.DataFrame:
    """1-min bars for one stock-day, 09:00 to the last continuous minute (16:54).

    Per minute: time-weighted sums for the quote (seconds covered, half-spread x
    seconds, 1-tick seconds) and trade sums, so coarser bars aggregate exactly.
    """
    start = day + pd.Timedelta(config.CONT_START)
    end = day + pd.Timedelta(config.LAST_CONT_TRADE)
    edges = pd.date_range(start, end, freq="min", inclusive="left")
    g = g[(g["ts"] >= start) & (g["ts"] < end)]

    # quote path: trade quotes plus minute boundaries, carried forward, durations to the next event
    q = pd.DataFrame(
        {
            "ts": g["ts"].to_numpy(),
            "hs": ((g["ask"] - g["bid"]) / 2 / g["mid"] * 1e4).to_numpy(),
            "one": (((g["ask"] - g["bid"]) / tick).round() == 1)
            .astype(float)
            .to_numpy(),
            "boundary": False,
        }
    )
    q = pd.concat([pd.DataFrame({"ts": edges, "boundary": True}), q], ignore_index=True)
    q = q.sort_values(
        ["ts", "boundary"], ascending=[True, False], kind="stable"
    )  # boundary before a trade at the same ts
    q[["hs", "one"]] = q[["hs", "one"]].ffill()  # NaN before the first trade of the day
    q["dur"] = q["ts"].shift(-1).fillna(end).sub(q["ts"]).dt.total_seconds()
    q["w"] = q["dur"].where(q["hs"].notna(), 0.0)
    q["minute"] = q["ts"].dt.floor("min")
    quote = pd.DataFrame(
        {
            "q_sec": q.groupby("minute")["w"].sum(),
            "hs_x_sec": (q["hs"].fillna(0) * q["w"]).groupby(q["minute"]).sum(),
            "one_x_sec": (q["one"].fillna(0) * q["w"]).groupby(q["minute"]).sum(),
        }
    )

    # trades: effective half-spread uses the trade sign; unsigned trades (rare) use |p - mid|
    eff = np.where(
        g["side"] != 0,
        g["side"] * (g["price"] - g["mid"]),
        (g["price"] - g["mid"]).abs(),
    )
    t = g.assign(
        eff_hs=eff / g["mid"] * 1e4,
        dkk=g["price"] * g["vol"],
        signed_vol=g["side"] * g["vol"],
        minute=g["ts"].dt.floor("min"),
    )
    trades = t.groupby("minute").agg(
        trades=("price", "size"),
        orders=("order_id", "nunique"),
        volume=("vol", "sum"),
        dkk_turnover=("dkk", "sum"),
        signed_vol=("signed_vol", "sum"),
        eff_hs_sum=("eff_hs", "sum"),
        mid_end=("mid", "last"),
    )

    m = quote.join(trades, how="outer").reindex(edges)
    m[["trades", "orders", "volume", "dkk_turnover", "signed_vol", "eff_hs_sum"]] = m[
        ["trades", "orders", "volume", "dkk_turnover", "signed_vol", "eff_hs_sum"]
    ].fillna(0)
    m["mid_end"] = m["mid_end"].ffill()
    m["ret_bps"] = (
        np.log(m["mid_end"]).diff() * 1e4
    )  # first minute NaN: opening-auction move excluded
    return m.rename_axis("ts").reset_index()


def aggregate(m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 1-min bars to `minutes`-min bars anchored at 09:00 and build the final columns."""
    m = m.assign(
        bar=(m["ts"] - m["date"] - pd.Timedelta(config.CONT_START)).dt.total_seconds()
        // (60 * minutes)
    )
    g = m.groupby(["stock", "date", "bar"])
    b = g.agg(
        ts=("ts", "first"),
        bar_minutes=("ts", "size"),
        q_sec=("q_sec", "sum"),
        hs_x_sec=("hs_x_sec", "sum"),
        one_x_sec=("one_x_sec", "sum"),
        trades=("trades", "sum"),
        orders=("orders", "sum"),
        volume=("volume", "sum"),
        dkk_turnover=("dkk_turnover", "sum"),
        signed_vol=("signed_vol", "sum"),
        eff_hs_sum=("eff_hs_sum", "sum"),
        mid_end=("mid_end", "last"),
    )
    b["ret_bps"] = g["ret_bps"].sum(min_count=1)
    b["rv_bps"] = np.sqrt(
        (m["ret_bps"] ** 2).groupby([m["stock"], m["date"], m["bar"]]).sum(min_count=1)
    )
    b = b.reset_index().drop(columns="bar")

    q = b["q_sec"].replace(0, np.nan)
    b["quoted_hs_bps"] = b["hs_x_sec"] / q
    b["pct_time_1tick"] = 100 * b["one_x_sec"] / q
    b["eff_hs_bps"] = b["eff_hs_sum"] / b["trades"].replace(0, np.nan)
    b["ofi"] = b["signed_vol"] / b["volume"].replace(0, np.nan)
    b["clock"] = b["ts"].dt.strftime("%H:%M")
    b["minute_of_day"] = (
        (b["ts"] - b["date"] - pd.Timedelta(config.CONT_START)).dt.total_seconds() // 60
    ).astype(int)
    b["min_since_us_open"] = (
        (b["ts"] - us_open(b["date"])).dt.total_seconds() // 60
    ).astype(int)
    b["group"] = np.where(
        b["stock"].isin(config.CONSTRAINED), "constrained", "unconstrained"
    )
    b["sample"] = np.where(b["date"] <= config.IS_END, "IS", "OOS")
    b["event"] = [
        config.EVENT_DAYS.get((s, str(d.date())), "")
        for s, d in zip(b["stock"], b["date"])
    ]
    b["thin"] = b["trades"] < 5
    cols = [
        "stock",
        "group",
        "date",
        "sample",
        "event",
        "ts",
        "clock",
        "minute_of_day",
        "min_since_us_open",
        "bar_minutes",
        "quoted_hs_bps",
        "eff_hs_bps",
        "pct_time_1tick",
        "q_sec",
        "mid_end",
        "ret_bps",
        "rv_bps",
        "trades",
        "orders",
        "volume",
        "dkk_turnover",
        "signed_vol",
        "ofi",
        "thin",
    ]
    return b[cols]
