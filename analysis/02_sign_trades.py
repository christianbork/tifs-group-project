# %% [markdown]
# # 02 Trade signing and quote-timing test
# 1. Which quote should a trade be compared with: the quote attached to the
#    trade, the previous row's quote, or the quote before the trade's timestamp?
# 2. Why do some trades print outside the quote? (orders walking the book)
# 3. Sign every continuous lit trade (+1 buy / -1 sell): quote rule, then tick test.
# 4. Validate against the ISS 30-Oct full quote stream.

# %% Setup
import numpy as np
import pandas as pd

from tifs_group_project import config
from tifs_group_project.io import read_clean
from tifs_group_project.signing import add_order_groups, quote_fit, rebuild_quotes, sign_trades

# %% Quote timing and signing, per stock
timing, walk, summary = [], [], []
for stock in config.STOCKS:
    d = read_clean(stock).sort_values("ts", kind="stable")
    d = d[d["kind"] != "close"]
    g = d.groupby("date")
    d["bid_prev_row"], d["ask_prev_row"] = g["bid"].shift(), g["ask"].shift()
    prev_ts = (d.groupby(["date", "ts"])[["bid", "ask"]].last().groupby(level=0).shift()
               .rename(columns={"bid": "bid_prev_ts", "ask": "ask_prev_ts"}).reset_index())
    d = d.merge(prev_ts, on=["date", "ts"], how="left")

    c = d[d["continuous"] & d["quote_ok"]].copy()
    # summary tables describe the October sample only; the forward week (FWD) is signed
    # and saved below but never enters a descriptive table before script 09
    oct_ = c["sample"] != "FWD"
    for label, b, a in [("attached", "bid", "ask"), ("previous row", "bid_prev_row", "ask_prev_row"),
                        ("before timestamp", "bid_prev_ts", "ask_prev_ts")]:
        timing.append({"stock": stock, "quote": label, **quote_fit(c.loc[oct_, "price"], c.loc[oct_, b], c.loc[oct_, a])})

    c = add_order_groups(c)
    c["outside"] = (c["price"] > c["ask"]) | (c["price"] < c["bid"])
    w = c[oct_].groupby(np.minimum(c.loc[oct_, "fill_pos"], 3))["outside"].mean().mul(100).round(1)
    walk.append({"stock": stock, **{f"fill_{'4+' if k == 3 else k + 1}": v for k, v in w.items()}})

    c = sign_trades(c)
    co = c[c["sample"] != "FWD"]
    multi = co[co["fills_in_order"] > 1].groupby("order_id")["side"].nunique()
    summary.append({
        "stock": stock,
        "trades": len(co),
        "orders": co["order_id"].nunique(),
        "fills_per_order": round(len(co) / co["order_id"].nunique(), 2),
        "pct_quote_rule": round(100 * (co["sign_rule"] == "quote").mean(), 1),
        "pct_tick_test": round(100 * (co["sign_rule"] == "tick").mean(), 1),
        "pct_unsigned": round(100 * (co["sign_rule"] == "unsigned").mean(), 2),
        "pct_buy_volume": round(100 * co.loc[co["side"] == 1, "vol"].sum() / co.loc[co["side"] != 0, "vol"].sum(), 1),
        "pct_multi_fill_orders_one_side": round(100 * (multi == 1).mean(), 1),
    })
    keep = ["ts", "date", "stock", "sample", "price", "vol", "bid", "ask", "bid_size", "ask_size", "mid",
            "order_id", "fill_pos", "fills_in_order", "side", "sign_rule", "venue"]
    c[keep].to_parquet(config.PROCESSED / f"signed_{stock}.parquet", index=False)

timing = pd.DataFrame(timing).pivot(index="stock", columns="quote", values="pct_outside")[
    ["attached", "previous row", "before timestamp"]]
walk, summary = pd.DataFrame(walk).set_index("stock"), pd.DataFrame(summary).set_index("stock")
timing.to_csv(config.TABLES / "02_quote_timing_pct_outside.csv")
walk.to_csv(config.TABLES / "02_pct_outside_by_fill_position.csv")
summary.to_csv(config.TABLES / "02_signing_summary.csv")
print("% of trades outside the quote, by quote choice:\n", timing, "\n")
print("% outside the quote by fill position within an order:\n", walk, "\n")
print("Signing summary:\n", summary.to_string(), "\n")

# %% Validation against the full L1 quote streams in data/raw/quotes_sample/
# One row per quote file: ISS 30-Oct (Christian), ISS 5-Nov, GN 5-Nov (event day, two parts)
# and a one-hour Novo window on 6-Nov. Trades are compared with the prevailing quote
# rebuilt from the stream; the Novo file only covers 10:00-11:00, so its trades are
# restricted to the span of the quote file.
import re

TICKER_TO_STOCK = {v: k for k, v in config.STOCKS.items()}
qfiles = {}
for f in sorted((config.RAW / "quotes_sample").glob("*-quotes*.csv")):
    m = re.match(r"(.+?)-CSE-(\d{8})-\d{8}-quotes", f.name)
    if m:
        qfiles.setdefault((TICKER_TO_STOCK[m.group(1)], m.group(2)), []).append(f)

valid_rows = []
for (stock, day), files in qfiles.items():
    raw = pd.concat(
        [
            pd.read_csv(f, dtype={"Trade Cond": str, "Venue": str, "Exch Time": str})
            for f in files
        ],
        ignore_index=True,
    )
    quotes = rebuild_quotes(raw).drop_duplicates("ts", keep="last").sort_values("ts")
    tr = pd.read_parquet(config.PROCESSED / f"signed_{stock}.parquet")
    tr = tr[
        (tr["date"] == pd.Timestamp(day))
        & (tr["ts"] >= quotes["ts"].min())
        & (tr["ts"] <= quotes["ts"].max())
    ].sort_values("ts")
    if tr.empty:
        print(f"{stock} {day}: no signed trades for this day (not loaded), skipped")
        continue
    m = pd.merge_asof(
        tr,
        quotes.rename(columns={"bid": "bid_pv", "ask": "ask_pv"}),
        on="ts",
        allow_exact_matches=False,
    )
    m["mid_pv"] = (m["bid_pv"] + m["ask_pv"]) / 2
    side_pv = np.select(
        [m["price"] > m["mid_pv"], m["price"] < m["mid_pv"]], [1, -1], 0
    )
    cont = quotes[
        (quotes["ts"].dt.strftime("%H:%M:%S") > config.CONT_START)
        & (quotes["ts"].dt.strftime("%H:%M:%S") < config.CONT_END)
    ].copy()
    cont = cont[cont["ask"] > cont["bid"]]
    dt = cont["ts"].shift(-1).sub(cont["ts"]).dt.total_seconds()
    spr = (cont["ask"] - cont["bid"]) / ((cont["ask"] + cont["bid"]) / 2) * 1e4
    valid_rows.append(
        {
            "stock": stock,
            "date": day,
            "span": f"{quotes['ts'].min():%H:%M}-{quotes['ts'].max():%H:%M}",
            "files": len(files),
            "quote_updates": len(quotes),
            "trades": len(m),
            "pct_attached_quote_equals_prevailing": round(
                100 * ((m["bid"] == m["bid_pv"]) & (m["ask"] == m["ask_pv"])).mean(), 1
            ),
            "pct_outside_attached": quote_fit(m["price"], m["bid"], m["ask"])[
                "pct_outside"
            ],
            "pct_outside_prevailing": quote_fit(m["price"], m["bid_pv"], m["ask_pv"])[
                "pct_outside"
            ],
            "pct_sign_agreement": round(
                100 * (m["side"][side_pv != 0] == side_pv[side_pv != 0]).mean(), 1
            ),
            "spread_bps_time_weighted": round(np.average(spr[:-1], weights=dt[:-1]), 2),
            "spread_bps_trade_weighted": round(
                ((m["ask"] - m["bid"]) / m["mid"] * 1e4).mean(), 2
            ),
        }
    )
valid = pd.DataFrame(valid_rows)
valid.to_csv(config.TABLES / "02_quote_validation.csv", index=False)
print("Validation against the full quote streams:\n", valid.to_string(index=False))
