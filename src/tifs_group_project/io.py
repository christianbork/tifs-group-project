"""Load and clean FactSet trade exports.

Each export is one stock-day, newest row first. Busy days are split into
`-part1`, `-part2`, ... files that overlap slightly at the boundary.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

COLS = {
    "Price": "price",
    "Vol": "vol",
    "Bid": "bid",
    "Bid Size": "bid_size",
    "Ask": "ask",
    "Ask Size": "ask_size",
    "CVol": "cvol",
    "VWAP": "vwap_cum",
    "Trade Cond": "cond",
    "Venue": "venue",
}

OFF_BOOK_FLAGS = ("XLT", "CPX", "SSL", "VWAP", "RES", "VLT")


def list_files(stock: str) -> dict[str, list[Path]]:
    """Map trading day (YYYYMMDD) -> sorted list of export files (parts in order)."""
    out: dict[str, list[Path]] = {}
    for f in sorted((config.RAW / stock).glob("*-trades*.csv")):
        day = re.search(r"-(\d{8})-\d{8}-", f.name).group(1)
        out.setdefault(day, []).append(f)
    for day, fs in out.items():
        out[day] = sorted(fs, key=lambda p: int(re.search(r"part(\d+)", p.name).group(1)) if "part" in p.name else 0)
    return out


def _read_export(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Trade Cond": str, "Venue": str, "Exch Time": str})
    df = df.iloc[::-1].reset_index(drop=True)  # oldest first
    df["ts"] = pd.to_datetime(df["Date"] + " " + df["Time (ms)"], format="%Y/%m/%d %H:%M:%S.%f")
    df["file"] = path.name
    return df


def _stitch(parts: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Join consecutive parts, dropping rows of part k that duplicate part k-1 inside the overlap window.

    Duplicates are only removed inside the overlap, because files also contain
    genuinely repeated rows (same ms, price and size) that must be kept.
    Missing trade conditions are filled with '' before the row key is built (pandas 3
    keeps NaN through astype(str), which broke the join on the November part files).
    """
    key = ["ts", "Price", "Vol", "Bid", "Ask", "CVol", "Trade Cond"]
    out, dropped = parts[0], 0
    for nxt in parts[1:]:
        lo, hi = nxt["ts"].min(), out["ts"].max()
        prev_ov = out[out["ts"] >= lo]
        in_ov = nxt["ts"] <= hi
        # multiset match: allow as many drops as copies in the previous part
        prev_counts = prev_ov[key].fillna("").astype(str).agg("|".join, axis=1).value_counts().to_dict()
        keep = []
        for is_ov, k in zip(in_ov, nxt[key].fillna("").astype(str).agg("|".join, axis=1)):
            if is_ov and prev_counts.get(k, 0) > 0:
                prev_counts[k] -= 1
                keep.append(False)
            else:
                keep.append(True)
        dropped += len(keep) - sum(keep)
        out = pd.concat([out, nxt[keep]], ignore_index=True)
    out = out.sort_values("ts", kind="stable").reset_index(drop=True)
    return out, dropped


def classify(cond: pd.Series, venue: pd.Series) -> pd.Series:
    """Label each row: indicative, close, auction, dark, off_book, other_venue, lit."""
    c = cond.fillna("")
    v = venue.fillna("")
    kind = np.select(
        [
            c.str.contains("IND"),
            c.eq("CLS"),
            c.str.contains("AUC"),
            c.str.contains("MPM") | v.eq("DCSE"),
            c.str.contains("|".join(OFF_BOOK_FLAGS)) | c.eq("X"),
            v.ne("XCSE"),
            c.isin(["", "ROU"]),
        ],
        ["indicative", "close", "auction", "dark", "off_book", "other_venue", "lit"],
        default="other",
    )
    return pd.Series(kind, index=cond.index)


def load_day(stock: str, files: list[Path]) -> tuple[pd.DataFrame, dict]:
    parts = [_read_export(f) for f in files]
    raw_rows = [len(p) for p in parts]
    df, dropped = _stitch(parts) if len(parts) > 1 else (parts[0], 0)

    df = df.rename(columns=COLS)
    df["stock"] = stock
    df["date"] = df["ts"].dt.normalize()
    df["kind"] = classify(df["cond"], df["venue"])
    t = df["ts"].dt.strftime("%H:%M:%S")
    df["continuous"] = (df["kind"] == "lit") & (t > config.CONT_START) & (t < config.CONT_END)
    df["quote_ok"] = (df["bid"] > 0) & (df["ask"] > df["bid"])
    df["mid"] = np.where(df["quote_ok"], (df["bid"] + df["ask"]) / 2, np.nan)
    df["sample"] = config.sample_label(df["date"])

    first, last = df["ts"].min().strftime("%H:%M:%S.%f")[:12], df["ts"].max().strftime("%H:%M:%S.%f")[:12]
    lit = df[df["continuous"] & df["quote_ok"]]
    qa = {
        "stock": stock,
        "date": df["date"].iloc[0].date(),
        "sample": df["sample"].iloc[0],
        "files": len(files),
        "raw_rows": sum(raw_rows),
        "overlap_dupes_dropped": dropped,
        "first_ts": first,
        "last_ts": last,
        "max_file_rows": max(raw_rows),
        "truncated_flag": (first > config.FIRST_EXPECTED) or (max(raw_rows) >= config.EXPORT_CAP) or (last < config.LAST_EXPECTED),
        **{f"n_{k}": int(v) for k, v in df["kind"].value_counts().items()},
        "n_continuous": int(df["continuous"].sum()),
        "pct_bad_quote": round(100 * (1 - df.loc[df["continuous"], "quote_ok"].mean()), 2),
        "pct_outside_quote": round(100 * ((lit["price"] > lit["ask"]) | (lit["price"] < lit["bid"])).mean(), 2),
        "median_spread_bps": round(((lit["ask"] - lit["bid"]) / lit["mid"] * 1e4).median(), 2),
        "dkk_turnover_m": round((lit["price"] * lit["vol"]).sum() / 1e6, 1),
        "event": config.EVENT_DAYS.get((stock, str(df["date"].iloc[0].date())), ""),
    }
    keep = ["ts", "date", "stock", "sample", "kind", "continuous", "price", "vol", "bid", "bid_size",
            "ask", "ask_size", "mid", "quote_ok", "cvol", "cond", "venue", "file"]
    return df.loc[df["kind"] != "indicative", keep].reset_index(drop=True), qa


def load_stock(stock: str) -> tuple[pd.DataFrame, list[dict]]:
    frames, qas = [], []
    for day, files in list_files(stock).items():
        d, qa = load_day(stock, files)
        frames.append(d)
        qas.append(qa)
    return pd.concat(frames, ignore_index=True), qas


def read_clean(stock: str | None = None) -> pd.DataFrame:
    """Read cleaned trades for one stock, or all stocks if None."""
    stocks = [stock] if stock else list(config.STOCKS)
    return pd.concat([pd.read_parquet(config.PROCESSED / f"trades_{s}.parquet") for s in stocks], ignore_index=True)
