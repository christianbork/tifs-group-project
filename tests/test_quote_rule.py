import numpy as np
import pandas as pd

from tifs_group_project.quote_rule import fills, paired_daily_test, scale_to_is_width, summarise


def _trades():
    ts = pd.to_datetime(["2026-10-22 10:01", "2026-10-22 10:02", "2026-10-22 10:03"])
    return pd.DataFrame(
        {
            "stock": "A",
            "date": ts.normalize(),
            "ts": ts,
            "bar_ts": ts.floor("5min"),
            "price": 100.0,
            "vol": 10,
            "eff_bps": [2.0, 4.0, 6.0],
            "impact_bps": [1.0, 1.0, 8.0],
        }
    )


def test_fills_only_when_the_print_reaches_the_quote():
    q = pd.DataFrame({"stock": ["A"], "bar_ts": [pd.Timestamp("2026-10-22 10:00")], "wide": [5.0], "tight": [2.0]})
    f_wide, f_tight = fills(_trades(), q, "wide", 1.0), fills(_trades(), q, "tight", 1.0)
    assert len(f_wide) == 1 and len(f_tight) == 3
    assert np.allclose(f_wide["pnl_bps"], 5.0 - 8.0)  # filled only by the toxic trade
    assert np.allclose(f_tight["pnl_bps"], [1.0, 1.0, -6.0])
    s = summarise(f_tight, q, "tight")
    assert s["fills"] == 3 and np.isclose(s["markout_pnl_bps"], -4 / 3)


def test_scale_and_paired_test():
    assert np.isclose(scale_to_is_width(pd.Series([1.0, 3.0]), pd.Series([4.0, 4.0])), 2.0)
    q = pd.DataFrame({"stock": ["A"], "bar_ts": [pd.Timestamp("2026-10-22 10:00")], "wide": [5.0], "tight": [2.0]})
    keys = pd.MultiIndex.from_tuples([("A", pd.Timestamp("2026-10-22"))], names=["stock", "date"])
    out = paired_daily_test(fills(_trades(), q, "wide"), fills(_trades(), q, "tight"), keys)
    assert out["n_stock_days"] == 1 and out["mean_diff_dkk"] > 0  # wide loses 3 bps once, tight 4 bps in total
