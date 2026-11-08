import numpy as np
import pandas as pd

from tifs_group_project.quoting import fills, orders, summarise


def _trades():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-11-02"] * 4),
            "side": [1, -1, 1, -1],
            "vol": [100, 500, 50, 10],
            "mid": [100.0, 100.0, 100.0, 100.0],
            "eff_bps": [5.0, 5.0, 1.0, 5.0],  # third trade printed inside a 2 bps quote
            "impact_5m_bps": [2.0, 8.0, 0.0, -1.0],
        }
    )


def test_fills_and_floor():
    t = _trades()
    f = fills(
        t,
        pd.Series([2.0, 2.0, 2.0, 0.1], index=t.index),
        quote_size=200,
        half_tick_bps=pd.Series([0.5] * 4),
    )
    assert f["filled"].tolist() == [True, True, False, True]
    assert f["size"].tolist() == [100.0, 200.0, 0.0, 10.0]  # capped at the quote size
    assert f.loc[3, "h_bps"] == 0.5  # floored at half a tick (one grid step)
    # realised = h - impact: 2-2, 2-8, -, 0.5-(-1)
    assert np.allclose(f.loc[f["filled"], "realised_bps"], [0.0, -6.0, 1.5])
    # market maker buys when the aggressor sells
    assert f["signed_size"].tolist() == [-100.0, 200.0, -0.0, 10.0]


def test_summary_weights_by_size():
    t = _trades()
    f = fills(
        t,
        pd.Series([2.0] * 4, index=t.index),
        quote_size=1e9,
        half_tick_bps=pd.Series([0.5] * 4),
    )
    s = summarise(t, f)
    assert s["fills"] == 3 and np.isclose(s["fill_rate_pct"], 75.0)
    assert np.isclose(s["adverse_selection_bps"], (100 * 2 + 500 * 8 + 10 * -1) / 610)
    assert np.isclose(s["pnl_dkk_per_day"], f["pnl_dkk"].sum())


def test_snap_to_tick_grid():
    t = _trades()
    # half tick 1.0 bps: 1.05 -> 1.0 (nearest), 1.6 -> 2.0, 0.2 -> 1.0 (floor)
    f = fills(t, pd.Series([1.05, 1.6, 0.2, 2.5], index=t.index), quote_size=1e9, half_tick_bps=pd.Series([1.0] * 4))
    assert f["h_bps"].tolist() == [1.0, 2.0, 1.0, 2.0]


def test_orders_collapse_fills():
    t = pd.DataFrame(
        {
            "stock": ["A"] * 3 + ["B"],
            "order_id": [0, 0, 1, 0],
            "fill_pos": [0, 1, 0, 0],
            "ts": pd.to_datetime(["2026-11-02 09:00:00.100"] * 2 + ["2026-11-02 09:00:01.000", "2026-11-02 09:00:00.500"]),
            "vol": [100, 50, 10, 7],
            "eff_bps": [1.0, 3.0, 2.0, 4.0],
            "mid": [100.0, 100.0, 100.0, 50.0],
        }
    )
    o = orders(t)
    assert len(o) == 3 and o["vol"].tolist() == [150, 7, 10]  # time order; A/0 uses the first fill's quote
    assert o.loc[0, "eff_bps"] == 1.0 and o.loc[0, "fills_in_order"] == 2


def test_scale_and_paired_daily_test():
    from tifs_group_project.quoting import paired_daily_test, scale_to_is_width

    assert np.isclose(scale_to_is_width(pd.Series([1.0, 3.0]), pd.Series([4.0, 4.0])), 2.0)
    idx = pd.MultiIndex.from_tuples([("A", "d1"), ("A", "d2"), ("B", "d1")], names=["stock", "date"])
    a = pd.Series([10.0, 12.0, 9.0], index=idx)
    b = pd.Series([8.0, 9.0], index=idx[:2])  # B-d1 had no fills under rule b -> 0
    out = paired_daily_test(a, b)
    assert out["n_stock_days"] == 3 and np.isclose(out["mean_diff_dkk"], (2 + 3 + 9) / 3)
    assert out["share_days_better"] == 1.0 and out["t"] > 0
