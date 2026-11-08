import numpy as np
import pandas as pd

from tifs_group_project.quoting import fills, summarise


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
    assert f.loc[3, "h_bps"] == 0.5  # floored at half a tick
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
