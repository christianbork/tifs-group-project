import numpy as np
import pandas as pd

from tifs_group_project.bars import aggregate, minute_bars


def _trades():
    day = pd.Timestamp("2026-10-01")
    ts = day + pd.to_timedelta(["09:00:30", "09:01:30"])
    # quote 100.0/100.2 (10 bps half-spread approx), then 100.0/100.1 (one tick of 0.1)
    return day, pd.DataFrame(
        {
            "ts": ts,
            "price": [100.2, 100.0],
            "vol": [10, 20],
            "bid": [100.0, 100.0],
            "ask": [100.2, 100.1],
            "mid": [100.1, 100.05],
            "side": [1, -1],
            "order_id": [0, 1],
        }
    )


def test_quote_carried_forward_and_time_weighted():
    day, t = _trades()
    m = minute_bars(t, day, tick=0.1)
    hs1, hs2 = 0.1 / 100.1 * 1e4, 0.05 / 100.05 * 1e4
    # minute 09:00: no quote for the first 30 s, then hs1 for 30 s
    assert m.loc[0, "q_sec"] == 30 and np.isclose(m.loc[0, "hs_x_sec"], 30 * hs1)
    # minute 09:01: hs1 carried forward 30 s, then hs2 (one tick) for 30 s
    assert m.loc[1, "q_sec"] == 60 and np.isclose(
        m.loc[1, "hs_x_sec"], 30 * hs1 + 30 * hs2
    )
    assert m.loc[1, "one_x_sec"] == 30
    # later minutes keep the last quote, with no trades
    assert m.loc[5, "trades"] == 0 and np.isclose(m.loc[5, "hs_x_sec"], 60 * hs2)


def test_aggregate_5min():
    day, t = _trades()
    b = aggregate(minute_bars(t, day, tick=0.1).assign(stock="Novo", date=day), 5)
    assert len(b) == 95 and b["trades"].sum() == 2
    assert np.isclose(b.loc[0, "ofi"], (10 - 20) / 30)
    assert np.isclose(b.loc[0, "pct_time_1tick"], 100 * (270 - 60) / 270)
