import numpy as np
import pandas as pd

from tifs_group_project.impact import cum_irf, order_events


def test_cum_irf_no_dynamics():
    # r_t = 2 x_t only: impact is 2 bps at every horizon
    p = 3
    v = {
        "p": p,
        "b": np.r_[2.0, np.zeros(p)],
        "a": np.zeros(p),
        "d": np.zeros(p),
        "g": np.zeros(p),
    }
    assert np.allclose(cum_irf(v, 5), 2.0)


def test_cum_irf_lagged_effect():
    # r_t = 1 x_t + 0.5 x_{t-1}: cumulative 1 at t=0, 1.5 afterwards
    p = 2
    v = {
        "p": p,
        "b": np.array([1.0, 0.5, 0.0]),
        "a": np.zeros(p),
        "d": np.zeros(p),
        "g": np.zeros(p),
    }
    assert np.allclose(cum_irf(v, 4), [1.0, 1.5, 1.5, 1.5, 1.5])


def test_order_events_return_stops_at_day_end():
    ts = pd.to_datetime(
        ["2026-10-01 10:00", "2026-10-01 10:00", "2026-10-01 11:00", "2026-10-02 10:00"]
    )
    d = pd.DataFrame(
        {
            "ts": ts,
            "date": ts.normalize(),
            "order_id": [0, 0, 1, 2],
            "fill_pos": [0, 1, 0, 0],
            "mid": [100.0, 100.5, 101.0, 102.0],
            "side": [1, 1, -1, 1],
            "vol": [5, 5, 3, 1],
        }
    )
    ev = order_events(d)
    assert (
        len(ev) == 3 and ev.loc[0, "vol"] == 10 and ev.loc[0, "mid"] == 100.0
    )  # first fill's mid
    assert np.isclose(ev.loc[0, "r_bps"], np.log(101 / 100) * 1e4)
    assert np.isnan(ev.loc[1, "r_bps"]) and np.isnan(
        ev.loc[2, "r_bps"]
    )  # last order of each day
