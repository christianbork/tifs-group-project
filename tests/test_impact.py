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


def _sim_orders(days, n_per_day, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for day in days:
        ts = (
            pd.Timestamp(day)
            + pd.Timedelta("09:00:00")
            + pd.to_timedelta(np.sort(rng.uniform(0, 7200, n_per_day)), unit="s")
        )
        sign = rng.choice([-1, 1], n_per_day)
        # pre-trade mid: order k moves the mid seen by order k+1
        mid = 100 * np.exp(
            np.r_[
                0, np.cumsum(2e-4 * sign + 1e-4 * rng.standard_normal(n_per_day))[:-1]
            ]
        )
        frames.append(
            pd.DataFrame({"date": ts.normalize(), "ts": ts, "mid": mid, "sign": sign})
        )
    ev = pd.concat(frames, ignore_index=True)
    ev["r_bps"] = np.log(ev.groupby("date")["mid"].shift(-1) / ev["mid"]) * 1e4
    return ev


def test_rolling_impact_uses_only_pre_bar_orders():
    from tifs_group_project.impact import rolling_impact

    ev = _sim_orders(["2026-10-01"], 400)
    tau = pd.Series([pd.Timestamp("2026-10-01 10:30")])
    base = rolling_impact(ev, tau, n_orders=150)

    # scramble everything from tau onwards: the estimate at tau must not change
    later = ev["ts"] >= tau[0]
    ev2 = ev.copy()
    ev2.loc[later, "sign"] = -ev2.loc[later, "sign"]
    ev2.loc[later, "mid"] *= 1.01
    ev2["r_bps"] = np.log(ev2.groupby("date")["mid"].shift(-1) / ev2["mid"]) * 1e4
    again = rolling_impact(ev2, tau, n_orders=150)
    assert np.isclose(
        base["impact_roll_var_bps"].iloc[0], again["impact_roll_var_bps"].iloc[0]
    )
    assert (
        base["impact_roll_var_bps"].iloc[0] > 0
    )  # buys push the mid up in the simulated data


def test_rolling_impact_window_spans_previous_day():
    from tifs_group_project.impact import rolling_impact

    ev = _sim_orders(["2026-10-01", "2026-10-02"], 200)
    taus = pd.Series(pd.to_datetime(["2026-10-01 09:30", "2026-10-02 09:05"]))
    out = rolling_impact(ev, taus, n_orders=150)
    assert np.isnan(out["impact_roll_var_bps"].iloc[0])  # too few orders yet on day 1
    assert (
        out["impact_roll_var_bps"].notna().iloc[1]
    )  # day 2 early: window reaches into day 1
    assert out["roll_window_span"].iloc[1] > pd.Timedelta("1D") - pd.Timedelta("1h")
