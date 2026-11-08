import numpy as np
import pandas as pd

from tifs_group_project.spread_model import dm_test, tod_bucket


def test_tod_bucket_us_override_moves_with_us_open():
    minute = pd.Series([0, 20, 330, 380, 400, 470])
    # in-sample day (US open 15:30 = minute 390) vs OOS day (14:30 = minute 330)
    is_day = tod_bucket(minute, minute - 390).tolist()
    oos_day = tod_bucket(minute, minute - 330).tolist()
    assert is_day == ["09:00", "09:15", "14:00-16:30", "US-30", "US+0", "16:30"]
    assert oos_day == ["09:00", "09:15", "US+0", "US+30", "14:00-16:30", "16:30"]


def test_dm_test_sign_and_null():
    rng = np.random.default_rng(0)
    e = pd.Series(rng.standard_normal(500) ** 2)
    assert np.isclose(dm_test(e, e, 5)["dm_stat"], 0) or np.isnan(
        dm_test(e, e, 5)["dm_stat"]
    )
    better = dm_test(e * 0.5, e, 5)
    assert better["dm_stat"] < -3 and better["p_value"] < 0.01


def test_coarse_bucket_keyed_to_us_open():
    from tifs_group_project.spread_model import coarse_bucket

    m = pd.Series([0, 20, 45, 100, 385, 395, 430, 465])
    assert coarse_bucket(m, m - 390).tolist() == [
        "09:00",
        "09:15",
        "09:30",
        "10:00-12:00",
        "12:00-US",
        "US+0-30",
        "afternoon",
        "last15",
    ]
    assert coarse_bucket(m, m - 330).tolist()[4:] == [
        "afternoon",
        "afternoon",
        "afternoon",
        "last15",
    ]


def _panel(seed=0, n_days=6, n_bars=40):
    rng = np.random.default_rng(seed)
    rows = []
    for s in ["A", "B", "C"]:
        for d in pd.date_range("2026-10-01", periods=n_days, freq="D"):
            for k in range(n_bars):
                rows.append({"stock": s, "date": d, "ts": d + pd.Timedelta(minutes=540 + 5 * k), "x": rng.standard_normal()})
    p = pd.DataFrame(rows)
    p["y"] = 1 + 0.5 * p["x"] + rng.standard_normal(len(p))
    p["tod"] = "12:00"
    return p


def test_lag_within_day_never_crosses_days():
    from tifs_group_project.spread_model import lag_within_day

    p = _panel(n_days=2, n_bars=3).sort_values(["stock", "ts"])
    lag = lag_within_day(p, "x")
    first = p.groupby(["stock", "date"]).cumcount() == 0
    assert lag[first].isna().all() and lag[~first].notna().all()
    assert np.allclose(lag[~first], p.groupby(["stock", "date"])["x"].shift(1)[~first])


def test_fit_ols_cov_choices_same_coefficients():
    from tifs_group_project.spread_model import fit_ols

    p = _panel()
    fits = {c: fit_ols(p, "y", ["x"], 5, cov=c)[0] for c in ["nw", "dk", "cluster_stockday", "cluster_day"]}
    b = [r.params["x"] for r in fits.values()]
    assert np.allclose(b, b[0]) and abs(b[0] - 0.5) < 0.1
    assert all(r.bse["x"] > 0 for r in fits.values())


def test_wald_matches_t_squared_for_one_term():
    from tifs_group_project.spread_model import fit_ols, wald

    res, _ = fit_ols(_panel(), "y", ["x"], 5)
    assert np.isclose(wald(res, ["x"])["chi2"], res.tvalues["x"] ** 2)


def test_long_run_effect_and_residual_acf():
    from tifs_group_project.spread_model import fit_ols, long_run, residual_acf

    p = _panel()
    p["lag_quoted_hs"] = p.groupby("stock")["y"].shift(1)
    res, d = fit_ols(p, "y", ["x", "lag_quoted_hs"], 5)
    lr = long_run(res, ["x"]).iloc[0]
    assert np.isclose(lr["long_run"], res.params["x"] / (1 - res.params["lag_quoted_hs"]))
    acf = residual_acf(res, d, lags=(1, 2))
    assert len(acf) == 3 and acf["acf_1"].abs().max() < 0.2  # iid errors
