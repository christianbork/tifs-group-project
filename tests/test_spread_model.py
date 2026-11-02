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
