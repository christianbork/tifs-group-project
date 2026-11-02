import numpy as np
import pandas as pd

from tifs_group_project.volatility import seasonal_variance


def _day(date, us_min, sample, rng):
    m = np.arange(475)
    sd = np.where(
        (m >= us_min) & (m < us_min + 10), 5.0, 1.0
    )  # vol spike for 10 min after the US open
    return pd.DataFrame(
        {
            "date": pd.Timestamp(date),
            "minute_of_day": m,
            "min_since_us_open": m - us_min,
            "sample": sample,
            "ret_bps": sd * rng.choice([-1.0, 1.0], len(m)),
        }
    )


def test_us_bump_moves_with_the_us_open():
    rng = np.random.default_rng(1)
    m1 = pd.concat(
        [
            _day("2026-10-20", 390, "IS", rng),
            _day("2026-10-21", 390, "IS", rng),
            _day("2026-10-27", 330, "OOS", rng),
        ],
        ignore_index=True,
    )
    e = seasonal_variance(m1, smooth=1)
    is_day, oos_day = (
        e[m1["date"] == "2026-10-20"].to_numpy(),
        e[m1["date"] == "2026-10-27"].to_numpy(),
    )
    assert np.allclose(is_day[390:400], 25) and np.allclose(
        is_day[300:390], 1
    )  # in-sample: exact v(m)
    assert np.allclose(oos_day[330:340], 25)  # bump re-applied at 14:30
    assert np.allclose(oos_day[390:400], 1)  # and gone from 15:30
