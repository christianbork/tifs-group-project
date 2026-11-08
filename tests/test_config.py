import numpy as np
import pandas as pd

from tifs_group_project import config
from tifs_group_project.io import _stitch


def test_sample_label_three_samples():
    dates = pd.Series(
        pd.to_datetime(
            [
                "2026-10-01",
                "2026-10-21",
                "2026-10-22",
                "2026-10-30",
                "2026-11-02",
                "2026-11-06",
            ]
        )
    )
    assert config.sample_label(dates).tolist() == [
        "IS",
        "IS",
        "OOS",
        "OOS",
        "FWD",
        "FWD",
    ]
    assert config.sample_label(pd.Series(["2026-10-21", "2026-11-02"])).tolist() == [
        "IS",
        "FWD",
    ]


def test_november_event_days_are_in_the_forward_window():
    for (_, day), _label in config.EVENT_DAYS.items():
        assert config.sample_label(pd.Series([day]))[0] in ("IS", "OOS", "FWD")
    assert ("Novo", "2026-11-04") in config.EVENT_DAYS


def test_stitch_handles_missing_trade_conditions():
    ts = pd.to_datetime(
        [
            "2026-11-04 09:00:00.001",
            "2026-11-04 09:00:00.002",
            "2026-11-04 09:00:00.003",
        ]
    )
    cols = ["ts", "Price", "Vol", "Bid", "Ask", "CVol", "Trade Cond"]
    part1 = pd.DataFrame(
        [[ts[0], 1.0, 10, 0.9, 1.1, 10, np.nan], [ts[1], 1.0, 20, 0.9, 1.1, 30, "MPM"]],
        columns=cols,
    )
    part2 = pd.DataFrame(
        [[ts[1], 1.0, 20, 0.9, 1.1, 30, "MPM"], [ts[2], 1.0, 5, 0.9, 1.1, 35, np.nan]],
        columns=cols,
    )
    out, dropped = _stitch([part1, part2])
    assert dropped == 1 and len(out) == 3
