"""Project-wide settings: paths, stocks, sample split and session times."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

# folder name -> FactSet ticker
STOCKS = {
    "Novo": "NOVO.B",
    "DSV": "DSV",
    "Vestas": "VWS",
    "Pandora": "PNDORA",
    "GN": "GN",
    "ISS": "ISS",
}

# In-sample (estimation) vs out-of-sample (testing) split, plus the forward test
IS_END = "2026-10-21"  # last in-sample day (inclusive)
OOS_END = "2026-10-30"  # last out-of-sample day (inclusive); 22-30 Oct was evaluated three times (07, 08)
FWD_START = "2026-11-02"  # forward test: 2-6 Nov 2026, exported after the models were final, evaluated ONCE (script 09)


def sample_label(dates):
    """IS / OOS / FWD label for a Series of dates (Timestamps or ISO strings)."""
    import numpy as np
    import pandas as pd

    d = pd.to_datetime(pd.Series(dates))
    return np.select(
        [d <= pd.Timestamp(IS_END), d <= pd.Timestamp(OOS_END)],
        ["IS", "OOS"],
        default="FWD",
    )


# Copenhagen local time
OPEN_AUCTION = "09:00:00"
CONT_START = "09:00:00"
CONT_END = "17:00:00"
LAST_CONT_TRADE = (
    "16:55:00"  # continuous trading stops here; 16:55-17:00 is the closing-auction call
)
FIRST_EXPECTED = "08:45:01"  # files should start by 08:45
LAST_EXPECTED = "17:20:00"  # official close print (CLS)
EXPORT_CAP = 25_000  # FactSet export row cap; truncation drops the morning

EVENT_DAYS = {
    ("DSV", "2026-10-21"): "Q3 earnings",
    ("ISS", "2026-10-29"): "news (check)",
    # forward-test window (2-6 Nov): flagged from volume / price moves in the exports, confirm the announcements
    ("Novo", "2026-11-04"): "Q3 report (check)",
    ("Pandora", "2026-11-04"): "Q3 report (check)",
    ("GN", "2026-11-05"): "Q3 report (check)",
}

CONSTRAINED = ["Novo", "DSV", "ISS"]  # spread at 1 tick >50% of the time
UNCONSTRAINED = ["Vestas", "Pandora", "GN"]

# US cash open (09:30 New York) in Copenhagen local time; from 2 Nov both sides are on
# standard time again, so the default 15:30 applies to the forward-test days
US_OPEN_CPH = {
    "default": "15:30",
    "2026-10-26": "14:30",
    "2026-10-27": "14:30",
    "2026-10-28": "14:30",
    "2026-10-29": "14:30",
    "2026-10-30": "14:30",
}

# Price-impact VAR (script 05): per stock-day for liquid stocks, rolling window for thin ones
VAR_DAILY = ["Novo", "DSV", "Vestas"]
VAR_ROLLING = ["Pandora", "GN", "ISS"]
ROLL_DAYS = 5  # window ends on day t; its estimate predicts day t+1
