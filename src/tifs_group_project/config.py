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

# In-sample (estimation) vs out-of-sample (testing) split
IS_END = "2026-10-21"  # last in-sample day (inclusive)

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

EVENT_DAYS = {("DSV", "2026-10-21"): "Q3 earnings"}
