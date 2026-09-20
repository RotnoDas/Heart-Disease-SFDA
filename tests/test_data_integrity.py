from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"


CORE8 = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
]


EXPECTED = {
    "cleveland.csv": {
        "n": 303,
        "positive": 139,
        "core8_missing": 0,
    },

    "hungary.csv": {
        "n": 294,
        "positive": 106,
        "core8_missing": 4,
    },

    "switzerland.csv": {
        "n": 123,
        "positive": 115,
        "core8_missing": 11,
    },

    "va_long_beach.csv": {
        "n": 200,
        "positive": 149,
        "core8_missing": 219,
    },
}


def test_processed_dataset_integrity():

    for filename, expected in EXPECTED.items():

        path = DATA_DIR / filename

        assert path.exists(), (
            f"Missing processed dataset: {path}"
        )

        df = pd.read_csv(path)

        assert len(df) == expected["n"]

        assert "target" in df.columns

        assert int(df["target"].sum()) == (
            expected["positive"]
        )

        assert set(
            df["target"].dropna().unique()
        ).issubset({0, 1})

        missing = int(
            df[CORE8]
            .isna()
            .sum()
            .sum()
        )

        assert missing == (
            expected["core8_missing"]
        )


def test_core8_does_not_include_target():

    assert "target" not in CORE8
    assert "num" not in CORE8