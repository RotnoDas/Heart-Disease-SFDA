from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "uci"


COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


RAW_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va_long_beach": "processed.va.data",
}


def load_raw_domain(domain: str) -> pd.DataFrame:
    if domain not in RAW_FILES:
        raise ValueError(
            f"Unknown domain: {domain}. "
            f"Expected one of {list(RAW_FILES)}"
        )

    path = RAW_DIR / RAW_FILES[domain]

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values=["?"],
    )

    # Force numeric conversion for the original UCI variables.
    for column in COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Stable identifier for reproducibility.
    df.insert(
        0,
        "row_id",
        [f"{domain}_{i:04d}" for i in range(len(df))],
    )

    df["domain"] = domain

    return df


def load_all_raw() -> dict[str, pd.DataFrame]:
    return {
        domain: load_raw_domain(domain)
        for domain in RAW_FILES
    }