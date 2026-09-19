from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
META_DIR = ROOT / "data" / "metadata"


FILES = {
    "cleveland": "cleveland.csv",
    "hungary": "hungary.csv",
    "switzerland": "switzerland.csv",
    "va_long_beach": "va_long_beach.csv",
}


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


FULL13 = [
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
]


def feature_set_summary(
    df,
    domain,
    feature_set_name,
    features,
):
    missing_cells = int(
        df[features]
        .isna()
        .sum()
        .sum()
    )

    total_cells = (
        len(df) * len(features)
    )

    missing_pct = (
        100 * missing_cells / total_cells
    )

    complete_rows = int(
        df[features]
        .notna()
        .all(axis=1)
        .sum()
    )

    complete_pct = (
        100 * complete_rows / len(df)
    )

    return {
        "domain": domain,
        "feature_set": feature_set_name,
        "n_features": len(features),
        "missing_cells": missing_cells,
        "total_cells": total_cells,
        "missing_pct": missing_pct,
        "complete_rows": complete_rows,
        "complete_rows_pct": complete_pct,
    }


def main():

    rows = []

    for domain, filename in FILES.items():

        df = pd.read_csv(
            DATA_DIR / filename
        )

        rows.append(
            feature_set_summary(
                df,
                domain,
                "CORE8",
                CORE8,
            )
        )

        rows.append(
            feature_set_summary(
                df,
                domain,
                "FULL13",
                FULL13,
            )
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False
        )
    )

    output_path = (
        META_DIR
        / "feature_set_missingness.csv"
    )

    result.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSaved:",
        output_path
    )


if __name__ == "__main__":
    main()