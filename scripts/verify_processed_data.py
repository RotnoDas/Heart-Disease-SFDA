from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"


FILES = {
    "cleveland": "cleveland.csv",
    "hungary": "hungary.csv",
    "switzerland": "switzerland.csv",
    "va_long_beach": "va_long_beach.csv",
}


FEATURES = [
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


def main():

    print("=" * 90)
    print("PROCESSED DATA VERIFICATION")
    print("=" * 90)

    for domain, filename in FILES.items():

        path = DATA_DIR / filename

        df = pd.read_csv(path)

        print("\n" + "-" * 90)
        print(domain.upper())
        print("-" * 90)

        print("Shape:", df.shape)

        print(
            "Target values:",
            df["target"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            "Total missing:",
            int(df[FEATURES].isna().sum().sum())
        )

        print(
            "Duplicate row_id:",
            int(df["row_id"].duplicated().sum())
        )

        unexpected_target = (
            set(df["target"].unique())
            - {0, 1}
        )

        if unexpected_target:
            raise ValueError(
                f"{domain}: invalid target values "
                f"{unexpected_target}"
            )

        print("Target validation: PASS")

    print("\nAll processed datasets verified.")


if __name__ == "__main__":
    main()