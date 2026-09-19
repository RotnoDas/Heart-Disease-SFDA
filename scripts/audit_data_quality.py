from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "uci"
META_DIR = ROOT / "data" / "metadata"

META_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va_long_beach": "processed.va.data",
}


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


# Zero is suspicious for these variables.
# NOTE:
# We are NOT converting them to NaN here.
# This stage is audit-only.
SUSPICIOUS_ZERO_FEATURES = [
    "age",
    "cp",
    "trestbps",
    "chol",
    "thalach",
    "slope",
    "thal",
]


def load_domain(filename):
    path = RAW_DIR / filename

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values="?"
    )

    return df


def main():

    domain_rows = []
    feature_rows = []

    print("\n" + "=" * 90)
    print("HEART DISEASE MULTI-DOMAIN DATA QUALITY AUDIT")
    print("=" * 90)

    for domain, filename in FILES.items():

        df = load_domain(filename)

        # Binary target:
        # 0 -> no disease
        # 1-4 -> disease present
        binary_target = (df["num"] > 0).astype(int)

        n = len(df)
        positive_n = int(binary_target.sum())
        negative_n = int(n - positive_n)
        positive_pct = 100 * positive_n / n

        domain_rows.append(
            {
                "domain": domain,
                "n": n,
                "positive_n": positive_n,
                "negative_n": negative_n,
                "positive_pct": positive_pct,
            }
        )

        print("\n" + "-" * 90)
        print(f"DOMAIN: {domain.upper()}")
        print("-" * 90)

        print(f"N: {n}")
        print(f"Positive: {positive_n}")
        print(f"Negative: {negative_n}")
        print(f"Positive rate: {positive_pct:.2f}%")

        print("\nFEATURE QUALITY")
        print("-" * 90)

        for feature in COLUMNS[:-1]:

            missing_n = int(df[feature].isna().sum())
            missing_pct = 100 * missing_n / n

            zero_n = int((df[feature] == 0).sum())
            zero_pct = 100 * zero_n / n

            valid = df[feature].dropna()

            minimum = valid.min() if len(valid) else np.nan
            maximum = valid.max() if len(valid) else np.nan
            unique = valid.nunique()

            feature_rows.append(
                {
                    "domain": domain,
                    "feature": feature,
                    "missing_n": missing_n,
                    "missing_pct": missing_pct,
                    "zero_n": zero_n,
                    "zero_pct": zero_pct,
                    "min": minimum,
                    "max": maximum,
                    "unique_values": unique,
                }
            )

            zero_flag = ""

            if (
                feature in SUSPICIOUS_ZERO_FEATURES
                and zero_n > 0
            ):
                zero_flag = " <-- CHECK ZERO VALUES"

            print(
                f"{feature:10s} "
                f"| missing={missing_n:3d} "
                f"({missing_pct:6.2f}%) "
                f"| zero={zero_n:3d} "
                f"({zero_pct:6.2f}%) "
                f"| min={minimum} "
                f"| max={maximum}"
                f"{zero_flag}"
            )

    domain_summary = pd.DataFrame(domain_rows)

    feature_summary = pd.DataFrame(feature_rows)

    domain_summary.to_csv(
        META_DIR / "domain_summary.csv",
        index=False
    )

    feature_summary.to_csv(
        META_DIR / "feature_quality.csv",
        index=False
    )

    print("\n" + "=" * 90)
    print("DOMAIN SUMMARY")
    print("=" * 90)

    print(domain_summary.to_string(index=False))

    print("\nSaved:")
    print(META_DIR / "domain_summary.csv")
    print(META_DIR / "feature_quality.csv")


if __name__ == "__main__":
    main()