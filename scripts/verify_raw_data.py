from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "uci"

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


def inspect_dataset(domain_name, filename):
    path = DATA_DIR / filename

    print("=" * 70)
    print(f"DOMAIN: {domain_name}")
    print(f"FILE:   {path}")

    if not path.exists():
        print("ERROR: File not found.")
        return

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values="?"
    )

    print(f"Rows:    {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nTarget values:")
    print(df["num"].value_counts(dropna=False).sort_index())

    print("\nMissing values:")
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False))

    print("\nFirst 3 rows:")
    print(df.head(3))

    print()


def main():
    print("\nRAW DATASET VERIFICATION\n")

    for domain, filename in FILES.items():
        inspect_dataset(domain, filename)


if __name__ == "__main__":
    main()