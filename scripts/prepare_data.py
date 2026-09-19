from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# Allow importing the src-layout package
sys.path.insert(
    0,
    str(ROOT / "src"),
)


from heart_sfda.data.loader import load_all_raw
from heart_sfda.data.cleaner import (
    PREDICTORS,
    clean_domain,
    validate_categories,
)


INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
META_DIR = ROOT / "data" / "metadata"


INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

META_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_NAMES = {
    "cleveland": "cleveland.csv",
    "hungary": "hungary.csv",
    "switzerland": "switzerland.csv",
    "va_long_beach": "va_long_beach.csv",
}


def main():

    datasets = load_all_raw()

    reports = []

    print("\n" + "=" * 80)
    print("DATA CLEANING AND HARMONIZATION")
    print("=" * 80)

    for domain, raw_df in datasets.items():

        print(f"\nProcessing: {domain}")

        cleaned_df, report = clean_domain(
            raw_df
        )

        warnings = validate_categories(
            cleaned_df
        )

        if warnings:
            print("Validation warnings:")

            for warning in warnings:
                print("  -", warning)

        else:
            print(
                "Categorical-value validation: PASS"
            )

        reports.append(report)

        # ---------------------------------------------
        # INTERIM
        #
        # Keeps original num for audit/reproducibility.
        # ---------------------------------------------

        interim_path = (
            INTERIM_DIR
            / f"{domain}_clean.csv"
        )

        cleaned_df.to_csv(
            interim_path,
            index=False,
        )


        # ---------------------------------------------
        # PROCESSED
        #
        # Remove original 'num' to avoid accidental
        # leakage. Keep only predictors + binary target
        # + identifiers.
        # ---------------------------------------------

        processed_columns = (
            ["row_id", "domain"]
            + PREDICTORS
            + ["target"]
        )

        processed_df = cleaned_df[
            processed_columns
        ].copy()

        output_path = (
            PROCESSED_DIR
            / OUTPUT_NAMES[domain]
        )

        processed_df.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Saved processed dataset: "
            f"{output_path}"
        )

    # ---------------------------------------------
    # Cleaning report
    # ---------------------------------------------

    report_df = pd.DataFrame(reports)

    report_path = (
        META_DIR
        / "cleaning_summary.csv"
    )

    report_df.to_csv(
        report_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)

    print(
        report_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved cleaning report:\n{report_path}"
    )


if __name__ == "__main__":
    main()