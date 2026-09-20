from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from heart_sfda.data.preprocessing import CORE8

from heart_sfda.domain_shift.statistics import (
    categorical_js_table,
    ks_shift_table,
    missingness_shift_table,
)

from heart_sfda.domain_shift.domain_classifier import (
    domain_classifier_auc,
)


DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "domain_shift"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGETS = {
    "hungary":
        "hungary.csv",

    "switzerland":
        "switzerland.csv",

    "va_long_beach":
        "va_long_beach.csv",
}


def load_dataset(filename):

    return pd.read_csv(
        DATA_DIR / filename
    )


def main():

    source_name = "cleveland"

    source_df = load_dataset(
        "cleveland.csv"
    )

    all_ks = []
    all_js = []
    all_missingness = []
    all_classifier = []


    print("\n" + "=" * 100)

    print(
        "DOMAIN SHIFT CHARACTERIZATION"
    )

    print("=" * 100)

    print(
        "\nSource domain: Cleveland"
    )

    print(
        "Feature set: CORE8"
    )


    for target_name, filename in (
        TARGETS.items()
    ):

        target_df = load_dataset(
            filename
        )

        print(
            "\n"
            + "-" * 100
        )

        print(
            f"CLEVELAND vs "
            f"{target_name.upper()}"
        )

        print(
            "-" * 100
        )


        # -----------------------------------------------
        # Continuous shift
        # -----------------------------------------------

        ks_df = ks_shift_table(
            source_df,
            target_df,
            source_name,
            target_name,
        )

        all_ks.append(
            ks_df
        )

        print(
            "\nContinuous-feature shift:"
        )

        print(
            ks_df[
                [
                    "feature",
                    "ks_statistic",
                    "ks_pvalue",
                    "source_median",
                    "target_median",
                ]
            ].to_string(
                index=False
            )
        )


        # -----------------------------------------------
        # Categorical shift
        # -----------------------------------------------

        js_df = categorical_js_table(
            source_df,
            target_df,
            source_name,
            target_name,
        )

        all_js.append(
            js_df
        )

        print(
            "\nCategorical-feature "
            "Jensen-Shannon distance:"
        )

        print(
            js_df[
                [
                    "feature",
                    "js_distance",
                ]
            ].to_string(
                index=False
            )
        )


        # -----------------------------------------------
        # Missingness shift
        # -----------------------------------------------

        missing_df = (
            missingness_shift_table(
                source_df,
                target_df,
                CORE8,
                source_name,
                target_name,
            )
        )

        all_missingness.append(
            missing_df
        )

        print(
            "\nLargest missingness shifts:"
        )

        print(
            missing_df
            .sort_values(
                "absolute_difference_pct",
                ascending=False,
            )
            .head(8)[
                [
                    "feature",
                    "source_missing_pct",
                    "target_missing_pct",
                    "absolute_difference_pct",
                ]
            ]
            .to_string(
                index=False
            )
        )


        # -----------------------------------------------
        # Domain classifier
        # -----------------------------------------------

        classifier_result = (
            domain_classifier_auc(
                source_df,
                target_df,
                source_name,
                target_name,
            )
        )

        all_classifier.append(
            classifier_result
        )

        print(
            "\nDomain classifier AUROC:"
        )

        print(
            f"{classifier_result['domain_classifier_auc']:.4f}"
        )


    # --------------------------------------------------
    # Save all tables
    # --------------------------------------------------

    ks_all = pd.concat(
        all_ks,
        ignore_index=True,
    )

    js_all = pd.concat(
        all_js,
        ignore_index=True,
    )

    missing_all = pd.concat(
        all_missingness,
        ignore_index=True,
    )

    classifier_all = pd.DataFrame(
        all_classifier
    )


    ks_all.to_csv(
        RESULT_DIR
        / "continuous_ks_shift.csv",
        index=False,
    )

    js_all.to_csv(
        RESULT_DIR
        / "categorical_js_shift.csv",
        index=False,
    )

    missing_all.to_csv(
        RESULT_DIR
        / "missingness_shift.csv",
        index=False,
    )

    classifier_all.to_csv(
        RESULT_DIR
        / "domain_classifier_auc.csv",
        index=False,
    )


    print("\n" + "=" * 100)

    print(
        "DOMAIN CLASSIFIER SUMMARY"
    )

    print("=" * 100)

    print(
        classifier_all.to_string(
            index=False
        )
    )


    print(
        "\nSaved results to:"
    )

    print(
        RESULT_DIR
    )


if __name__ == "__main__":
    main()