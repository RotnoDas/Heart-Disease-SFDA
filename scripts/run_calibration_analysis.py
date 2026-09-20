from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from heart_sfda.evaluation.calibration import (
    compute_calibration_metrics,
)


PREDICTION_FILE = (
    ROOT
    / "results"
    / "statistics"
    / "patient_level_sfda_predictions.csv"
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "calibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


METHODS = [
    "source_only",
    "pseudo_label_sfda",
    "entropy_sfda",
    "reliability_gated_sfda",
]


def main():

    df = pd.read_csv(
        PREDICTION_FILE
    )

    rows = []


    for target in (
        df["target"]
        .unique()
    ):

        target_df = (
            df[
                df["target"]
                == target
            ]
            .copy()
        )

        y_true = (
            target_df[
                "y_true"
            ]
            .to_numpy()
        )


        for method in METHODS:

            probability = (
                target_df[
                    method
                ]
                .to_numpy()
            )


            metrics = (
                compute_calibration_metrics(
                    y_true,
                    probability,
                )
            )


            rows.append(
                {
                    "target":
                        target,

                    "method":
                        method,

                    "n":
                        len(
                            target_df
                        ),

                    "positive_n":
                        int(
                            y_true.sum()
                        ),

                    "negative_n":
                        int(
                            len(y_true)
                            - y_true.sum()
                        ),

                    **metrics,
                }
            )


    result_df = pd.DataFrame(
        rows
    )


    output_path = (
        OUTPUT_DIR
        / "calibration_metrics.csv"
    )


    result_df.to_csv(
        output_path,
        index=False,
    )


    print(
        "\n"
        + "=" * 120
    )

    print(
        "CALIBRATION ANALYSIS"
    )

    print(
        "=" * 120
    )


    display_columns = [
        "target",
        "method",
        "brier",
        "log_loss",
        "ece_5",
        "ece_10",
        "calibration_intercept",
        "calibration_slope",
        "calibration_fit_success",
    ]


    print(
        result_df[
            display_columns
        ].to_string(
            index=False
        )
    )


    print(
        "\nSaved:",
        output_path
    )


if __name__ == "__main__":
    main()