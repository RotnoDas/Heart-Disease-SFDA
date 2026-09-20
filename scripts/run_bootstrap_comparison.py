from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from heart_sfda.evaluation.bootstrap import (
    paired_bootstrap_comparison,
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
    / "statistics"
)


METHODS = [
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
        df["target"].unique()
    ):

        target_df = (
            df[
                df["target"]
                == target
            ]
            .copy()
        )

        y = target_df[
            "y_true"
        ].to_numpy()

        source_probability = (
            target_df[
                "source_only"
            ]
            .to_numpy()
        )

        for method in METHODS:

            result = (
                paired_bootstrap_comparison(
                    y=y,
                    source_probability=
                        source_probability,
                    adapted_probability=
                        target_df[
                            method
                        ].to_numpy(),
                    iterations=5000,
                    seed=42,
                )
            )

            for metric, values in (
                result.items()
            ):

                rows.append(
                    {
                        "target":
                            target,

                        "method":
                            method,

                        "metric":
                            metric,

                        "observed":
                            values[
                                "observed"
                            ],

                        "ci_lower":
                            values[
                                "ci_lower"
                            ],

                        "ci_upper":
                            values[
                                "ci_upper"
                            ],

                        "ci_contains_zero":
                            (
                                values[
                                    "ci_lower"
                                ]
                                <= 0
                                <=
                                values[
                                    "ci_upper"
                                ]
                            ),
                    }
                )

    result_df = pd.DataFrame(
        rows
    )

    output_path = (
        OUTPUT_DIR
        / "paired_bootstrap_sfda.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "PAIRED STRATIFIED BOOTSTRAP"
    )

    print(
        "=" * 110
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        output_path
    )


if __name__ == "__main__":
    main()