from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.calibration import calibration_curve


ROOT = Path(__file__).resolve().parents[1]


PREDICTION_FILE = (
    ROOT
    / "results"
    / "statistics"
    / "patient_level_sfda_predictions.csv"
)


FIGURE_DIR = (
    ROOT
    / "figures"
    / "calibration"
)

FIGURE_DIR.mkdir(
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


    for target in (
        df["target"].unique()
    ):

        subset = (
            df[
                df["target"]
                == target
            ]
        )

        y_true = (
            subset[
                "y_true"
            ]
            .to_numpy()
        )


        plt.figure(
            figsize=(7, 6)
        )


        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Perfect calibration",
        )


        for method in METHODS:

            probability = (
                subset[
                    method
                ]
                .to_numpy()
            )


            fraction_positive, mean_predicted = (
                calibration_curve(
                    y_true,
                    probability,
                    n_bins=5,
                    strategy="quantile",
                )
            )


            plt.plot(
                mean_predicted,
                fraction_positive,
                marker="o",
                label=method,
            )


        plt.xlabel(
            "Mean predicted probability"
        )

        plt.ylabel(
            "Observed event rate"
        )

        plt.title(
            f"Calibration Curve — {target}"
        )

        plt.legend()

        plt.tight_layout()


        output_path = (
            FIGURE_DIR
            / f"{target}_calibration.png"
        )


        plt.savefig(
            output_path,
            dpi=300,
        )

        plt.close()


        print(
            "Saved:",
            output_path
        )


if __name__ == "__main__":
    main()