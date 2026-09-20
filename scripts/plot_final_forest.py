from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "final"
    / "final_hierarchical_bootstrap_all_methods.csv"
)

FIGURE_DIR = (
    ROOT
    / "figures"
    / "final"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


METHOD_ORDER = [
    "conservative_candidate",
    "dlar_lcl",
    "reliability_gated_sfda",
    "pseudo_label_sfda",
    "entropy_sfda",
]


METHOD_LABELS = {
    "conservative_candidate":
        "Conservative",

    "dlar_lcl":
        "DLAR-LCL",

    "reliability_gated_sfda":
        "Reliability v1",

    "pseudo_label_sfda":
        "Pseudo-label",

    "entropy_sfda":
        "Entropy",
}


DIRECTION_ORDER = [
    "Cleveland → Hungary",
    "Cleveland → Switzerland",
    "Cleveland → VA Long Beach",

    "Hungary → Cleveland",
    "Hungary → Switzerland",
    "Hungary → VA Long Beach",

    "Switzerland → Cleveland",
    "Switzerland → Hungary",
    "Switzerland → VA Long Beach",

    "VA Long Beach → Cleveland",
    "VA Long Beach → Hungary",
    "VA Long Beach → Switzerland",
]


def forest_plot(
    df,
    metric,
    xlabel,
    title,
    output_filename,
):

    subset = (
        df[
            df[
                "metric"
            ]
            == metric
        ]
        .copy()
    )


    rows = []


    for direction in DIRECTION_ORDER:

        for method in METHOD_ORDER:

            match = subset[
                (
                    subset[
                        "direction"
                    ]
                    == direction
                )
                &
                (
                    subset[
                        "method"
                    ]
                    == method
                )
            ]


            if len(match) != 1:

                continue


            row = (
                match.iloc[0]
            )


            rows.append(
                {
                    "direction":
                        direction,

                    "method":
                        method,

                    "label":
                        (
                            direction
                            +
                            " | "
                            +
                            METHOD_LABELS[
                                method
                            ]
                        ),

                    "estimate":
                        row[
                            "observed"
                        ],

                    "lower":
                        row[
                            "ci_lower"
                        ],

                    "upper":
                        row[
                            "ci_upper"
                        ],
                }
            )


    plot_df = pd.DataFrame(
        rows
    )


    n = len(
        plot_df
    )


    y = np.arange(
        n
    )


    fig, ax = plt.subplots(
        figsize=(
            10,
            max(
                10,
                0.27 * n
            ),
        )
    )


    estimates = (
        plot_df[
            "estimate"
        ]
        .to_numpy()
    )


    lower = (
        plot_df[
            "lower"
        ]
        .to_numpy()
    )


    upper = (
        plot_df[
            "upper"
        ]
        .to_numpy()
    )


    xerr = np.vstack(
        [
            estimates
            - lower,

            upper
            - estimates,
        ]
    )


    ax.errorbar(
        estimates,
        y,
        xerr=xerr,
        fmt="o",
        capsize=2,
    )


    ax.axvline(
        0.0,
        linewidth=1,
    )


    ax.set_yticks(
        y
    )


    ax.set_yticklabels(
        plot_df[
            "label"
        ]
    )


    ax.invert_yaxis()


    ax.set_xlabel(
        xlabel
    )


    ax.set_title(
        title
    )


    fig.tight_layout()


    output_path = (
        FIGURE_DIR
        / output_filename
    )


    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print(
        "Saved:",
        output_path
    )


def main():

    df = pd.read_csv(
        INPUT_FILE
    )


    forest_plot(
        df=df,

        metric=
            "delta_roc_auc",

        xlabel=
            "ΔAUROC (adapted − source-only)",

        title=
            "Cross-hospital adaptation effects on discrimination",

        output_filename=
            "forest_delta_auroc.png",
    )


    forest_plot(
        df=df,

        metric=
            "brier_improvement",

        xlabel=
            "Brier improvement (source-only − adapted)",

        title=
            "Cross-hospital adaptation effects on probabilistic accuracy",

        output_filename=
            "forest_brier_improvement.png",
    )


if __name__ == "__main__":
    main()