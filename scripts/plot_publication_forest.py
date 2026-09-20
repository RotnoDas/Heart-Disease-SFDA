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

OUTPUT_DIR = (
    ROOT
    / "figures"
    / "final"
)

OUTPUT_DIR.mkdir(
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
        "Conservative candidate",

    "dlar_lcl":
        "DLAR-LCL port",

    "reliability_gated_sfda":
        "Reliability-gated v1",

    "pseudo_label_sfda":
        "Pseudo-label SFDA",

    "entropy_sfda":
        "Entropy SFDA",
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


def make_forest(
    df,
    metric,
    xlabel,
    filename,
):

    metric_df = (
        df[
            df["metric"]
            == metric
        ]
        .copy()
    )

    fig, axes = plt.subplots(
        nrows=len(METHOD_ORDER),
        ncols=1,
        figsize=(9, 18),
        sharex=True,
    )

    for ax, method in zip(
        axes,
        METHOD_ORDER,
    ):

        method_df = (
            metric_df[
                metric_df["method"]
                == method
            ]
            .copy()
        )

        method_df[
            "direction"
        ] = pd.Categorical(
            method_df["direction"],
            categories=DIRECTION_ORDER,
            ordered=True,
        )

        method_df = (
            method_df
            .sort_values(
                "direction"
            )
        )

        y = np.arange(
            len(method_df)
        )

        estimate = (
            method_df[
                "observed"
            ]
            .to_numpy()
        )

        lower = (
            method_df[
                "ci_lower"
            ]
            .to_numpy()
        )

        upper = (
            method_df[
                "ci_upper"
            ]
            .to_numpy()
        )

        xerr = np.vstack(
            [
                estimate - lower,
                upper - estimate,
            ]
        )

        ax.errorbar(
            estimate,
            y,
            xerr=xerr,
            fmt="o",
            capsize=3,
        )

        ax.axvline(
            0.0,
            linewidth=1,
        )

        ax.set_yticks(
            y
        )

        ax.set_yticklabels(
            method_df[
                "direction"
            ]
            .astype(str)
        )

        ax.invert_yaxis()

        ax.set_title(
            METHOD_LABELS[
                method
            ]
        )

        ax.grid(
            axis="x",
            alpha=0.25,
        )

    axes[-1].set_xlabel(
        xlabel
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / filename
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

    make_forest(
        df=df,

        metric=
            "delta_roc_auc",

        xlabel=
            "ΔAUROC (adapted − source-only)",

        filename=
            "publication_forest_auroc.png",
    )

    make_forest(
        df=df,

        metric=
            "brier_improvement",

        xlabel=
            "Brier improvement (source-only − adapted)",

        filename=
            "publication_forest_brier.png",
    )


if __name__ == "__main__":
    main()