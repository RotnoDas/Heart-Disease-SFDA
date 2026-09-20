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


def prepare_method_df(
    df,
    metric,
    method,
):
    subset = (
        df[
            (df["metric"] == metric)
            &
            (df["method"] == method)
        ]
        .copy()
    )

    subset["direction"] = pd.Categorical(
        subset["direction"],
        categories=DIRECTION_ORDER,
        ordered=True,
    )

    subset = (
        subset
        .sort_values("direction")
        .reset_index(drop=True)
    )

    return subset


def plot_one_axis(
    ax,
    method_df,
    title,
):
    y = np.arange(
        len(method_df)
    )

    estimate = (
        method_df["observed"]
        .to_numpy()
    )

    lower = (
        method_df["ci_lower"]
        .to_numpy()
    )

    upper = (
        method_df["ci_upper"]
        .to_numpy()
    )

    xerr = np.vstack(
        [
            estimate - lower,
            upper - estimate,
        ]
    )

    # --------------------------------------------------------
    # Draw confidence intervals first
    # --------------------------------------------------------

    ax.errorbar(
        estimate,
        y,
        xerr=xerr,
        fmt="none",
        capsize=3,
        linewidth=1.2,
    )

    # --------------------------------------------------------
    # Filled marker:
    # 95% hierarchical bootstrap CI excludes zero
    #
    # Hollow marker:
    # CI contains zero
    # --------------------------------------------------------

    contains_zero = (
        method_df[
            "ci_contains_zero"
        ]
        .astype(bool)
        .to_numpy()
    )

    significant_mask = (
        ~contains_zero
    )

    inconclusive_mask = (
        contains_zero
    )

    if (
        significant_mask.any()
    ):
        ax.scatter(
            estimate[
                significant_mask
            ],
            y[
                significant_mask
            ],
            s=36,
            zorder=3,
        )

    if (
        inconclusive_mask.any()
    ):
        ax.scatter(
            estimate[
                inconclusive_mask
            ],
            y[
                inconclusive_mask
            ],
            s=36,
            facecolors="none",
            zorder=3,
        )

    # --------------------------------------------------------
    # Neutral zero-reference line
    # --------------------------------------------------------

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
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
        title,
        fontsize=12,
        pad=7,
    )

    ax.grid(
        axis="x",
        alpha=0.18,
    )


def make_full_forest(
    df,
    metric,
    xlabel,
    filename,
):
    fig, axes = plt.subplots(
        nrows=len(METHOD_ORDER),
        ncols=1,
        figsize=(9.5, 18),
        sharex=True,
    )

    for ax, method in zip(
        axes,
        METHOD_ORDER,
    ):
        method_df = (
            prepare_method_df(
                df,
                metric,
                method,
            )
        )

        plot_one_axis(
            ax,
            method_df,
            METHOD_LABELS[
                method
            ],
        )

    axes[-1].set_xlabel(
        xlabel,
        fontsize=11,
    )

    fig.tight_layout(
        h_pad=1.2
    )

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


def make_brier_safety_zoom(
    df,
):
    methods = [
        "conservative_candidate",
        "dlar_lcl",
        "reliability_gated_sfda",
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(9.5, 11),
        sharex=True,
    )

    for ax, method in zip(
        axes,
        methods,
    ):
        method_df = (
            prepare_method_df(
                df,
                "brier_improvement",
                method,
            )
        )

        plot_one_axis(
            ax,
            method_df,
            METHOD_LABELS[
                method
            ],
        )

    # --------------------------------------------------------
    # Zoom chosen only for visualization.
    # It does NOT alter any estimates or inference.
    # --------------------------------------------------------

    axes[-1].set_xlim(
        -0.006,
        0.004,
    )

    axes[-1].set_xlabel(
        "Brier improvement (source-only − adapted)",
        fontsize=11,
    )

    fig.tight_layout(
        h_pad=1.2
    )

    output_path = (
        OUTPUT_DIR
        / "publication_forest_brier_safety_zoom.png"
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

    make_full_forest(
        df=df,

        metric=
            "delta_roc_auc",

        xlabel=
            "ΔAUROC (adapted − source-only)",

        filename=
            "publication_forest_auroc_v2.png",
    )

    make_full_forest(
        df=df,

        metric=
            "brier_improvement",

        xlabel=
            "Brier improvement (source-only − adapted)",

        filename=
            "publication_forest_brier_full_v2.png",
    )

    make_brier_safety_zoom(
        df
    )


if __name__ == "__main__":
    main()