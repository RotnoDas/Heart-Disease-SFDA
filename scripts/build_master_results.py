from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

STATS_DIR = (
    ROOT
    / "results"
    / "statistics"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MULTISEED_FILE = (
    STATS_DIR
    / "multiseed_sfda_core8.csv"
)

CONSERVATIVE_FILE = (
    STATS_DIR
    / "conservative_candidate_ablation.csv"
)


PRIMARY_METHODS = [
    "source_only",
    "pseudo_label_sfda",
    "entropy_sfda",
    "reliability_gated_sfda",
]


def main():

    # ========================================================
    # Original multi-seed methods
    # ========================================================

    multiseed = pd.read_csv(
        MULTISEED_FILE
    )

    multiseed = (
        multiseed[
            multiseed["method"]
            .isin(
                PRIMARY_METHODS
            )
        ]
        .copy()
    )


    # ========================================================
    # Conservative candidate
    # ========================================================

    conservative = pd.read_csv(
        CONSERVATIVE_FILE
    )

    conservative = (
        conservative[
            conservative["variant"]
            == "candidate_full"
        ]
        .copy()
    )


    # Conservative file already contains source metrics
    candidate_rows = conservative[
        [
            "seed",
            "target",
            "roc_auc",
            "brier",
        ]
    ].copy()


    candidate_rows[
        "method"
    ] = "conservative_candidate"


    # PR-AUC and balanced accuracy are not currently stored
    # in the conservative ablation output.
    #
    # Keep them explicitly missing rather than inventing them.
    candidate_rows[
        "pr_auc"
    ] = np.nan

    candidate_rows[
        "balanced_accuracy"
    ] = np.nan


    # ========================================================
    # Harmonize
    # ========================================================

    columns = [
        "seed",
        "target",
        "method",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "brier",
    ]


    combined = pd.concat(
        [
            multiseed[
                columns
            ],
            candidate_rows[
                columns
            ],
        ],
        ignore_index=True,
    )


    # ========================================================
    # Source references
    # ========================================================

    source_reference = (
        combined[
            combined["method"]
            == "source_only"
        ][
            [
                "seed",
                "target",
                "roc_auc",
                "brier",
            ]
        ]
        .rename(
            columns={
                "roc_auc":
                    "source_roc_auc",

                "brier":
                    "source_brier",
            }
        )
    )


    combined = combined.merge(
        source_reference,
        on=[
            "seed",
            "target",
        ],
        how="left",
    )


    combined[
        "delta_roc_auc"
    ] = (
        combined["roc_auc"]
        -
        combined["source_roc_auc"]
    )


    combined[
        "brier_improvement"
    ] = (
        combined["source_brier"]
        -
        combined["brier"]
    )


    combined[
        "negative_adaptation"
    ] = (
        combined["delta_roc_auc"]
        < -1e-8
    )


    # Source only is reference, not adaptation.
    combined.loc[
        combined["method"]
        == "source_only",
        "negative_adaptation",
    ] = False


    raw_path = (
        OUTPUT_DIR
        / "master_multiseed_results.csv"
    )


    combined.to_csv(
        raw_path,
        index=False,
    )


    # ========================================================
    # Summary
    # ========================================================

    summary = (
        combined
        .groupby(
            [
                "target",
                "method",
            ],
            as_index=False,
        )
        .agg(

            mean_roc_auc=(
                "roc_auc",
                "mean",
            ),

            sd_roc_auc=(
                "roc_auc",
                "std",
            ),

            mean_pr_auc=(
                "pr_auc",
                "mean",
            ),

            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),

            mean_brier=(
                "brier",
                "mean",
            ),

            sd_brier=(
                "brier",
                "std",
            ),

            mean_delta_auc=(
                "delta_roc_auc",
                "mean",
            ),

            mean_brier_improvement=(
                "brier_improvement",
                "mean",
            ),

            negative_adaptation_rate=(
                "negative_adaptation",
                "mean",
            ),
        )
    )


    summary[
        "negative_adaptation_rate_pct"
    ] = (
        100
        * summary[
            "negative_adaptation_rate"
        ]
    )


    summary_path = (
        OUTPUT_DIR
        / "master_multiseed_summary.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    print(
        "\n"
        + "=" * 150
    )

    print(
        "FROZEN PRIMARY METHOD SUMMARY"
    )

    print(
        "=" * 150
    )


    print(
        summary[
            [
                "target",
                "method",
                "mean_roc_auc",
                "sd_roc_auc",
                "mean_brier",
                "sd_brier",
                "mean_delta_auc",
                "mean_brier_improvement",
                "negative_adaptation_rate_pct",
            ]
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\nSaved raw:",
        raw_path
    )

    print(
        "Saved summary:",
        summary_path
    )


if __name__ == "__main__":
    main()