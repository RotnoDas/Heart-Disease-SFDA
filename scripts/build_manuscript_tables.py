from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)

CROSS_SOURCE_DIR = (
    ROOT
    / "results"
    / "cross_source"
)

DLAR_DIR = (
    ROOT
    / "results"
    / "dlar_lcl"
)

FINAL_DIR = (
    ROOT
    / "results"
    / "final"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

DOMAINS = {
    "cleveland":
        "cleveland.csv",

    "hungary":
        "hungary.csv",

    "switzerland":
        "switzerland.csv",

    "va_long_beach":
        "va_long_beach.csv",
}


DOMAIN_LABELS = {
    "cleveland":
        "Cleveland",

    "hungary":
        "Hungary",

    "switzerland":
        "Switzerland",

    "va_long_beach":
        "VA Long Beach",
}


CORE8 = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
]


METHOD_LABELS = {
    "pseudo_label_sfda":
        "Pseudo-label SFDA",

    "entropy_sfda":
        "Entropy SFDA",

    "reliability_gated_sfda":
        "Reliability-gated v1",

    "conservative_candidate":
        "Conservative candidate",

    "dlar_lcl":
        "DLAR-LCL port",
}


# ============================================================
# Helpers
# ============================================================

def fmt_mean_sd(
    mean_value,
    sd_value,
    digits=3,
):
    return (
        f"{mean_value:.{digits}f} "
        f"± "
        f"{sd_value:.{digits}f}"
    )


def fmt_effect_ci(
    estimate,
    lower,
    upper,
    digits=4,
):
    return (
        f"{estimate:+.{digits}f} "
        f"[{lower:+.{digits}f}, "
        f"{upper:+.{digits}f}]"
    )


# ============================================================
# Table 1:
# Dataset/domain characteristics
# ============================================================

def build_table_1():

    rows = []

    for domain, filename in (
        DOMAINS.items()
    ):

        df = pd.read_csv(
            DATA_DIR
            / filename
        )

        y = (
            df["target"]
            .to_numpy()
        )

        core = (
            df[
                CORE8
            ]
        )

        missing_cells = int(
            core
            .isna()
            .sum()
            .sum()
        )

        total_cells = int(
            core.shape[0]
            * core.shape[1]
        )

        complete_rows = int(
            core
            .notna()
            .all(
                axis=1
            )
            .sum()
        )

        rows.append(
            {
                "Domain":
                    DOMAIN_LABELS[
                        domain
                    ],

                "N":
                    len(df),

                "Positive_n":
                    int(
                        y.sum()
                    ),

                "Negative_n":
                    int(
                        len(y)
                        -
                        y.sum()
                    ),

                "Positive_prevalence_pct":
                    100.0
                    * float(
                        y.mean()
                    ),

                "CORE8_missing_cells":
                    missing_cells,

                "CORE8_missing_pct":
                    100.0
                    * missing_cells
                    / total_cells,

                "CORE8_complete_rows":
                    complete_rows,

                "CORE8_complete_rows_pct":
                    100.0
                    * complete_rows
                    / len(df),
            }
        )

    table = pd.DataFrame(
        rows
    )

    path = (
        FINAL_DIR
        / "manuscript_table1_domain_characteristics.csv"
    )

    table.to_csv(
        path,
        index=False,
    )

    return table, path


# ============================================================
# Table 2:
# Source-only cross-hospital performance
#
# Single-head architecture only.
# ============================================================

def build_table_2():

    input_file = (
        CROSS_SOURCE_DIR
        / "cross_source_multiseed_results.csv"
    )

    df = pd.read_csv(
        input_file
    )

    df = (
        df[
            df["method"]
            == "source_only"
        ]
        .copy()
    )

    summary = (
        df
        .groupby(
            [
                "source",
                "target",
            ],
            as_index=False,
        )
        .agg(
            mean_auroc=(
                "roc_auc",
                "mean",
            ),

            sd_auroc=(
                "roc_auc",
                "std",
            ),

            mean_pr_auc=(
                "pr_auc",
                "mean",
            ),

            sd_pr_auc=(
                "pr_auc",
                "std",
            ),

            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),

            sd_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),

            mean_brier=(
                "brier",
                "mean",
            ),

            sd_brier=(
                "brier",
                "std",
            ),
        )
    )

    summary[
        "Source"
    ] = (
        summary[
            "source"
        ]
        .map(
            DOMAIN_LABELS
        )
    )

    summary[
        "Target"
    ] = (
        summary[
            "target"
        ]
        .map(
            DOMAIN_LABELS
        )
    )

    summary[
        "AUROC_mean_sd"
    ] = [
        fmt_mean_sd(
            m,
            s,
        )
        for m, s
        in zip(
            summary[
                "mean_auroc"
            ],
            summary[
                "sd_auroc"
            ],
        )
    ]

    summary[
        "PR_AUC_mean_sd"
    ] = [
        fmt_mean_sd(
            m,
            s,
        )
        for m, s
        in zip(
            summary[
                "mean_pr_auc"
            ],
            summary[
                "sd_pr_auc"
            ],
        )
    ]

    summary[
        "Balanced_accuracy_mean_sd"
    ] = [
        fmt_mean_sd(
            m,
            s,
        )
        for m, s
        in zip(
            summary[
                "mean_balanced_accuracy"
            ],
            summary[
                "sd_balanced_accuracy"
            ],
        )
    ]

    summary[
        "Brier_mean_sd"
    ] = [
        fmt_mean_sd(
            m,
            s,
        )
        for m, s
        in zip(
            summary[
                "mean_brier"
            ],
            summary[
                "sd_brier"
            ],
        )
    ]

    manuscript = summary[
        [
            "Source",
            "Target",
            "AUROC_mean_sd",
            "PR_AUC_mean_sd",
            "Balanced_accuracy_mean_sd",
            "Brier_mean_sd",
        ]
    ].copy()

    path = (
        FINAL_DIR
        / "manuscript_table2_source_only_generalization.csv"
    )

    manuscript.to_csv(
        path,
        index=False,
    )

    return manuscript, path


# ============================================================
# Table 3:
# Hierarchical bootstrap adaptation effects
#
# Includes all methods.
# ============================================================

def build_table_3():

    input_file = (
        FINAL_DIR
        / "final_hierarchical_bootstrap_all_methods.csv"
    )

    df = pd.read_csv(
        input_file
    )

    # --------------------------------------------------------
    # Only AUROC + Brier for main manuscript
    # --------------------------------------------------------

    df = (
        df[
            df["metric"]
            .isin(
                [
                    "delta_roc_auc",
                    "brier_improvement",
                ]
            )
        ]
        .copy()
    )

    df[
        "Source"
    ] = (
        df[
            "source"
        ]
        .map(
            DOMAIN_LABELS
        )
    )

    df[
        "Target"
    ] = (
        df[
            "target"
        ]
        .map(
            DOMAIN_LABELS
        )
    )

    df[
        "Method"
    ] = (
        df[
            "method"
        ]
        .map(
            METHOD_LABELS
        )
    )

    # --------------------------------------------------------
    # Pivot AUROC and Brier into same row
    # --------------------------------------------------------

    auc = (
        df[
            df["metric"]
            == "delta_roc_auc"
        ][
            [
                "source",
                "target",
                "method",
                "observed",
                "ci_lower",
                "ci_upper",
                "ci_contains_zero",
            ]
        ]
        .copy()
    )

    auc = auc.rename(
        columns={
            "observed":
                "delta_auroc",

            "ci_lower":
                "delta_auroc_ci_lower",

            "ci_upper":
                "delta_auroc_ci_upper",

            "ci_contains_zero":
                "delta_auroc_ci_contains_zero",
        }
    )


    brier = (
        df[
            df["metric"]
            == "brier_improvement"
        ][
            [
                "source",
                "target",
                "method",
                "observed",
                "ci_lower",
                "ci_upper",
                "ci_contains_zero",
            ]
        ]
        .copy()
    )

    brier = brier.rename(
        columns={
            "observed":
                "brier_improvement",

            "ci_lower":
                "brier_ci_lower",

            "ci_upper":
                "brier_ci_upper",

            "ci_contains_zero":
                "brier_ci_contains_zero",
        }
    )


    merged = auc.merge(
        brier,
        on=[
            "source",
            "target",
            "method",
        ],
        how="inner",
    )


    merged[
        "Source"
    ] = (
        merged[
            "source"
        ]
        .map(
            DOMAIN_LABELS
        )
    )

    merged[
        "Target"
    ] = (
        merged[
            "target"
        ]
        .map(
            DOMAIN_LABELS
        )
    )

    merged[
        "Method"
    ] = (
        merged[
            "method"
        ]
        .map(
            METHOD_LABELS
        )
    )


    merged[
        "Delta_AUROC_95CI"
    ] = [
        fmt_effect_ci(
            estimate,
            lower,
            upper,
        )
        for (
            estimate,
            lower,
            upper,
        )
        in zip(
            merged[
                "delta_auroc"
            ],
            merged[
                "delta_auroc_ci_lower"
            ],
            merged[
                "delta_auroc_ci_upper"
            ],
        )
    ]


    merged[
        "Brier_improvement_95CI"
    ] = [
        fmt_effect_ci(
            estimate,
            lower,
            upper,
        )
        for (
            estimate,
            lower,
            upper,
        )
        in zip(
            merged[
                "brier_improvement"
            ],
            merged[
                "brier_ci_lower"
            ],
            merged[
                "brier_ci_upper"
            ],
        )
    ]


    manuscript = merged[
        [
            "Source",
            "Target",
            "Method",
            "Delta_AUROC_95CI",
            "Brier_improvement_95CI",
        ]
    ].copy()


    path = (
        FINAL_DIR
        / "manuscript_table3_adaptation_effects.csv"
    )

    manuscript.to_csv(
        path,
        index=False,
    )

    return manuscript, path


# ============================================================
# Table 4:
# Direction-level safety / evidence counts
# ============================================================

def build_table_4():

    brier_file = (
        FINAL_DIR
        / "table_brier_safety_counts.csv"
    )

    auc_file = (
        FINAL_DIR
        / "table_auroc_direction_counts.csv"
    )


    brier = pd.read_csv(
        brier_file
    )

    auc = pd.read_csv(
        auc_file
    )


    brier = brier.rename(
        columns={
            "improvement_directions":
                "Brier_improvement",

            "deterioration_directions":
                "Brier_deterioration",

            "inconclusive_directions":
                "Brier_inconclusive",
        }
    )


    auc = auc.rename(
        columns={
            "improvement_directions":
                "AUROC_improvement",

            "deterioration_directions":
                "AUROC_deterioration",

            "inconclusive_directions":
                "AUROC_inconclusive",
        }
    )


    merged = (
        auc[
            [
                "method",
                "method_label",
                "AUROC_improvement",
                "AUROC_deterioration",
                "AUROC_inconclusive",
            ]
        ]
        .merge(
            brier[
                [
                    "method",
                    "Brier_improvement",
                    "Brier_deterioration",
                    "Brier_inconclusive",
                ]
            ],
            on="method",
            how="inner",
        )
    )


    method_order = {
        "conservative_candidate": 0,
        "dlar_lcl": 1,
        "reliability_gated_sfda": 2,
        "pseudo_label_sfda": 3,
        "entropy_sfda": 4,
    }


    merged[
        "_order"
    ] = (
        merged[
            "method"
        ]
        .map(
            method_order
        )
    )


    merged = (
        merged
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
    )


    manuscript = (
        merged[
            [
                "method_label",
                "AUROC_improvement",
                "AUROC_deterioration",
                "AUROC_inconclusive",
                "Brier_improvement",
                "Brier_deterioration",
                "Brier_inconclusive",
            ]
        ]
        .rename(
            columns={
                "method_label":
                    "Method",
            }
        )
    )


    path = (
        FINAL_DIR
        / "manuscript_table4_direction_counts.csv"
    )


    manuscript.to_csv(
        path,
        index=False,
    )


    return manuscript, path


# ============================================================
# Main
# ============================================================

def main():

    table1, path1 = (
        build_table_1()
    )

    table2, path2 = (
        build_table_2()
    )

    table3, path3 = (
        build_table_3()
    )

    table4, path4 = (
        build_table_4()
    )


    print(
        "\n"
        + "=" * 130
    )

    print(
        "TABLE 1 — DOMAIN CHARACTERISTICS"
    )

    print(
        "=" * 130
    )

    print(
        table1.to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 130
    )

    print(
        "TABLE 2 — SOURCE-ONLY GENERALIZATION"
    )

    print(
        "=" * 130
    )

    print(
        table2.to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 130
    )

    print(
        "TABLE 4 — DIRECTION-LEVEL EVIDENCE COUNTS"
    )

    print(
        "=" * 130
    )

    print(
        table4.to_string(
            index=False
        )
    )


    print(
        "\nSaved:"
    )

    print(
        path1
    )

    print(
        path2
    )

    print(
        path3
    )

    print(
        path4
    )


if __name__ == "__main__":
    main()