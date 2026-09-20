from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

STAT_DIR = (
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


MAIN_BOOTSTRAP = (
    STAT_DIR
    / "hierarchical_bootstrap_direction_summary.csv"
)

DLAR_BOOTSTRAP = (
    STAT_DIR
    / "dlar_lcl_hierarchical_bootstrap_summary.csv"
)


METHOD_ORDER = [
    "pseudo_label_sfda",
    "entropy_sfda",
    "reliability_gated_sfda",
    "conservative_candidate",
    "dlar_lcl",
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


def classify_ci(
    lower,
    upper,
):

    if (
        lower > 0
    ):
        return "improvement"

    if (
        upper < 0
    ):
        return "deterioration"

    return "inconclusive"


def main():

    # ========================================================
    # Existing single-head methods
    # ========================================================

    main_df = pd.read_csv(
        MAIN_BOOTSTRAP
    )


    main_df = (
        main_df[
            main_df[
                "method"
            ].isin(
                METHOD_ORDER[
                    :-1
                ]
            )
        ]
        .copy()
    )


    # ========================================================
    # DLAR-LCL
    # ========================================================

    dlar_df = pd.read_csv(
        DLAR_BOOTSTRAP
    )


    dlar_df[
        "method"
    ] = "dlar_lcl"


    # ========================================================
    # Combine
    # ========================================================

    combined = pd.concat(
        [
            main_df,
            dlar_df,
        ],
        ignore_index=True,
        sort=False,
    )


    combined[
        "method_label"
    ] = (
        combined[
            "method"
        ]
        .map(
            METHOD_LABELS
        )
    )


    combined[
        "source_label"
    ] = (
        combined[
            "source"
        ]
        .map(
            DOMAIN_LABELS
        )
    )


    combined[
        "target_label"
    ] = (
        combined[
            "target"
        ]
        .map(
            DOMAIN_LABELS
        )
    )


    combined[
        "direction"
    ] = (
        combined[
            "source_label"
        ]
        +
        " → "
        +
        combined[
            "target_label"
        ]
    )


    combined[
        "ci_classification"
    ] = [
        classify_ci(
            row.ci_lower,
            row.ci_upper,
        )
        for row
        in combined.itertuples()
    ]


    # ========================================================
    # Save long-form complete table
    # ========================================================

    long_path = (
        OUTPUT_DIR
        / "final_hierarchical_bootstrap_all_methods.csv"
    )


    combined.to_csv(
        long_path,
        index=False,
    )


    # ========================================================
    # AUROC table
    # ========================================================

    auc = (
        combined[
            combined[
                "metric"
            ]
            == "delta_roc_auc"
        ]
        .copy()
    )


    auc[
        "effect_ci"
    ] = (
        auc.apply(
            lambda r:
                f"{r['observed']:+.4f} "
                f"[{r['ci_lower']:+.4f}, "
                f"{r['ci_upper']:+.4f}]",
            axis=1,
        )
    )


    auc_path = (
        OUTPUT_DIR
        / "table_auroc_hierarchical_ci.csv"
    )


    auc[
        [
            "source",
            "target",
            "direction",
            "method",
            "method_label",
            "observed",
            "ci_lower",
            "ci_upper",
            "ci_contains_zero",
            "bootstrap_positive_fraction",
            "ci_classification",
            "effect_ci",
        ]
    ].to_csv(
        auc_path,
        index=False,
    )


    # ========================================================
    # Brier table
    # ========================================================

    brier = (
        combined[
            combined[
                "metric"
            ]
            == "brier_improvement"
        ]
        .copy()
    )


    brier[
        "effect_ci"
    ] = (
        brier.apply(
            lambda r:
                f"{r['observed']:+.4f} "
                f"[{r['ci_lower']:+.4f}, "
                f"{r['ci_upper']:+.4f}]",
            axis=1,
        )
    )


    brier_path = (
        OUTPUT_DIR
        / "table_brier_hierarchical_ci.csv"
    )


    brier[
        [
            "source",
            "target",
            "direction",
            "method",
            "method_label",
            "observed",
            "ci_lower",
            "ci_upper",
            "ci_contains_zero",
            "bootstrap_positive_fraction",
            "ci_classification",
            "effect_ci",
        ]
    ].to_csv(
        brier_path,
        index=False,
    )


    # ========================================================
    # Safety-count summary
    # ========================================================

    safety = (
        brier
        .groupby(
            [
                "method",
                "method_label",
            ],
            as_index=False,
        )
        .agg(

            improvement_directions=(
                "ci_classification",
                lambda x:
                    int(
                        (
                            x
                            == "improvement"
                        ).sum()
                    ),
            ),

            deterioration_directions=(
                "ci_classification",
                lambda x:
                    int(
                        (
                            x
                            == "deterioration"
                        ).sum()
                    ),
            ),

            inconclusive_directions=(
                "ci_classification",
                lambda x:
                    int(
                        (
                            x
                            == "inconclusive"
                        ).sum()
                    ),
            ),
        )
    )


    safety[
        "total_directions"
    ] = (
        safety[
            "improvement_directions"
        ]
        +
        safety[
            "deterioration_directions"
        ]
        +
        safety[
            "inconclusive_directions"
        ]
    )


    safety[
        "method_order"
    ] = (
        safety[
            "method"
        ]
        .map(
            {
                method: index
                for index, method
                in enumerate(
                    METHOD_ORDER
                )
            }
        )
    )


    safety = (
        safety
        .sort_values(
            "method_order"
        )
        .drop(
            columns=[
                "method_order"
            ]
        )
    )


    safety_path = (
        OUTPUT_DIR
        / "table_brier_safety_counts.csv"
    )


    safety.to_csv(
        safety_path,
        index=False,
    )


    # ========================================================
    # AUROC CI classification counts
    # ========================================================

    auc_counts = (
        auc
        .groupby(
            [
                "method",
                "method_label",
            ],
            as_index=False,
        )
        .agg(

            improvement_directions=(
                "ci_classification",
                lambda x:
                    int(
                        (
                            x
                            == "improvement"
                        ).sum()
                    ),
            ),

            deterioration_directions=(
                "ci_classification",
                lambda x:
                    int(
                        (
                            x
                            == "deterioration"
                        ).sum()
                    ),
            ),

            inconclusive_directions=(
                "ci_classification",
                lambda x:
                    int(
                        (
                            x
                            == "inconclusive"
                        ).sum()
                    ),
            ),
        )
    )


    auc_counts_path = (
        OUTPUT_DIR
        / "table_auroc_direction_counts.csv"
    )


    auc_counts.to_csv(
        auc_counts_path,
        index=False,
    )


    # ========================================================
    # Print
    # ========================================================

    print(
        "\n"
        + "=" * 110
    )

    print(
        "BRIER SAFETY COUNT SUMMARY"
    )

    print(
        "=" * 110
    )


    print(
        safety.to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 110
    )

    print(
        "AUROC DIRECTION COUNT SUMMARY"
    )

    print(
        "=" * 110
    )


    print(
        auc_counts.to_string(
            index=False
        )
    )


    print(
        "\nSaved:"
    )

    print(
        long_path
    )

    print(
        auc_path
    )

    print(
        brier_path
    )

    print(
        safety_path
    )

    print(
        auc_counts_path
    )


if __name__ == "__main__":
    main()