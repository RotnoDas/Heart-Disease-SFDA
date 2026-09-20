from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "cross_source"
    / "cross_source_patient_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "statistics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

BOOTSTRAP_ITERATIONS = 5000

RANDOM_SEED = 42


METHODS = [
    "pseudo_label_sfda",
    "entropy_sfda",
    "reliability_gated_sfda",
    "conservative_candidate",
]


DOMAIN_ORDER = {
    "cleveland": 1,
    "hungary": 2,
    "switzerland": 3,
    "va_long_beach": 4,
}


# ============================================================
# Metrics
# ============================================================

def balanced_accuracy_binary(
    y_true,
    probability,
    threshold=0.5,
):

    y_true = np.asarray(
        y_true
    )

    prediction = (
        np.asarray(
            probability
        )
        >= threshold
    ).astype(int)


    positive_mask = (
        y_true == 1
    )

    negative_mask = (
        y_true == 0
    )


    sensitivity = float(
        prediction[
            positive_mask
        ].mean()
    )


    specificity = float(
        (
            prediction[
                negative_mask
            ]
            == 0
        ).mean()
    )


    return (
        sensitivity
        + specificity
    ) / 2.0


def metric_values(
    y_true,
    probability,
):

    y_true = np.asarray(
        y_true
    )

    probability = np.asarray(
        probability
    )


    return {
        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    probability,
                )
            ),

        "brier":
            float(
                np.mean(
                    (
                        probability
                        - y_true
                    ) ** 2
                )
            ),

        "balanced_accuracy":
            balanced_accuracy_binary(
                y_true,
                probability,
            ),
    }


# ============================================================
# Prepare one source-target direction
# ============================================================

def prepare_direction(
    direction_df,
):

    seeds = sorted(
        direction_df[
            "seed"
        ].unique()
    )


    # --------------------------------------------------------
    # Use first seed to establish patient order
    # --------------------------------------------------------

    first_seed_df = (
        direction_df[
            direction_df["seed"]
            == seeds[0]
        ]
        .sort_values(
            "row_id"
        )
        .reset_index(
            drop=True
        )
    )


    row_ids = (
        first_seed_df[
            "row_id"
        ]
        .astype(str)
        .to_numpy()
    )


    y_true = (
        first_seed_df[
            "y_true"
        ]
        .to_numpy(
            dtype=int
        )
    )


    prediction_columns = [
        "source_only",
        *METHODS,
    ]


    matrices = {
        column: []
        for column
        in prediction_columns
    }


    # --------------------------------------------------------
    # Every seed must contain same target patients
    # --------------------------------------------------------

    for seed in seeds:

        seed_df = (
            direction_df[
                direction_df["seed"]
                == seed
            ]
            .copy()
        )


        seed_df[
            "row_id"
        ] = (
            seed_df[
                "row_id"
            ]
            .astype(str)
        )


        seed_df = (
            seed_df
            .set_index(
                "row_id"
            )
            .loc[
                row_ids
            ]
        )


        seed_y = (
            seed_df[
                "y_true"
            ]
            .to_numpy(
                dtype=int
            )
        )


        if not np.array_equal(
            seed_y,
            y_true,
        ):

            raise ValueError(
                "Target labels/order "
                "differ across seeds."
            )


        for column in (
            prediction_columns
        ):

            matrices[
                column
            ].append(
                seed_df[
                    column
                ]
                .to_numpy(
                    dtype=float
                )
            )


    # --------------------------------------------------------
    # Shape:
    #
    # [n_seeds, n_patients]
    # --------------------------------------------------------

    for column in matrices:

        matrices[
            column
        ] = np.vstack(
            matrices[
                column
            ]
        )


    return (
        np.asarray(
            seeds
        ),
        row_ids,
        y_true,
        matrices,
    )


# ============================================================
# Observed paired effect
# ============================================================

def observed_effects(
    y_true,
    matrices,
):

    n_seeds = (
        matrices[
            "source_only"
        ].shape[0]
    )


    result = {}


    for method in METHODS:

        delta_auc = []

        brier_improvement = []

        delta_balanced_accuracy = []


        for seed_index in range(
            n_seeds
        ):

            source_metrics = (
                metric_values(
                    y_true,
                    matrices[
                        "source_only"
                    ][
                        seed_index
                    ],
                )
            )


            adapted_metrics = (
                metric_values(
                    y_true,
                    matrices[
                        method
                    ][
                        seed_index
                    ],
                )
            )


            delta_auc.append(
                adapted_metrics[
                    "roc_auc"
                ]
                -
                source_metrics[
                    "roc_auc"
                ]
            )


            brier_improvement.append(
                source_metrics[
                    "brier"
                ]
                -
                adapted_metrics[
                    "brier"
                ]
            )


            delta_balanced_accuracy.append(
                adapted_metrics[
                    "balanced_accuracy"
                ]
                -
                source_metrics[
                    "balanced_accuracy"
                ]
            )


        result[
            method
        ] = {

            "delta_roc_auc":
                float(
                    np.mean(
                        delta_auc
                    )
                ),

            "brier_improvement":
                float(
                    np.mean(
                        brier_improvement
                    )
                ),

            "delta_balanced_accuracy":
                float(
                    np.mean(
                        delta_balanced_accuracy
                    )
                ),
        }


    return result


# ============================================================
# Hierarchical bootstrap
# ============================================================

def hierarchical_bootstrap_direction(
    y_true,
    matrices,
    iterations,
    random_seed,
):

    rng = np.random.default_rng(
        random_seed
    )


    n_seeds = (
        matrices[
            "source_only"
        ].shape[0]
    )


    positive_indices = np.where(
        y_true == 1
    )[0]


    negative_indices = np.where(
        y_true == 0
    )[0]


    bootstrap_values = {

        method: {

            "delta_roc_auc":
                np.empty(
                    iterations,
                    dtype=float,
                ),

            "brier_improvement":
                np.empty(
                    iterations,
                    dtype=float,
                ),

            "delta_balanced_accuracy":
                np.empty(
                    iterations,
                    dtype=float,
                ),
        }

        for method in METHODS
    }


    for bootstrap_id in range(
        iterations
    ):

        # ====================================================
        # Level 1:
        # resample source initialization seeds
        # ====================================================

        seed_draw = rng.integers(
            low=0,
            high=n_seeds,
            size=n_seeds,
        )


        seed_counts = np.bincount(
            seed_draw,
            minlength=n_seeds,
        )


        active_seed_indices = (
            np.where(
                seed_counts > 0
            )[0]
        )


        seed_weights = (
            seed_counts[
                active_seed_indices
            ]
            / n_seeds
        )


        # ====================================================
        # Level 2:
        # stratified patient bootstrap
        #
        # Preserves observed positive/negative counts.
        # ====================================================

        positive_sample = (
            rng.choice(
                positive_indices,
                size=len(
                    positive_indices
                ),
                replace=True,
            )
        )


        negative_sample = (
            rng.choice(
                negative_indices,
                size=len(
                    negative_indices
                ),
                replace=True,
            )
        )


        patient_indices = (
            np.concatenate(
                [
                    negative_sample,
                    positive_sample,
                ]
            )
        )


        rng.shuffle(
            patient_indices
        )


        y_boot = (
            y_true[
                patient_indices
            ]
        )


        # ----------------------------------------------------
        # Running weighted paired deltas
        # ----------------------------------------------------

        replicate_values = {

            method: {
                "delta_roc_auc": 0.0,
                "brier_improvement": 0.0,
                "delta_balanced_accuracy": 0.0,
            }

            for method in METHODS
        }


        # ====================================================
        # Metrics for each sampled source seed
        # ====================================================

        for (
            weight,
            seed_index,
        ) in zip(
            seed_weights,
            active_seed_indices,
        ):

            source_probability = (
                matrices[
                    "source_only"
                ][
                    seed_index,
                    patient_indices
                ]
            )


            source_metrics = (
                metric_values(
                    y_boot,
                    source_probability,
                )
            )


            for method in METHODS:

                adapted_probability = (
                    matrices[
                        method
                    ][
                        seed_index,
                        patient_indices
                    ]
                )


                adapted_metrics = (
                    metric_values(
                        y_boot,
                        adapted_probability,
                    )
                )


                replicate_values[
                    method
                ][
                    "delta_roc_auc"
                ] += (
                    weight
                    * (
                        adapted_metrics[
                            "roc_auc"
                        ]
                        -
                        source_metrics[
                            "roc_auc"
                        ]
                    )
                )


                replicate_values[
                    method
                ][
                    "brier_improvement"
                ] += (
                    weight
                    * (
                        source_metrics[
                            "brier"
                        ]
                        -
                        adapted_metrics[
                            "brier"
                        ]
                    )
                )


                replicate_values[
                    method
                ][
                    "delta_balanced_accuracy"
                ] += (
                    weight
                    * (
                        adapted_metrics[
                            "balanced_accuracy"
                        ]
                        -
                        source_metrics[
                            "balanced_accuracy"
                        ]
                    )
                )


        # ====================================================
        # Store bootstrap replicate
        # ====================================================

        for method in METHODS:

            for metric_name in (
                bootstrap_values[
                    method
                ]
            ):

                bootstrap_values[
                    method
                ][
                    metric_name
                ][
                    bootstrap_id
                ] = (
                    replicate_values[
                        method
                    ][
                        metric_name
                    ]
                )


    return bootstrap_values


# ============================================================
# Main
# ============================================================

def main():

    df = pd.read_csv(
        INPUT_FILE
    )


    print(
        "\n"
        + "=" * 130
    )

    print(
        "HIERARCHICAL PAIRED BOOTSTRAP"
    )

    print(
        "=" * 130
    )


    print(
        "\nIterations:",
        BOOTSTRAP_ITERATIONS
    )


    print(
        "Hierarchy:"
        " source seeds + stratified target patients"
    )


    summary_rows = []

    replicate_rows = []


    directions = (
        df[
            [
                "source",
                "target",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "source",
                "target",
            ]
        )
    )


    for _, direction in (
        directions.iterrows()
    ):

        source = (
            direction[
                "source"
            ]
        )

        target = (
            direction[
                "target"
            ]
        )


        print(
            f"\n{source}"
            f" -> "
            f"{target}"
        )


        direction_df = (
            df[
                (
                    df["source"]
                    == source
                )
                &
                (
                    df["target"]
                    == target
                )
            ]
            .copy()
        )


        (
            seeds,
            row_ids,
            y_true,
            matrices,
        ) = prepare_direction(
            direction_df
        )


        print(
            f"  seeds={len(seeds)}"
            f" patients={len(y_true)}"
            f" positive={int(y_true.sum())}"
            f" negative="
            f"{int(len(y_true) - y_true.sum())}"
        )


        observed = (
            observed_effects(
                y_true,
                matrices,
            )
        )


        # ----------------------------------------------------
        # Deterministic direction-specific RNG seed
        # ----------------------------------------------------

        direction_seed = (

            RANDOM_SEED

            +

            DOMAIN_ORDER[
                source
            ]
            * 100

            +

            DOMAIN_ORDER[
                target
            ]
            * 10
        )


        bootstrap_values = (
            hierarchical_bootstrap_direction(
                y_true=
                    y_true,

                matrices=
                    matrices,

                iterations=
                    BOOTSTRAP_ITERATIONS,

                random_seed=
                    direction_seed,
            )
        )


        # ====================================================
        # Summaries
        # ====================================================

        for method in METHODS:

            for metric_name in [
                "delta_roc_auc",
                "brier_improvement",
                "delta_balanced_accuracy",
            ]:

                values = (
                    bootstrap_values[
                        method
                    ][
                        metric_name
                    ]
                )


                ci_lower = float(
                    np.percentile(
                        values,
                        2.5,
                    )
                )


                ci_upper = float(
                    np.percentile(
                        values,
                        97.5,
                    )
                )


                observed_value = (
                    observed[
                        method
                    ][
                        metric_name
                    ]
                )


                # --------------------------------------------
                # This is descriptive bootstrap support,
                # NOT a classical p-value.
                # --------------------------------------------

                positive_fraction = float(
                    np.mean(
                        values > 0
                    )
                )


                summary_rows.append(
                    {
                        "source":
                            source,

                        "target":
                            target,

                        "method":
                            method,

                        "metric":
                            metric_name,

                        "observed":
                            observed_value,

                        "ci_lower":
                            ci_lower,

                        "ci_upper":
                            ci_upper,

                        "ci_contains_zero":
                            bool(
                                ci_lower
                                <= 0
                                <= ci_upper
                            ),

                        "bootstrap_positive_fraction":
                            positive_fraction,

                        "n_source_seeds":
                            len(
                                seeds
                            ),

                        "n_target_patients":
                            len(
                                y_true
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
                    }
                )


            # ------------------------------------------------
            # Save replicate distribution
            # ------------------------------------------------

            for bootstrap_id in range(
                BOOTSTRAP_ITERATIONS
            ):

                replicate_rows.append(
                    {
                        "source":
                            source,

                        "target":
                            target,

                        "method":
                            method,

                        "bootstrap_id":
                            bootstrap_id,

                        "delta_roc_auc":
                            bootstrap_values[
                                method
                            ][
                                "delta_roc_auc"
                            ][
                                bootstrap_id
                            ],

                        "brier_improvement":
                            bootstrap_values[
                                method
                            ][
                                "brier_improvement"
                            ][
                                bootstrap_id
                            ],

                        "delta_balanced_accuracy":
                            bootstrap_values[
                                method
                            ][
                                "delta_balanced_accuracy"
                            ][
                                bootstrap_id
                            ],
                    }
                )


    # ========================================================
    # Save summary
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_path = (
        OUTPUT_DIR
        / "hierarchical_bootstrap_direction_summary.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False,
    )


    # ========================================================
    # Save bootstrap replicates
    #
    # gzip keeps file reasonably small.
    # ========================================================

    replicate_df = pd.DataFrame(
        replicate_rows
    )


    replicate_path = (
        OUTPUT_DIR
        / "hierarchical_bootstrap_replicates.csv.gz"
    )


    replicate_df.to_csv(
        replicate_path,
        index=False,
        compression="gzip",
    )


    # ========================================================
    # Print AUROC
    # ========================================================

    print(
        "\n"
        + "=" * 160
    )

    print(
        "HIERARCHICAL BOOTSTRAP — AUROC"
    )

    print(
        "=" * 160
    )


    auc_df = (
        summary_df[
            summary_df[
                "metric"
            ]
            == "delta_roc_auc"
        ]
    )


    print(
        auc_df[
            [
                "source",
                "target",
                "method",
                "observed",
                "ci_lower",
                "ci_upper",
                "ci_contains_zero",
                "bootstrap_positive_fraction",
            ]
        ]
        .to_string(
            index=False
        )
    )


    # ========================================================
    # Print Brier
    # ========================================================

    print(
        "\n"
        + "=" * 160
    )

    print(
        "HIERARCHICAL BOOTSTRAP — BRIER IMPROVEMENT"
    )

    print(
        "=" * 160
    )


    brier_df = (
        summary_df[
            summary_df[
                "metric"
            ]
            == "brier_improvement"
        ]
    )


    print(
        brier_df[
            [
                "source",
                "target",
                "method",
                "observed",
                "ci_lower",
                "ci_upper",
                "ci_contains_zero",
                "bootstrap_positive_fraction",
            ]
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\nSaved summary:",
        summary_path
    )


    print(
        "Saved replicate distributions:",
        replicate_path
    )


if __name__ == "__main__":
    main()