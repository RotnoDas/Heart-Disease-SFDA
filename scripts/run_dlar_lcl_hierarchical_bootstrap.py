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
    / "dlar_lcl"
    / "dlar_lcl_patient_predictions.csv"
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


DOMAIN_ORDER = {
    "cleveland": 1,
    "hungary": 2,
    "switzerland": 3,
    "va_long_beach": 4,
}


SOURCE_COLUMN = (
    "dual_head_source_only"
)

ADAPTED_COLUMN = (
    "dlar_lcl"
)


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

    probability = np.asarray(
        probability
    )

    prediction = (
        probability
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


    roc_auc = float(
        roc_auc_score(
            y_true,
            probability,
        )
    )


    brier = float(
        np.mean(
            (
                probability
                - y_true
            ) ** 2
        )
    )


    balanced_accuracy = (
        balanced_accuracy_binary(
            y_true,
            probability,
        )
    )


    return {
        "roc_auc":
            roc_auc,

        "brier":
            brier,

        "balanced_accuracy":
            balanced_accuracy,
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


    first_seed = (
        seeds[0]
    )


    first_df = (
        direction_df[
            direction_df[
                "seed"
            ]
            == first_seed
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Stable patient identity
    # --------------------------------------------------------

    first_df[
        "row_id_key"
    ] = (
        first_df[
            "row_id"
        ]
        .astype(str)
    )


    first_df = (
        first_df
        .sort_values(
            "row_id_key"
        )
        .reset_index(
            drop=True
        )
    )


    row_ids = (
        first_df[
            "row_id_key"
        ]
        .to_numpy()
    )


    y_true = (
        first_df[
            "y_true"
        ]
        .to_numpy(
            dtype=int
        )
    )


    source_matrix = []

    adapted_matrix = []


    for seed in seeds:

        seed_df = (
            direction_df[
                direction_df[
                    "seed"
                ]
                == seed
            ]
            .copy()
        )


        seed_df[
            "row_id_key"
        ] = (
            seed_df[
                "row_id"
            ]
            .astype(str)
        )


        seed_df = (
            seed_df
            .set_index(
                "row_id_key"
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
                f"Target labels differ "
                f"across seeds. Seed={seed}"
            )


        source_matrix.append(
            seed_df[
                SOURCE_COLUMN
            ]
            .to_numpy(
                dtype=float
            )
        )


        adapted_matrix.append(
            seed_df[
                ADAPTED_COLUMN
            ]
            .to_numpy(
                dtype=float
            )
        )


    source_matrix = np.vstack(
        source_matrix
    )


    adapted_matrix = np.vstack(
        adapted_matrix
    )


    return (
        np.asarray(
            seeds
        ),
        row_ids,
        y_true,
        source_matrix,
        adapted_matrix,
    )


# ============================================================
# Observed mean paired effect across source seeds
# ============================================================

def observed_effect(
    y_true,
    source_matrix,
    adapted_matrix,
):

    n_seeds = (
        source_matrix.shape[0]
    )


    delta_auc = []

    brier_improvement = []

    delta_balanced_accuracy = []


    for seed_index in range(
        n_seeds
    ):

        source_metrics = (
            metric_values(
                y_true,
                source_matrix[
                    seed_index
                ],
            )
        )


        adapted_metrics = (
            metric_values(
                y_true,
                adapted_matrix[
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


        # Positive = adapted model has lower Brier.
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


    return {
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


# ============================================================
# Hierarchical paired bootstrap
# ============================================================

def bootstrap_direction(
    y_true,
    source_matrix,
    adapted_matrix,
    iterations,
    random_seed,
):

    rng = np.random.default_rng(
        random_seed
    )


    n_seeds = (
        source_matrix.shape[0]
    )


    positive_indices = np.where(
        y_true == 1
    )[0]


    negative_indices = np.where(
        y_true == 0
    )[0]


    if (
        len(
            positive_indices
        )
        == 0
        or
        len(
            negative_indices
        )
        == 0
    ):

        raise ValueError(
            "Both classes are required "
            "for stratified bootstrap."
        )


    output = {
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


    for bootstrap_id in range(
        iterations
    ):

        # ====================================================
        # Level 1:
        # source-model initialization bootstrap
        # ====================================================

        seed_draw = rng.integers(
            low=0,
            high=n_seeds,
            size=n_seeds,
        )


        # ====================================================
        # Level 2:
        # target-patient bootstrap, stratified by outcome
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


        seed_delta_auc = []

        seed_brier_improvement = []

        seed_delta_balanced_accuracy = []


        # ====================================================
        # Same target bootstrap sample is used for
        # source-only and adapted predictions.
        #
        # Seed sampling is also paired.
        # ====================================================

        for seed_index in (
            seed_draw
        ):

            source_probability = (
                source_matrix[
                    seed_index,
                    patient_indices
                ]
            )


            adapted_probability = (
                adapted_matrix[
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


            adapted_metrics = (
                metric_values(
                    y_boot,
                    adapted_probability,
                )
            )


            seed_delta_auc.append(
                adapted_metrics[
                    "roc_auc"
                ]
                -
                source_metrics[
                    "roc_auc"
                ]
            )


            seed_brier_improvement.append(
                source_metrics[
                    "brier"
                ]
                -
                adapted_metrics[
                    "brier"
                ]
            )


            seed_delta_balanced_accuracy.append(
                adapted_metrics[
                    "balanced_accuracy"
                ]
                -
                source_metrics[
                    "balanced_accuracy"
                ]
            )


        output[
            "delta_roc_auc"
        ][
            bootstrap_id
        ] = float(
            np.mean(
                seed_delta_auc
            )
        )


        output[
            "brier_improvement"
        ][
            bootstrap_id
        ] = float(
            np.mean(
                seed_brier_improvement
            )
        )


        output[
            "delta_balanced_accuracy"
        ][
            bootstrap_id
        ] = float(
            np.mean(
                seed_delta_balanced_accuracy
            )
        )


    return output


# ============================================================
# Main
# ============================================================

def main():

    df = pd.read_csv(
        INPUT_FILE
    )


    required_columns = {
        "source",
        "target",
        "seed",
        "row_id",
        "y_true",
        SOURCE_COLUMN,
        ADAPTED_COLUMN,
    }


    missing = (
        required_columns
        -
        set(
            df.columns
        )
    )


    if missing:

        raise ValueError(
            "Missing columns: "
            + str(
                sorted(
                    missing
                )
            )
        )


    print(
        "\n"
        + "=" * 140
    )

    print(
        "DLAR-LCL HIERARCHICAL "
        "PAIRED BOOTSTRAP"
    )

    print(
        "=" * 140
    )


    print(
        f"\nIterations: "
        f"{BOOTSTRAP_ITERATIONS}"
    )


    print(
        "Hierarchy:"
        " source initialization seeds"
        " + stratified target patients"
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


        direction_df = (
            df[
                (
                    df[
                        "source"
                    ]
                    == source
                )
                &
                (
                    df[
                        "target"
                    ]
                    == target
                )
            ]
            .copy()
        )


        (
            seeds,
            row_ids,
            y_true,
            source_matrix,
            adapted_matrix,
        ) = prepare_direction(
            direction_df
        )


        print(
            f"\n{source}"
            f" -> "
            f"{target}"
        )


        print(
            f"  seeds="
            f"{len(seeds)}"
            f" patients="
            f"{len(y_true)}"
            f" positive="
            f"{int(y_true.sum())}"
            f" negative="
            f"{int(len(y_true) - y_true.sum())}"
        )


        observed = (
            observed_effect(
                y_true=
                    y_true,

                source_matrix=
                    source_matrix,

                adapted_matrix=
                    adapted_matrix,
            )
        )


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


        boot = (
            bootstrap_direction(
                y_true=
                    y_true,

                source_matrix=
                    source_matrix,

                adapted_matrix=
                    adapted_matrix,

                iterations=
                    BOOTSTRAP_ITERATIONS,

                random_seed=
                    direction_seed,
            )
        )


        for metric in [
            "delta_roc_auc",
            "brier_improvement",
            "delta_balanced_accuracy",
        ]:

            values = (
                boot[
                    metric
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


            summary_rows.append(
                {
                    "source":
                        source,

                    "target":
                        target,

                    "method":
                        "dlar_lcl",

                    "metric":
                        metric,

                    "observed":
                        observed[
                            metric
                        ],

                    "ci_lower":
                        ci_lower,

                    "ci_upper":
                        ci_upper,

                    "ci_contains_zero":
                        bool(
                            ci_lower
                            <= 0.0
                            <= ci_upper
                        ),

                    "bootstrap_positive_fraction":
                        float(
                            np.mean(
                                values > 0.0
                            )
                        ),

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
                            -
                            y_true.sum()
                        ),
                }
            )


        # ----------------------------------------------------
        # Save bootstrap distributions for forest plots etc.
        # ----------------------------------------------------

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
                        "dlar_lcl",

                    "bootstrap_id":
                        bootstrap_id,

                    "delta_roc_auc":
                        boot[
                            "delta_roc_auc"
                        ][
                            bootstrap_id
                        ],

                    "brier_improvement":
                        boot[
                            "brier_improvement"
                        ][
                            bootstrap_id
                        ],

                    "delta_balanced_accuracy":
                        boot[
                            "delta_balanced_accuracy"
                        ][
                            bootstrap_id
                        ],
                }
            )


    # ========================================================
    # Save
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_path = (
        OUTPUT_DIR
        / "dlar_lcl_hierarchical_bootstrap_summary.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False,
    )


    replicate_df = pd.DataFrame(
        replicate_rows
    )


    replicate_path = (
        OUTPUT_DIR
        / "dlar_lcl_hierarchical_bootstrap_replicates.csv.gz"
    )


    replicate_df.to_csv(
        replicate_path,
        index=False,
        compression="gzip",
    )


    # ========================================================
    # AUROC
    # ========================================================

    auc_df = (
        summary_df[
            summary_df[
                "metric"
            ]
            == "delta_roc_auc"
        ]
    )


    print(
        "\n"
        + "=" * 155
    )

    print(
        "DLAR-LCL HIERARCHICAL BOOTSTRAP — AUROC"
    )

    print(
        "=" * 155
    )


    print(
        auc_df[
            [
                "source",
                "target",
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
    # Brier
    # ========================================================

    brier_df = (
        summary_df[
            summary_df[
                "metric"
            ]
            == "brier_improvement"
        ]
    )


    print(
        "\n"
        + "=" * 155
    )

    print(
        "DLAR-LCL HIERARCHICAL BOOTSTRAP — "
        "BRIER IMPROVEMENT"
    )

    print(
        "=" * 155
    )


    print(
        brier_df[
            [
                "source",
                "target",
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
        "Saved bootstrap replicates:",
        replicate_path
    )


if __name__ == "__main__":
    main()