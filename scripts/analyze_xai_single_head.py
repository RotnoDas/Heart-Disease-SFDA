from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

XAI_DIR = (
    ROOT
    / "results"
    / "explainability"
    / "xai_single_head"
)

FIGURE_DIR = (
    ROOT
    / "figures"
    / "explainability"
)

STABILITY_FILE = (
    XAI_DIR
    / "explanation_stability_seed.csv"
)

IMPORTANCE_FILE = (
    XAI_DIR
    / "global_feature_importance.csv"
)

DELTA_FILE = (
    ROOT
    / "results"
    / "cross_source"
    / "cross_source_multiseed_deltas.csv"
)


# ============================================================
# Frozen analysis settings
# ============================================================

ANALYSIS_VERSION = "xai_association_v1"

BOOTSTRAP_ITERATIONS = 5000

BOOTSTRAP_SEED = 20260921

# Canonical modeling pipeline uses this tolerance.
#
# negative_adaptation_auc:
#     delta_roc_auc < -1e-8
#
# brier_worsened:
#     brier_improvement < -1e-8
#
EFFECT_TOLERANCE = 1e-8


METHOD_ORDER = [
    "pseudo_label_sfda",
    "entropy_sfda",
    "reliability_gated_sfda",
    "conservative_candidate",
]


METHOD_LABELS = {
    "pseudo_label_sfda":
        "Pseudo-label",

    "entropy_sfda":
        "Entropy",

    "reliability_gated_sfda":
        "Reliability-gated",

    "conservative_candidate":
        "Conservative",

    "source_only":
        "Source-only",
}


XAI_METRICS = [
    "global_importance_spearman",
    "global_top3_overlap_fraction",
    "global_importance_relative_l1_drift",
    "patient_l1_drift_mean",
    "patient_l2_drift_mean",
    "patient_cosine_similarity_mean",
    "patient_top3_overlap_mean",
    "mean_absolute_probability_shift",
]


SUMMARY_METRICS = [
    # Performance
    "delta_roc_auc",
    "brier_improvement",
    "brier_degradation",
    "delta_balanced_accuracy",

    # XAI
    *XAI_METRICS,
]


# ============================================================
# Boolean parsing
# ============================================================

def coerce_boolean_column(
    series,
    column_name,
):

    if pd.api.types.is_bool_dtype(
        series
    ):

        return series.astype(bool)


    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }


    invalid = (
        ~normalized.isin(
            mapping.keys()
        )
    )


    if invalid.any():

        bad_values = (
            series[
                invalid
            ]
            .drop_duplicates()
            .tolist()
        )

        raise RuntimeError(
            f"Cannot parse boolean column "
            f"{column_name}: "
            f"{bad_values}"
        )


    return (
        normalized
        .map(
            mapping
        )
        .astype(bool)
    )


# ============================================================
# Generic summaries
# ============================================================

def flatten_summary(
    df,
    group_columns,
    metrics,
):

    grouped = (
        df
        .groupby(
            group_columns,
            sort=True,
        )[metrics]
        .agg(
            [
                "mean",
                "std",
                "median",
            ]
        )
        .reset_index()
    )


    flattened_columns = []


    for column in grouped.columns:

        if isinstance(
            column,
            tuple,
        ):

            components = [
                str(x)
                for x
                in column
                if (
                    x is not None
                    and
                    str(x) != ""
                )
            ]


            flattened_columns.append(
                "_".join(
                    components
                )
            )

        else:

            flattened_columns.append(
                str(
                    column
                )
            )


    grouped.columns = (
        flattened_columns
    )


    return grouped


# ============================================================
# Correlation helpers
# ============================================================

def safe_spearman(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )


    keep = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )


    x = x[
        keep
    ]

    y = y[
        keep
    ]


    if (
        len(x) < 3
        or
        len(
            np.unique(x)
        ) < 2
        or
        len(
            np.unique(y)
        ) < 2
    ):

        return (
            np.nan,
            np.nan,
            len(x),
        )


    result = spearmanr(
        x,
        y,
    )


    return (
        float(
            result.statistic
        ),

        float(
            result.pvalue
        ),

        len(x),
    )


def bootstrap_spearman(
    x,
    y,
    rng,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )


    keep = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )


    x = x[
        keep
    ]

    y = y[
        keep
    ]


    n = len(
        x
    )


    if n < 3:

        return (
            np.nan,
            np.nan,
            0,
        )


    values = []


    for _ in range(
        BOOTSTRAP_ITERATIONS
    ):

        indices = (
            rng.integers(
                0,
                n,
                size=n,
            )
        )


        xb = x[
            indices
        ]

        yb = y[
            indices
        ]


        if (
            len(
                np.unique(
                    xb
                )
            ) < 2
            or
            len(
                np.unique(
                    yb
                )
            ) < 2
        ):

            continue


        rho = (
            spearmanr(
                xb,
                yb,
            )
            .statistic
        )


        if np.isfinite(
            rho
        ):

            values.append(
                float(
                    rho
                )
            )


    if not values:

        return (
            np.nan,
            np.nan,
            0,
        )


    values = np.asarray(
        values,
        dtype=float,
    )


    return (
        float(
            np.percentile(
                values,
                2.5,
            )
        ),

        float(
            np.percentile(
                values,
                97.5,
            )
        ),

        len(
            values
        ),
    )


# ============================================================
# Figure helper
# ============================================================

def save_figure(
    fig,
    filename,
):

    path = (
        FIGURE_DIR
        / filename
    )


    fig.tight_layout()


    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print(
        "Saved figure:",
        path
    )


# ============================================================
# Load + validate
# ============================================================

def load_analysis_data():

    stability = pd.read_csv(
        STABILITY_FILE
    )


    deltas = pd.read_csv(
        DELTA_FILE
    )


    importance = pd.read_csv(
        IMPORTANCE_FILE
    )


    # --------------------------------------------------------
    # Column validation
    # --------------------------------------------------------

    expected_stability = {
        "source",
        "target",
        "seed",
        "method",
        *XAI_METRICS,
    }


    missing = (
        expected_stability
        -
        set(
            stability.columns
        )
    )


    if missing:

        raise RuntimeError(
            "Missing stability columns: "
            f"{sorted(missing)}"
        )


    expected_delta = {
        "source",
        "target",
        "seed",
        "method",
        "delta_roc_auc",
        "delta_balanced_accuracy",
        "brier_improvement",
        "negative_adaptation_auc",
        "brier_worsened",
    }


    missing = (
        expected_delta
        -
        set(
            deltas.columns
        )
    )


    if missing:

        raise RuntimeError(
            "Missing delta columns: "
            f"{sorted(missing)}"
        )


    expected_importance = {
        "source",
        "target",
        "seed",
        "model_state",
        "feature",
        "mean_abs_shap",
        "importance_rank",
    }


    missing = (
        expected_importance
        -
        set(
            importance.columns
        )
    )


    if missing:

        raise RuntimeError(
            "Missing importance columns: "
            f"{sorted(missing)}"
        )


    # --------------------------------------------------------
    # Restrict to frozen single-head methods
    # --------------------------------------------------------

    deltas = (
        deltas[
            deltas[
                "method"
            ].isin(
                METHOD_ORDER
            )
        ]
        .copy()
    )


    stability = (
        stability[
            stability[
                "method"
            ].isin(
                METHOD_ORDER
            )
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Duplicate-key checks before merge
    # --------------------------------------------------------

    key_columns = [
        "source",
        "target",
        "seed",
        "method",
    ]


    if (
        stability.duplicated(
            key_columns
        )
        .any()
    ):

        duplicate_rows = (
            stability[
                stability.duplicated(
                    key_columns,
                    keep=False,
                )
            ][
                key_columns
            ]
            .sort_values(
                key_columns
            )
        )


        raise RuntimeError(
            "Duplicate keys in XAI stability data:\n"
            +
            duplicate_rows.to_string(
                index=False
            )
        )


    if (
        deltas.duplicated(
            key_columns
        )
        .any()
    ):

        duplicate_rows = (
            deltas[
                deltas.duplicated(
                    key_columns,
                    keep=False,
                )
            ][
                key_columns
            ]
            .sort_values(
                key_columns
            )
        )


        raise RuntimeError(
            "Duplicate keys in performance delta data:\n"
            +
            duplicate_rows.to_string(
                index=False
            )
        )


    # --------------------------------------------------------
    # Exact one-to-one merge
    # --------------------------------------------------------

    merged = (
        stability.merge(
            deltas,
            on=key_columns,
            how="inner",
            validate="one_to_one",
        )
    )


    if (
        len(
            merged
        )
        != len(
            stability
        )
        or
        len(
            merged
        )
        != len(
            deltas
        )
    ):

        raise RuntimeError(
            "XAI/performance merge lost rows: "
            f"stability={len(stability)}, "
            f"deltas={len(deltas)}, "
            f"merged={len(merged)}"
        )


    # --------------------------------------------------------
    # Derived Brier degradation
    #
    # brier_improvement:
    #   positive = better
    #
    # brier_degradation:
    #   positive = worse
    # --------------------------------------------------------

    merged[
        "brier_degradation"
    ] = (
        -
        merged[
            "brier_improvement"
        ]
    )


    merged[
        "direction"
    ] = (
        merged[
            "source"
        ]
        .astype(str)
        +
        " -> "
        +
        merged[
            "target"
        ]
        .astype(str)
    )


    # ========================================================
    # Canonical frozen effect flags
    # ========================================================

    merged[
        "negative_adaptation_auc"
    ] = coerce_boolean_column(
        merged[
            "negative_adaptation_auc"
        ],
        "negative_adaptation_auc",
    )


    merged[
        "brier_worsened"
    ] = coerce_boolean_column(
        merged[
            "brier_worsened"
        ],
        "brier_worsened",
    )


    # --------------------------------------------------------
    # Exact canonical definitions from:
    # scripts/run_cross_source_replication.py
    #
    # negative adaptation:
    #     delta_roc_auc < -1e-8
    #
    # Brier worsening:
    #     brier_improvement < -1e-8
    # --------------------------------------------------------

    expected_negative = (
        merged[
            "delta_roc_auc"
        ]
        <
        -EFFECT_TOLERANCE
    )


    expected_brier_worsened = (
        merged[
            "brier_improvement"
        ]
        <
        -EFFECT_TOLERANCE
    )


    negative_match = (
        expected_negative.to_numpy()
        ==
        merged[
            "negative_adaptation_auc"
        ].to_numpy()
    )


    brier_match = (
        expected_brier_worsened.to_numpy()
        ==
        merged[
            "brier_worsened"
        ].to_numpy()
    )


    # --------------------------------------------------------
    # Save boundary audit
    # --------------------------------------------------------

    boundary_audit = (
        merged[
            [
                "source",
                "target",
                "seed",
                "method",
                "delta_roc_auc",
                "negative_adaptation_auc",
                "brier_improvement",
                "brier_worsened",
            ]
        ]
        .copy()
    )


    boundary_audit[
        "expected_negative_adaptation_auc"
    ] = (
        expected_negative
        .to_numpy()
    )


    boundary_audit[
        "negative_adaptation_flag_match"
    ] = negative_match


    boundary_audit[
        "expected_brier_worsened"
    ] = (
        expected_brier_worsened
        .to_numpy()
    )


    boundary_audit[
        "brier_worsened_flag_match"
    ] = brier_match


    boundary_audit[
        "effect_tolerance"
    ] = (
        EFFECT_TOLERANCE
    )


    boundary_audit.to_csv(
        XAI_DIR
        / "canonical_flag_boundary_audit.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Fail only if frozen CSV actually disagrees with
    # frozen canonical definition.
    # --------------------------------------------------------

    if not np.all(
        negative_match
    ):

        mismatch = (
            boundary_audit[
                ~boundary_audit[
                    "negative_adaptation_flag_match"
                ]
            ]
        )


        raise RuntimeError(
            "Canonical negative-adaptation "
            "flag mismatch.\n"
            +
            mismatch.to_string(
                index=False
            )
        )


    if not np.all(
        brier_match
    ):

        mismatch = (
            boundary_audit[
                ~boundary_audit[
                    "brier_worsened_flag_match"
                ]
            ]
        )


        raise RuntimeError(
            "Canonical Brier-worsening "
            "flag mismatch.\n"
            +
            mismatch.to_string(
                index=False
            )
        )


    return (
        merged,
        importance,
    )


# ============================================================
# Method + direction summaries
# ============================================================

def build_method_summaries(
    merged,
):

    method_summary = (
        flatten_summary(
            df=
                merged,

            group_columns=[
                "method",
            ],

            metrics=
                SUMMARY_METRICS,
        )
    )


    counts = (
        merged
        .groupby(
            "method"
        )
        .size()
    )


    negative_rates = (
        merged
        .groupby(
            "method"
        )[
            "negative_adaptation_auc"
        ]
        .mean()
    )


    brier_rates = (
        merged
        .groupby(
            "method"
        )[
            "brier_worsened"
        ]
        .mean()
    )


    method_summary[
        "n_seed_direction_pairs"
    ] = (
        method_summary[
            "method"
        ]
        .map(
            counts
        )
    )


    method_summary[
        "negative_adaptation_auc_rate"
    ] = (
        method_summary[
            "method"
        ]
        .map(
            negative_rates
        )
    )


    method_summary[
        "brier_worsened_rate"
    ] = (
        method_summary[
            "method"
        ]
        .map(
            brier_rates
        )
    )


    method_summary.to_csv(
        XAI_DIR
        / "xai_method_summary.csv",
        index=False,
    )


    direction_summary = (
        flatten_summary(
            df=
                merged,

            group_columns=[
                "source",
                "target",
                "method",
            ],

            metrics=
                SUMMARY_METRICS,
        )
    )


    direction_summary.to_csv(
        XAI_DIR
        / "xai_direction_summary.csv",
        index=False,
    )


    return (
        method_summary,
        direction_summary,
    )


# ============================================================
# Feature importance summary
#
# Absolute SHAP magnitudes from different target-background
# jobs should not be interpreted as directly exchangeable.
#
# Therefore:
#   - retain mean_abs_shap descriptively
#   - normalize importance within every
#     source-target-seed-model job
# ============================================================

def build_feature_summary(
    importance,
):

    importance = (
        importance.copy()
    )


    job_columns = [
        "source",
        "target",
        "seed",
        "model_state",
    ]


    total_importance = (
        importance
        .groupby(
            job_columns
        )[
            "mean_abs_shap"
        ]
        .transform(
            "sum"
        )
    )


    importance[
        "importance_fraction"
    ] = np.where(
        total_importance
        >
        0,

        importance[
            "mean_abs_shap"
        ]
        /
        total_importance,

        np.nan,
    )


    importance[
        "top3"
    ] = (
        importance[
            "importance_rank"
        ]
        <= 3
    )


    feature_summary = (
        importance
        .groupby(
            [
                "model_state",
                "feature",
            ],
            sort=True,
        )
        .agg(
            mean_abs_shap_mean=(
                "mean_abs_shap",
                "mean",
            ),

            mean_abs_shap_sd=(
                "mean_abs_shap",
                "std",
            ),

            importance_fraction_mean=(
                "importance_fraction",
                "mean",
            ),

            importance_fraction_sd=(
                "importance_fraction",
                "std",
            ),

            importance_fraction_median=(
                "importance_fraction",
                "median",
            ),

            mean_importance_rank=(
                "importance_rank",
                "mean",
            ),

            median_importance_rank=(
                "importance_rank",
                "median",
            ),

            top3_frequency=(
                "top3",
                "mean",
            ),

            n_jobs=(
                "importance_fraction",
                "count",
            ),
        )
        .reset_index()
    )


    feature_summary[
        "aggregate_fraction_rank"
    ] = (
        feature_summary
        .groupby(
            "model_state"
        )[
            "importance_fraction_mean"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )


    feature_summary = (
        feature_summary
        .sort_values(
            [
                "model_state",
                "aggregate_fraction_rank",
                "feature",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    feature_summary.to_csv(
        XAI_DIR
        / "xai_feature_importance_summary.csv",
        index=False,
    )


    return (
        importance,
        feature_summary,
    )


# ============================================================
# Plot-ready analysis tables
# ============================================================

def build_plot_datasets(
    merged,
):

    auroc_columns = [
        "source",
        "target",
        "direction",
        "seed",
        "method",
        "delta_roc_auc",
        "negative_adaptation_auc",
        *XAI_METRICS,
    ]


    brier_columns = [
        "source",
        "target",
        "direction",
        "seed",
        "method",
        "brier_improvement",
        "brier_degradation",
        "brier_worsened",
        *XAI_METRICS,
    ]


    merged[
        auroc_columns
    ].to_csv(
        XAI_DIR
        / "xai_drift_vs_auroc.csv",
        index=False,
    )


    merged[
        brier_columns
    ].to_csv(
        XAI_DIR
        / "xai_drift_vs_brier.csv",
        index=False,
    )


# ============================================================
# Negative-adaptation descriptive comparison
# ============================================================

def build_negative_adaptation_summary(
    merged,
):

    rows = []


    conditions = [
        "negative_adaptation_auc",
        "brier_worsened",
    ]


    scopes = [
        "ALL",
        *METHOD_ORDER,
    ]


    for condition in conditions:

        for scope in scopes:

            if (
                scope
                == "ALL"
            ):

                subset = (
                    merged
                )

            else:

                subset = (
                    merged[
                        merged[
                            "method"
                        ]
                        == scope
                    ]
                )


            positive_group = (
                subset[
                    subset[
                        condition
                    ]
                ]
            )


            negative_group = (
                subset[
                    ~subset[
                        condition
                    ]
                ]
            )


            for metric in XAI_METRICS:

                true_mean = (
                    float(
                        positive_group[
                            metric
                        ]
                        .mean()
                    )
                    if len(
                        positive_group
                    )
                    else np.nan
                )


                false_mean = (
                    float(
                        negative_group[
                            metric
                        ]
                        .mean()
                    )
                    if len(
                        negative_group
                    )
                    else np.nan
                )


                rows.append(
                    {
                        "condition":
                            condition,

                        "method_scope":
                            scope,

                        "metric":
                            metric,

                        "n_condition_true":
                            int(
                                len(
                                    positive_group
                                )
                            ),

                        "n_condition_false":
                            int(
                                len(
                                    negative_group
                                )
                            ),

                        "mean_condition_true":
                            true_mean,

                        "mean_condition_false":
                            false_mean,

                        "median_condition_true":
                            (
                                float(
                                    positive_group[
                                        metric
                                    ]
                                    .median()
                                )
                                if len(
                                    positive_group
                                )
                                else np.nan
                            ),

                        "median_condition_false":
                            (
                                float(
                                    negative_group[
                                        metric
                                    ]
                                    .median()
                                )
                                if len(
                                    negative_group
                                )
                                else np.nan
                            ),

                        "mean_difference_true_minus_false":
                            (
                                true_mean
                                -
                                false_mean
                                if (
                                    np.isfinite(
                                        true_mean
                                    )
                                    and
                                    np.isfinite(
                                        false_mean
                                    )
                                )
                                else np.nan
                            ),
                    }
                )


    result = pd.DataFrame(
        rows
    )


    result.to_csv(
        XAI_DIR
        / "xai_negative_adaptation_summary.csv",
        index=False,
    )


    return result


# ============================================================
# XAI-performance associations
#
# 1. Seed-level:
#    descriptive only; repeated observations are correlated.
#
# 2. Direction-level method-specific:
#    average 10 source seeds within each source-target direction.
#    n = 12 directions per method.
#
# Bootstrap CI is exploratory and resamples directions.
# No multiplicity correction.
# ============================================================

def build_association_statistics(
    merged,
):

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )


    rows = []


    outcomes = {
        "delta_roc_auc":
            (
                "AUROC change; "
                "positive = improvement"
            ),

        "brier_degradation":
            (
                "Brier degradation; "
                "positive = worsening"
            ),
    }


    # --------------------------------------------------------
    # Seed-level descriptive associations
    # --------------------------------------------------------

    for scope in [
        "ALL",
        *METHOD_ORDER,
    ]:

        if (
            scope
            == "ALL"
        ):

            subset = (
                merged
            )

        else:

            subset = (
                merged[
                    merged[
                        "method"
                    ]
                    == scope
                ]
            )


        for metric in XAI_METRICS:

            for (
                outcome,
                interpretation,
            ) in outcomes.items():

                (
                    rho,
                    p_value,
                    n,
                ) = safe_spearman(
                    subset[
                        metric
                    ],

                    subset[
                        outcome
                    ],
                )


                rows.append(
                    {
                        "analysis_level":
                            "seed_level_descriptive",

                        "method":
                            scope,

                        "xai_metric":
                            metric,

                        "outcome":
                            outcome,

                        "outcome_interpretation":
                            interpretation,

                        "n":
                            int(
                                n
                            ),

                        "spearman_rho":
                            rho,

                        "nominal_p_value":
                            p_value,

                        "bootstrap_ci_low":
                            np.nan,

                        "bootstrap_ci_high":
                            np.nan,

                        "bootstrap_valid_iterations":
                            0,

                        "inference_note":
                            (
                                "Descriptive only; "
                                "seed/direction observations "
                                "are not treated as independent."
                            ),
                    }
                )


    # --------------------------------------------------------
    # Direction-level means
    # --------------------------------------------------------

    numeric_columns = [
        *XAI_METRICS,
        "delta_roc_auc",
        "brier_degradation",
    ]


    direction_means = (
        merged
        .groupby(
            [
                "source",
                "target",
                "method",
            ],
            sort=True,
        )[
            numeric_columns
        ]
        .mean()
        .reset_index()
    )


    direction_means[
        "direction"
    ] = (
        direction_means[
            "source"
        ]
        .astype(str)
        +
        " -> "
        +
        direction_means[
            "target"
        ]
        .astype(str)
    )


    for method in METHOD_ORDER:

        subset = (
            direction_means[
                direction_means[
                    "method"
                ]
                == method
            ]
            .copy()
        )


        if (
            len(
                subset
            )
            != 12
        ):

            raise RuntimeError(
                f"Expected 12 direction-level rows "
                f"for {method}, found "
                f"{len(subset)}."
            )


        for metric in XAI_METRICS:

            for (
                outcome,
                interpretation,
            ) in outcomes.items():

                (
                    rho,
                    p_value,
                    n,
                ) = safe_spearman(
                    subset[
                        metric
                    ],

                    subset[
                        outcome
                    ],
                )


                (
                    ci_low,
                    ci_high,
                    valid_bootstrap,
                ) = bootstrap_spearman(
                    subset[
                        metric
                    ],

                    subset[
                        outcome
                    ],

                    rng=
                        rng,
                )


                rows.append(
                    {
                        "analysis_level":
                            (
                                "direction_level_"
                                "method_specific"
                            ),

                        "method":
                            method,

                        "xai_metric":
                            metric,

                        "outcome":
                            outcome,

                        "outcome_interpretation":
                            interpretation,

                        "n":
                            int(
                                n
                            ),

                        "spearman_rho":
                            rho,

                        "nominal_p_value":
                            p_value,

                        "bootstrap_ci_low":
                            ci_low,

                        "bootstrap_ci_high":
                            ci_high,

                        "bootstrap_valid_iterations":
                            int(
                                valid_bootstrap
                            ),

                        "inference_note":
                            (
                                "Exploratory. "
                                "10 source seeds averaged "
                                "within each of 12 directions. "
                                "Bootstrap resamples directions; "
                                "no multiplicity correction."
                            ),
                    }
                )


    result = pd.DataFrame(
        rows
    )


    result.to_csv(
        XAI_DIR
        / "xai_association_statistics.csv",
        index=False,
    )


    direction_means.to_csv(
        XAI_DIR
        / "xai_direction_level_association_data.csv",
        index=False,
    )


    return (
        result,
        direction_means,
    )


# ============================================================
# Figures
# ============================================================

def build_figures(
    merged,
    feature_summary,
):

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Figure 1:
    # Source-only normalized global importance
    # ========================================================

    source_importance = (
        feature_summary[
            feature_summary[
                "model_state"
            ]
            == "source_only"
        ]
        .sort_values(
            "importance_fraction_mean",
            ascending=True,
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            7.5,
            5.0,
        )
    )


    ax.barh(
        source_importance[
            "feature"
        ],

        source_importance[
            "importance_fraction_mean"
        ],

        xerr=
            source_importance[
                "importance_fraction_sd"
            ],
    )


    ax.set_xlabel(
        "Normalized mean |SHAP| fraction"
    )


    ax.set_ylabel(
        "CORE8 feature"
    )


    ax.set_title(
        "Source-only global feature importance"
    )


    save_figure(
        fig,
        "shap_global_importance.png",
    )


    # ========================================================
    # Figure 2:
    # Rank stability
    # ========================================================

    data = [
        merged[
            merged[
                "method"
            ]
            == method
        ][
            "global_importance_spearman"
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )

        for method
        in METHOD_ORDER
    ]


    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.0,
        )
    )


    ax.boxplot(
        data,

        tick_labels=[
            METHOD_LABELS[
                method
            ]
            for method
            in METHOD_ORDER
        ],

        showfliers=False,
    )


    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )


    ax.set_ylabel(
        "Spearman correlation"
    )


    ax.set_title(
        "Source vs adapted "
        "feature-importance rank stability"
    )


    ax.tick_params(
        axis="x",
        rotation=20,
    )


    save_figure(
        fig,
        "explanation_rank_stability.png",
    )


    # ========================================================
    # Figure 3:
    # Relative global importance drift
    # ========================================================

    data = [
        merged[
            merged[
                "method"
            ]
            == method
        ][
            "global_importance_relative_l1_drift"
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )

        for method
        in METHOD_ORDER
    ]


    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.0,
        )
    )


    ax.boxplot(
        data,

        tick_labels=[
            METHOD_LABELS[
                method
            ]
            for method
            in METHOD_ORDER
        ],

        showfliers=False,
    )


    ax.set_ylabel(
        "Relative L1 drift in "
        "global |SHAP| importance"
    )


    ax.set_title(
        "Explanation drift "
        "by adaptation method"
    )


    ax.tick_params(
        axis="x",
        rotation=20,
    )


    save_figure(
        fig,
        "explanation_drift_by_method.png",
    )


    # ========================================================
    # Figure 4:
    # Drift vs Brier degradation
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            7.0,
            5.5,
        )
    )


    for method in METHOD_ORDER:

        subset = (
            merged[
                merged[
                    "method"
                ]
                == method
            ]
        )


        ax.scatter(
            subset[
                "global_importance_relative_l1_drift"
            ],

            subset[
                "brier_degradation"
            ],

            alpha=0.55,

            label=
                METHOD_LABELS[
                    method
                ],
        )


    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )


    ax.set_xlabel(
        "Relative global explanation drift"
    )


    ax.set_ylabel(
        "Brier degradation\n"
        "(positive = worse)"
    )


    ax.set_title(
        "Explanation drift vs "
        "probability-quality degradation"
    )


    ax.legend(
        frameon=False
    )


    save_figure(
        fig,
        "xai_drift_vs_brier.png",
    )


    # ========================================================
    # Figure 5:
    # Drift vs AUROC
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(
            7.0,
            5.5,
        )
    )


    for method in METHOD_ORDER:

        subset = (
            merged[
                merged[
                    "method"
                ]
                == method
            ]
        )


        ax.scatter(
            subset[
                "global_importance_relative_l1_drift"
            ],

            subset[
                "delta_roc_auc"
            ],

            alpha=0.55,

            label=
                METHOD_LABELS[
                    method
                ],
        )


    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )


    ax.set_xlabel(
        "Relative global explanation drift"
    )


    ax.set_ylabel(
        "Delta AUROC\n"
        "(positive = improvement)"
    )


    ax.set_title(
        "Explanation drift vs "
        "discrimination change"
    )


    ax.legend(
        frameon=False
    )


    save_figure(
        fig,
        "xai_drift_vs_auroc.png",
    )


    # ========================================================
    # Figure 6:
    # AUROC-negative-adaptation comparison
    # ========================================================

    no_negative = (
        merged[
            ~merged[
                "negative_adaptation_auc"
            ]
        ][
            "global_importance_relative_l1_drift"
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )


    negative = (
        merged[
            merged[
                "negative_adaptation_auc"
            ]
        ][
            "global_importance_relative_l1_drift"
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            6.5,
            5.0,
        )
    )


    ax.boxplot(
        [
            no_negative,
            negative,
        ],

        tick_labels=[
            "No AUROC loss",
            "Negative adaptation",
        ],

        showfliers=False,
    )


    ax.set_ylabel(
        "Relative global explanation drift"
    )


    ax.set_title(
        "Explanation drift and "
        "AUROC negative adaptation"
    )


    save_figure(
        fig,
        "xai_negative_adaptation_comparison.png",
    )


# ============================================================
# Console audit
# ============================================================

def print_audit(
    merged,
    method_summary,
    feature_summary,
    association,
):

    print(
        "\n"
        + "=" * 110
    )


    print(
        "SINGLE-HEAD XAI "
        "ASSOCIATION ANALYSIS COMPLETE"
    )


    print(
        "=" * 110
    )


    print(
        "\nMerged seed-level rows:",
        len(
            merged
        )
    )


    print(
        "Unique directions:",
        merged[
            [
                "source",
                "target",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )


    print(
        "Methods:",
        merged[
            "method"
        ]
        .nunique()
    )


    print(
        "\nExpected merged rows: "
        "12 directions x 10 seeds "
        "x 4 methods = 480"
    )


    print(
        "\nCanonical effect tolerance:",
        EFFECT_TOLERANCE
    )


    print(
        "AUROC negative-adaptation flag:",
        "delta_roc_auc < -1e-8"
    )


    print(
        "Brier worsening flag:",
        "brier_improvement < -1e-8"
    )


    # --------------------------------------------------------
    # Top source-only features
    # --------------------------------------------------------

    print(
        "\nTop-3 source-only features "
        "by normalized mean |SHAP|:"
    )


    top_source = (
        feature_summary[
            feature_summary[
                "model_state"
            ]
            == "source_only"
        ]
        .sort_values(
            "aggregate_fraction_rank"
        )
        .head(
            3
        )
    )


    for row in (
        top_source
        .itertuples(
            index=False
        )
    ):

        print(
            f"  "
            f"{row.aggregate_fraction_rank}. "
            f"{row.feature}: "
            f"{row.importance_fraction_mean:.4f}"
        )


    # --------------------------------------------------------
    # Method-level explanation stability
    # --------------------------------------------------------

    print(
        "\nMethod-level "
        "explanation stability:"
    )


    method_summary_print = (
        method_summary.set_index(
            "method"
        )
    )


    for method in METHOD_ORDER:

        row = (
            method_summary_print
            .loc[
                method
            ]
        )


        print(
            f"  "
            f"{METHOD_LABELS[method]:20s}"
            f" Spearman="
            f"{row['global_importance_spearman_mean']:.4f}"
            f"  relative_L1="
            f"{row['global_importance_relative_l1_drift_mean']:.4f}"
            f"  patient_cosine="
            f"{row['patient_cosine_similarity_mean_mean']:.4f}"
        )


    print(
        "\nAssociation rows:",
        len(
            association
        )
    )


    # --------------------------------------------------------
    # Boundary flag counts
    # --------------------------------------------------------

    print(
        "\nCanonical seed-level flag counts:"
    )


    print(
        "  AUROC negative adaptation:",
        int(
            merged[
                "negative_adaptation_auc"
            ]
            .sum()
        ),
        "/",
        len(
            merged
        ),
    )


    print(
        "  Brier worsening:",
        int(
            merged[
                "brier_worsened"
            ]
            .sum()
        ),
        "/",
        len(
            merged
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    XAI_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Load
    # ========================================================

    (
        merged,
        importance,
    ) = load_analysis_data()


    # ========================================================
    # Expected design
    #
    # 12 directions
    # x 10 seeds
    # x 4 adapted methods
    # = 480
    # ========================================================

    if (
        len(
            merged
        )
        != 480
    ):

        raise RuntimeError(
            "Expected exactly 480 merged "
            "single-head XAI/performance rows, "
            f"found {len(merged)}."
        )


    if (
        merged[
            [
                "source",
                "target",
            ]
        ]
        .drop_duplicates()
        .shape[0]
        != 12
    ):

        raise RuntimeError(
            "Expected exactly 12 "
            "source-target directions."
        )


    if (
        merged[
            "method"
        ]
        .nunique()
        != 4
    ):

        raise RuntimeError(
            "Expected exactly 4 "
            "single-head adaptation methods."
        )


    # ========================================================
    # Summaries
    # ========================================================

    (
        method_summary,
        direction_summary,
    ) = build_method_summaries(
        merged
    )


    (
        importance_enriched,
        feature_summary,
    ) = build_feature_summary(
        importance
    )


    importance_enriched.to_csv(
        XAI_DIR
        / "global_feature_importance_normalized.csv",
        index=False,
    )


    # ========================================================
    # Plot datasets
    # ========================================================

    build_plot_datasets(
        merged
    )


    # ========================================================
    # Negative adaptation summaries
    # ========================================================

    negative_summary = (
        build_negative_adaptation_summary(
            merged
        )
    )


    # ========================================================
    # Associations
    # ========================================================

    (
        association,
        direction_association,
    ) = build_association_statistics(
        merged
    )


    # ========================================================
    # Figures
    # ========================================================

    build_figures(
        merged=
            merged,

        feature_summary=
            feature_summary,
    )


    # ========================================================
    # Analysis manifest
    # ========================================================

    manifest = {
        "analysis_version":
            ANALYSIS_VERSION,

        "bootstrap_iterations":
            BOOTSTRAP_ITERATIONS,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "canonical_effect_tolerance":
            EFFECT_TOLERANCE,

        "negative_adaptation_auc_definition":
            (
                "delta_roc_auc < -1e-8"
            ),

        "brier_worsened_definition":
            (
                "brier_improvement < -1e-8"
            ),

        "primary_explanation_comparison":
            (
                "paired source-only vs adapted "
                "SHAP using identical fold background"
            ),

        "feature_importance_cross_domain":
            (
                "normalized within "
                "source-target-seed-model job"
            ),

        "association_analysis":
            (
                "exploratory; "
                "no multiplicity correction"
            ),

        "direction_level_unit":
            (
                "10 source seeds averaged "
                "within source-target-method"
            ),

        "direction_level_n_per_method":
            12,

        "causal_interpretation":
            False,

        "negative_adaptation_summary_type":
            (
                "descriptive post-hoc analysis"
            ),
    }


    with open(
        XAI_DIR
        / "xai_analysis_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )


    # ========================================================
    # Console
    # ========================================================

    print_audit(
        merged=
            merged,

        method_summary=
            method_summary,

        feature_summary=
            feature_summary,

        association=
            association,
    )


if __name__ == "__main__":
    main()