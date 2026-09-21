from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SINGLE_FILE = (
    ROOT
    / "results"
    / "cross_source"
    / "cross_source_patient_predictions.csv"
)

SINGLE_DELTA_FILE = (
    ROOT
    / "results"
    / "cross_source"
    / "cross_source_multiseed_deltas.csv"
)

DLAR_FILE = (
    ROOT
    / "results"
    / "dlar_lcl"
    / "dlar_lcl_patient_predictions.csv"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "final"
    / "diagnostics"
)

FIGURE_DIR = (
    ROOT
    / "figures"
    / "final"
    / "diagnostics"
)


# ============================================================
# Frozen diagnostic settings
# ============================================================

THRESHOLD = 0.5

N_CALIBRATION_BINS = 10

CURVE_GRID_SIZE = 201


# IMPORTANT:
# Patient-level probabilities were serialized to CSV.
# Tiny metric-level differences on the order of 1e-8 can
# therefore arise purely from floating-point serialization.
#
# Earlier frozen probability reproduction tolerance was 1e-6,
# so we use the same strict-but-realistic tolerance here.
METRIC_TOLERANCE = 1e-6


DOMAINS = [
    "cleveland",
    "hungary",
    "switzerland",
    "va_long_beach",
]


DOMAIN_LABELS = {
    "cleveland": "Cleveland",
    "hungary": "Hungary",
    "switzerland": "Switzerland",
    "va_long_beach": "VA Long Beach",
}


SINGLE_METHODS = [
    "source_only",
    "pseudo_label_sfda",
    "entropy_sfda",
    "reliability_gated_sfda",
    "conservative_candidate",
]


SINGLE_METHOD_LABELS = {
    "source_only":
        "Source-only",

    "pseudo_label_sfda":
        "Pseudo-label",

    "entropy_sfda":
        "Entropy",

    "reliability_gated_sfda":
        "Reliability-gated",

    "conservative_candidate":
        "Conservative",
}


DLAR_METHODS = [
    "dual_head_source_only",
    "dlar_lcl",
]


DLAR_METHOD_LABELS = {
    "dual_head_source_only":
        "Dual-head source-only",

    "dlar_lcl":
        "DLAR-LCL",
}


# ============================================================
# General helpers
# ============================================================

def ensure_dirs():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_figure(
    fig,
    stem,
):

    png_path = (
        FIGURE_DIR
        / f"{stem}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"{stem}.pdf"
    )


    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )


    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )


    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print(
        "Saved:",
        png_path
    )


def build_direction_order():

    directions = []


    for source in DOMAINS:

        for target in DOMAINS:

            if source == target:
                continue

            directions.append(
                (
                    source,
                    target,
                )
            )


    return directions


DIRECTIONS = build_direction_order()


def direction_name(
    source,
    target,
):

    return (
        f"{DOMAIN_LABELS[source]}"
        f" -> "
        f"{DOMAIN_LABELS[target]}"
    )


# ============================================================
# Patient-file validation
# ============================================================

def validate_patient_file(
    df,
    methods,
    name,
):

    required = {
        "source",
        "target",
        "seed",
        "row_id",
        "y_true",
        *methods,
    }


    missing = (
        required
        -
        set(
            df.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"{name}: missing columns "
            f"{sorted(missing)}"
        )


    # --------------------------------------------------------
    # y_true must be stable across source seeds.
    # --------------------------------------------------------

    y_instability = (
        df
        .groupby(
            [
                "source",
                "target",
                "row_id",
            ]
        )[
            "y_true"
        ]
        .nunique()
    )


    if (
        y_instability
        > 1
    ).any():

        raise RuntimeError(
            f"{name}: y_true differs "
            "across seeds."
        )


    # --------------------------------------------------------
    # Probability sanity
    # --------------------------------------------------------

    for method in methods:

        probability = (
            df[
                method
            ]
            .to_numpy(
                dtype=float
            )
        )


        if not np.all(
            np.isfinite(
                probability
            )
        ):

            raise RuntimeError(
                f"{name}: non-finite "
                f"probabilities in {method}"
            )


        if (
            np.any(
                probability < 0
            )
            or
            np.any(
                probability > 1
            )
        ):

            raise RuntimeError(
                f"{name}: probability "
                f"outside [0, 1] in {method}"
            )


# ============================================================
# Basic metrics
# ============================================================

def specificity_score(
    y_true,
    y_pred,
):

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )


    denominator = (
        tn
        +
        fp
    )


    if denominator == 0:
        return np.nan


    return float(
        tn
        /
        denominator
    )


def sensitivity_score(
    y_true,
    y_pred,
):

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )


    denominator = (
        tp
        +
        fn
    )


    if denominator == 0:
        return np.nan


    return float(
        tp
        /
        denominator
    )


def expected_calibration_error(
    y_true,
    probability,
    n_bins=N_CALIBRATION_BINS,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )


    edges = np.linspace(
        0,
        1,
        n_bins + 1,
    )


    bin_index = np.digitize(
        probability,
        edges[
            1:-1
        ],
        right=False,
    )


    ece = 0.0


    for b in range(
        n_bins
    ):

        mask = (
            bin_index
            == b
        )


        n = int(
            mask.sum()
        )


        if n == 0:
            continue


        observed = float(
            y_true[
                mask
            ]
            .mean()
        )


        predicted = float(
            probability[
                mask
            ]
            .mean()
        )


        ece += (
            n
            /
            len(
                y_true
            )
        ) * abs(
            observed
            -
            predicted
        )


    return float(
        ece
    )


def compute_seed_metrics(
    df,
    methods,
    architecture,
):

    rows = []


    for (
        source,
        target,
        seed,
    ), group in df.groupby(
        [
            "source",
            "target",
            "seed",
        ],
        sort=True,
    ):

        y_true = (
            group[
                "y_true"
            ]
            .to_numpy(
                dtype=int
            )
        )


        for method in methods:

            probability = (
                group[
                    method
                ]
                .to_numpy(
                    dtype=float
                )
            )


            y_pred = (
                probability
                >= THRESHOLD
            ).astype(
                int
            )


            rows.append(
                {
                    "architecture":
                        architecture,

                    "source":
                        source,

                    "target":
                        target,

                    "seed":
                        int(
                            seed
                        ),

                    "method":
                        method,

                    "n":
                        int(
                            len(
                                y_true
                            )
                        ),

                    "prevalence":
                        float(
                            y_true.mean()
                        ),

                    "roc_auc":
                        float(
                            roc_auc_score(
                                y_true,
                                probability,
                            )
                        ),

                    "average_precision":
                        float(
                            average_precision_score(
                                y_true,
                                probability,
                            )
                        ),

                    "brier":
                        float(
                            brier_score_loss(
                                y_true,
                                probability,
                            )
                        ),

                    "ece10":
                        expected_calibration_error(
                            y_true,
                            probability,
                        ),

                    "accuracy":
                        float(
                            accuracy_score(
                                y_true,
                                y_pred,
                            )
                        ),

                    "balanced_accuracy":
                        float(
                            balanced_accuracy_score(
                                y_true,
                                y_pred,
                            )
                        ),

                    "sensitivity":
                        sensitivity_score(
                            y_true,
                            y_pred,
                        ),

                    "specificity":
                        specificity_score(
                            y_true,
                            y_pred,
                        ),

                    "predicted_positive_rate":
                        float(
                            y_pred.mean()
                        ),

                    "mean_probability":
                        float(
                            probability.mean()
                        ),
                }
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# Frozen single-head metric reproduction
# ============================================================

def verify_single_metrics(
    metrics,
):

    frozen = pd.read_csv(
        SINGLE_DELTA_FILE
    )


    frozen = (
        frozen[
            frozen[
                "method"
            ].isin(
                SINGLE_METHODS[
                    1:
                ]
            )
        ]
        .copy()
    )


    adapted = (
        metrics[
            metrics[
                "method"
            ]
            != "source_only"
        ]
        .copy()
    )


    check = adapted.merge(
        frozen[
            [
                "source",
                "target",
                "seed",
                "method",
                "roc_auc",
                "brier",
                "source_roc_auc",
                "source_brier",
            ]
        ],

        on=[
            "source",
            "target",
            "seed",
            "method",
        ],

        how="inner",

        suffixes=(
            "_recomputed",
            "_frozen",
        ),

        validate="one_to_one",
    )


    if (
        len(
            check
        )
        != 480
    ):

        raise RuntimeError(
            "Metric verification expected "
            f"480 adapted rows, "
            f"found {len(check)}."
        )


    # --------------------------------------------------------
    # Adapted metrics
    # --------------------------------------------------------

    check[
        "adapted_auc_diff"
    ] = np.abs(
        check[
            "roc_auc_recomputed"
        ]
        -
        check[
            "roc_auc_frozen"
        ]
    )


    check[
        "adapted_brier_diff"
    ] = np.abs(
        check[
            "brier_recomputed"
        ]
        -
        check[
            "brier_frozen"
        ]
    )


    # --------------------------------------------------------
    # Source-only metrics
    # --------------------------------------------------------

    source_metrics = (
        metrics[
            metrics[
                "method"
            ]
            == "source_only"
        ][
            [
                "source",
                "target",
                "seed",
                "roc_auc",
                "brier",
            ]
        ]
        .rename(
            columns={
                "roc_auc":
                    "source_auc_recomputed",

                "brier":
                    "source_brier_recomputed",
            }
        )
    )


    source_reference = (
        frozen[
            [
                "source",
                "target",
                "seed",
                "source_roc_auc",
                "source_brier",
            ]
        ]
        .drop_duplicates(
            [
                "source",
                "target",
                "seed",
            ]
        )
    )


    source_check = (
        source_metrics.merge(
            source_reference,

            on=[
                "source",
                "target",
                "seed",
            ],

            how="inner",

            validate="one_to_one",
        )
    )


    if (
        len(
            source_check
        )
        != 120
    ):

        raise RuntimeError(
            "Metric verification expected "
            f"120 source-only rows, "
            f"found {len(source_check)}."
        )


    source_check[
        "source_auc_diff"
    ] = np.abs(
        source_check[
            "source_auc_recomputed"
        ]
        -
        source_check[
            "source_roc_auc"
        ]
    )


    source_check[
        "source_brier_diff"
    ] = np.abs(
        source_check[
            "source_brier_recomputed"
        ]
        -
        source_check[
            "source_brier"
        ]
    )


    # --------------------------------------------------------
    # Detailed audit
    # --------------------------------------------------------

    max_adapted_auc_diff = float(
        check[
            "adapted_auc_diff"
        ]
        .max()
    )


    max_adapted_brier_diff = float(
        check[
            "adapted_brier_diff"
        ]
        .max()
    )


    max_source_auc_diff = float(
        source_check[
            "source_auc_diff"
        ]
        .max()
    )


    max_source_brier_diff = float(
        source_check[
            "source_brier_diff"
        ]
        .max()
    )


    worst = max(
        max_adapted_auc_diff,
        max_adapted_brier_diff,
        max_source_auc_diff,
        max_source_brier_diff,
    )


    print(
        "Max adapted AUROC diff:",
        f"{max_adapted_auc_diff:.12f}"
    )


    print(
        "Max adapted Brier diff:",
        f"{max_adapted_brier_diff:.12f}"
    )


    print(
        "Max source AUROC diff:",
        f"{max_source_auc_diff:.12f}"
    )


    print(
        "Max source Brier diff:",
        f"{max_source_brier_diff:.12f}"
    )


    print(
        "Worst frozen metric reproduction diff:",
        f"{worst:.12f}"
    )


    print(
        "Metric reproduction tolerance:",
        f"{METRIC_TOLERANCE:.12f}"
    )


    # --------------------------------------------------------
    # Save audit
    # --------------------------------------------------------

    check.to_csv(
        RESULT_DIR
        / "single_head_metric_reproduction_adapted.csv",
        index=False,
    )


    source_check.to_csv(
        RESULT_DIR
        / "single_head_metric_reproduction_source.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Strict frozen-result check
    # --------------------------------------------------------

    if (
        worst
        >
        METRIC_TOLERANCE
    ):

        raise RuntimeError(
            "Frozen metric reproduction failed: "
            f"worst={worst:.12g}, "
            f"tolerance={METRIC_TOLERANCE:.12g}"
        )


    print(
        "Frozen metric reproduction: PASS"
    )


# ============================================================
# ROC curve summary
# ============================================================

def mean_roc_over_seeds(
    group,
    method,
):

    grid = np.linspace(
        0,
        1,
        CURVE_GRID_SIZE,
    )


    curves = []
    aucs = []


    for (
        seed,
        seed_df,
    ) in group.groupby(
        "seed",
        sort=True,
    ):

        y_true = (
            seed_df[
                "y_true"
            ]
            .to_numpy(
                dtype=int
            )
        )


        probability = (
            seed_df[
                method
            ]
            .to_numpy(
                dtype=float
            )
        )


        fpr, tpr, _ = roc_curve(
            y_true,
            probability,
        )


        interpolated = np.interp(
            grid,
            fpr,
            tpr,
        )


        interpolated[
            0
        ] = 0.0


        interpolated[
            -1
        ] = 1.0


        curves.append(
            interpolated
        )


        aucs.append(
            roc_auc_score(
                y_true,
                probability,
            )
        )


    curves = np.asarray(
        curves,
        dtype=float,
    )


    return {
        "x":
            grid,

        "mean":
            curves.mean(
                axis=0
            ),

        "sd":
            curves.std(
                axis=0,
                ddof=1,
            ),

        "metric_mean":
            float(
                np.mean(
                    aucs
                )
            ),
    }


# ============================================================
# PR curve summary
# ============================================================

def mean_pr_over_seeds(
    group,
    method,
):

    recall_grid = np.linspace(
        0,
        1,
        CURVE_GRID_SIZE,
    )


    curves = []
    average_precisions = []


    for (
        seed,
        seed_df,
    ) in group.groupby(
        "seed",
        sort=True,
    ):

        y_true = (
            seed_df[
                "y_true"
            ]
            .to_numpy(
                dtype=int
            )
        )


        probability = (
            seed_df[
                method
            ]
            .to_numpy(
                dtype=float
            )
        )


        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                probability,
            )
        )


        # sklearn returns recall in descending order.
        recall_ascending = (
            recall[
                ::-1
            ]
        )


        precision_ascending = (
            precision[
                ::-1
            ]
        )


        interpolated = np.interp(
            recall_grid,
            recall_ascending,
            precision_ascending,
        )


        curves.append(
            interpolated
        )


        average_precisions.append(
            average_precision_score(
                y_true,
                probability,
            )
        )


    curves = np.asarray(
        curves,
        dtype=float,
    )


    return {
        "x":
            recall_grid,

        "mean":
            curves.mean(
                axis=0
            ),

        "sd":
            curves.std(
                axis=0,
                ddof=1,
            ),

        "metric_mean":
            float(
                np.mean(
                    average_precisions
                )
            ),
    }


# ============================================================
# Calibration curves
# ============================================================

def seed_calibration_points(
    y_true,
    probability,
):

    edges = np.linspace(
        0,
        1,
        N_CALIBRATION_BINS + 1,
    )


    bin_index = np.digitize(
        probability,
        edges[
            1:-1
        ],
        right=False,
    )


    mean_probability = np.full(
        N_CALIBRATION_BINS,
        np.nan,
    )


    observed_rate = np.full(
        N_CALIBRATION_BINS,
        np.nan,
    )


    count = np.zeros(
        N_CALIBRATION_BINS,
        dtype=int,
    )


    for b in range(
        N_CALIBRATION_BINS
    ):

        mask = (
            bin_index
            == b
        )


        if not mask.any():
            continue


        count[
            b
        ] = int(
            mask.sum()
        )


        mean_probability[
            b
        ] = float(
            probability[
                mask
            ]
            .mean()
        )


        observed_rate[
            b
        ] = float(
            y_true[
                mask
            ]
            .mean()
        )


    return (
        mean_probability,
        observed_rate,
        count,
    )


def mean_calibration_over_seeds(
    group,
    method,
):

    predicted = []
    observed = []
    counts = []
    eces = []


    for (
        seed,
        seed_df,
    ) in group.groupby(
        "seed",
        sort=True,
    ):

        y_true = (
            seed_df[
                "y_true"
            ]
            .to_numpy(
                dtype=int
            )
        )


        probability = (
            seed_df[
                method
            ]
            .to_numpy(
                dtype=float
            )
        )


        (
            mean_probability,
            observed_rate,
            count,
        ) = seed_calibration_points(
            y_true,
            probability,
        )


        predicted.append(
            mean_probability
        )


        observed.append(
            observed_rate
        )


        counts.append(
            count
        )


        eces.append(
            expected_calibration_error(
                y_true,
                probability,
            )
        )


    predicted = np.asarray(
        predicted,
        dtype=float,
    )


    observed = np.asarray(
        observed,
        dtype=float,
    )


    counts = np.asarray(
        counts,
        dtype=int,
    )


    # Suppress warnings when one bin is empty for every seed.
    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )


        mean_predicted = np.nanmean(
            predicted,
            axis=0,
        )


        mean_observed = np.nanmean(
            observed,
            axis=0,
        )


    occupancy = (
        counts.sum(
            axis=0
        )
    )


    valid = (
        occupancy
        > 0
    )


    return {
        "predicted":
            mean_predicted[
                valid
            ],

        "observed":
            mean_observed[
                valid
            ],

        "ece_mean":
            float(
                np.mean(
                    eces
                )
            ),
    }


# ============================================================
# Direction title
# ============================================================

def direction_title_from_group(
    source,
    target,
    group,
):

    first_seed_number = int(
        group[
            "seed"
        ]
        .min()
    )


    first_seed = (
        group[
            group[
                "seed"
            ]
            == first_seed_number
        ]
    )


    y_true = (
        first_seed[
            "y_true"
        ]
        .to_numpy(
            dtype=int
        )
    )


    return (
        f"{direction_name(source, target)}\n"
        f"n={len(y_true)}, "
        f"prev={y_true.mean():.3f}"
    )


# ============================================================
# ROC grid
# ============================================================

def plot_roc_grid(
    df,
    methods,
    labels,
    stem,
    title,
):

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(
            16,
            16,
        ),
        sharex=True,
        sharey=True,
    )


    axes = axes.ravel()


    legend_handles = None
    legend_labels = None


    for (
        ax,
        (
            source,
            target,
        ),
    ) in zip(
        axes,
        DIRECTIONS,
    ):

        group = (
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
        )


        for method in methods:

            result = (
                mean_roc_over_seeds(
                    group,
                    method,
                )
            )


            ax.plot(
                result[
                    "x"
                ],
                result[
                    "mean"
                ],
                linewidth=1.6,
                label=
                    labels[
                        method
                    ],
            )


        ax.plot(
            [
                0,
                1,
            ],
            [
                0,
                1,
            ],
            linestyle="--",
            linewidth=1,
        )


        ax.set_title(
            direction_title_from_group(
                source,
                target,
                group,
            ),
            fontsize=10,
        )


        ax.set_xlim(
            0,
            1,
        )


        ax.set_ylim(
            0,
            1.02,
        )


        ax.grid(
            alpha=0.2
        )


        if (
            legend_handles
            is None
        ):

            (
                legend_handles,
                legend_labels,
            ) = (
                ax
                .get_legend_handles_labels()
            )


    for row in range(
        4
    ):

        axes[
            row
            *
            3
        ].set_ylabel(
            "True positive rate"
        )


    for col in range(
        3
    ):

        axes[
            9
            +
            col
        ].set_xlabel(
            "False positive rate"
        )


    fig.suptitle(
        title,
        fontsize=15,
    )


    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=
            min(
                len(
                    methods
                ),
                5,
            ),
        frameon=False,
    )


    fig.subplots_adjust(
        bottom=0.08
    )


    save_figure(
        fig,
        stem,
    )


# ============================================================
# PR grid
# ============================================================

def plot_pr_grid(
    df,
    methods,
    labels,
    stem,
    title,
):

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(
            16,
            16,
        ),
        sharex=True,
        sharey=True,
    )


    axes = axes.ravel()


    legend_handles = None
    legend_labels = None


    for (
        ax,
        (
            source,
            target,
        ),
    ) in zip(
        axes,
        DIRECTIONS,
    ):

        group = (
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
        )


        first_seed_number = int(
            group[
                "seed"
            ]
            .min()
        )


        first_seed = (
            group[
                group[
                    "seed"
                ]
                == first_seed_number
            ]
        )


        prevalence = float(
            first_seed[
                "y_true"
            ]
            .mean()
        )


        for method in methods:

            result = (
                mean_pr_over_seeds(
                    group,
                    method,
                )
            )


            ax.plot(
                result[
                    "x"
                ],
                result[
                    "mean"
                ],
                linewidth=1.6,
                label=
                    labels[
                        method
                    ],
            )


        ax.axhline(
            prevalence,
            linestyle="--",
            linewidth=1,
        )


        ax.set_title(
            direction_title_from_group(
                source,
                target,
                group,
            ),
            fontsize=10,
        )


        ax.set_xlim(
            0,
            1,
        )


        ax.set_ylim(
            0,
            1.02,
        )


        ax.grid(
            alpha=0.2
        )


        if (
            legend_handles
            is None
        ):

            (
                legend_handles,
                legend_labels,
            ) = (
                ax
                .get_legend_handles_labels()
            )


    for row in range(
        4
    ):

        axes[
            row
            *
            3
        ].set_ylabel(
            "Precision"
        )


    for col in range(
        3
    ):

        axes[
            9
            +
            col
        ].set_xlabel(
            "Recall"
        )


    fig.suptitle(
        title,
        fontsize=15,
    )


    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=
            min(
                len(
                    methods
                ),
                5,
            ),
        frameon=False,
    )


    fig.subplots_adjust(
        bottom=0.08
    )


    save_figure(
        fig,
        stem,
    )


# ============================================================
# Calibration grid
# ============================================================

def plot_calibration_grid(
    df,
    methods,
    labels,
    stem,
    title,
):

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(
            16,
            16,
        ),
        sharex=True,
        sharey=True,
    )


    axes = axes.ravel()


    legend_handles = None
    legend_labels = None


    for (
        ax,
        (
            source,
            target,
        ),
    ) in zip(
        axes,
        DIRECTIONS,
    ):

        group = (
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
        )


        ax.plot(
            [
                0,
                1,
            ],
            [
                0,
                1,
            ],
            linestyle="--",
            linewidth=1,
        )


        for method in methods:

            result = (
                mean_calibration_over_seeds(
                    group,
                    method,
                )
            )


            ax.plot(
                result[
                    "predicted"
                ],
                result[
                    "observed"
                ],
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=
                    labels[
                        method
                    ],
            )


        ax.set_title(
            direction_title_from_group(
                source,
                target,
                group,
            ),
            fontsize=10,
        )


        ax.set_xlim(
            0,
            1,
        )


        ax.set_ylim(
            0,
            1,
        )


        ax.grid(
            alpha=0.2
        )


        if (
            legend_handles
            is None
        ):

            (
                legend_handles,
                legend_labels,
            ) = (
                ax
                .get_legend_handles_labels()
            )


    for row in range(
        4
    ):

        axes[
            row
            *
            3
        ].set_ylabel(
            "Observed event rate"
        )


    for col in range(
        3
    ):

        axes[
            9
            +
            col
        ].set_xlabel(
            "Mean predicted probability"
        )


    fig.suptitle(
        title,
        fontsize=15,
    )


    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=
            min(
                len(
                    methods
                ),
                5,
            ),
        frameon=False,
    )


    fig.subplots_adjust(
        bottom=0.08
    )


    save_figure(
        fig,
        stem,
    )


# ============================================================
# Confusion matrix
# ============================================================

def mean_normalized_confusion(
    group,
    method,
):

    matrices = []


    for (
        seed,
        seed_df,
    ) in group.groupby(
        "seed",
        sort=True,
    ):

        y_true = (
            seed_df[
                "y_true"
            ]
            .to_numpy(
                dtype=int
            )
        )


        probability = (
            seed_df[
                method
            ]
            .to_numpy(
                dtype=float
            )
        )


        y_pred = (
            probability
            >= THRESHOLD
        ).astype(
            int
        )


        cm = (
            confusion_matrix(
                y_true,
                y_pred,
                labels=[
                    0,
                    1,
                ],
            )
            .astype(
                float
            )
        )


        row_sum = (
            cm.sum(
                axis=1,
                keepdims=True,
            )
        )


        normalized = np.zeros_like(
            cm,
            dtype=float,
        )


        np.divide(
            cm,
            row_sum,
            out=normalized,
            where=
                row_sum
                != 0,
        )


        matrices.append(
            normalized
        )


    return (
        np.mean(
            np.asarray(
                matrices,
                dtype=float,
            ),
            axis=0,
        )
    )


def plot_confusion_grids(
    df,
    methods,
    labels,
    prefix,
):

    for method in methods:

        fig, axes = plt.subplots(
            4,
            3,
            figsize=(
                13,
                14,
            ),
        )


        axes = axes.ravel()


        image = None


        for (
            ax,
            (
                source,
                target,
            ),
        ) in zip(
            axes,
            DIRECTIONS,
        ):

            group = (
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
            )


            cm = (
                mean_normalized_confusion(
                    group,
                    method,
                )
            )


            image = ax.imshow(
                cm,
                vmin=0,
                vmax=1,
            )


            for i in range(
                2
            ):

                for j in range(
                    2
                ):

                    ax.text(
                        j,
                        i,
                        f"{cm[i, j]:.2f}",
                        ha="center",
                        va="center",
                    )


            ax.set_xticks(
                [
                    0,
                    1,
                ],
                labels=[
                    "Pred 0",
                    "Pred 1",
                ],
            )


            ax.set_yticks(
                [
                    0,
                    1,
                ],
                labels=[
                    "True 0",
                    "True 1",
                ],
            )


            ax.set_title(
                direction_name(
                    source,
                    target,
                ),
                fontsize=9,
            )


        fig.colorbar(
            image,
            ax=
                axes.tolist(),
            fraction=0.02,
            pad=0.02,
            label=
                (
                    "Mean row-normalized "
                    "proportion"
                ),
        )


        fig.suptitle(
            (
                "Mean confusion matrices "
                "across seeds: "
                f"{labels[method]}"
            ),
            fontsize=14,
        )


        save_figure(
            fig,
            (
                f"{prefix}_confusion_"
                f"{method}"
            ),
        )


# ============================================================
# Probability distributions
# ============================================================

def patient_mean_probabilities(
    group,
    method,
):

    return (
        group
        .groupby(
            [
                "row_id",
                "y_true",
            ],
            as_index=False,
        )[
            method
        ]
        .mean()
    )


def plot_probability_distributions(
    df,
    source_method,
    adapted_methods,
    labels,
    prefix,
):

    bins = np.linspace(
        0,
        1,
        21,
    )


    for method in adapted_methods:

        fig, axes = plt.subplots(
            4,
            3,
            figsize=(
                16,
                15,
            ),
            sharex=True,
            sharey=False,
        )


        axes = axes.ravel()


        for (
            ax,
            (
                source,
                target,
            ),
        ) in zip(
            axes,
            DIRECTIONS,
        ):

            group = (
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
            )


            source_probability = (
                patient_mean_probabilities(
                    group,
                    source_method,
                )[
                    source_method
                ]
                .to_numpy(
                    dtype=float
                )
            )


            adapted_probability = (
                patient_mean_probabilities(
                    group,
                    method,
                )[
                    method
                ]
                .to_numpy(
                    dtype=float
                )
            )


            ax.hist(
                source_probability,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.6,
                label=
                    labels[
                        source_method
                    ],
            )


            ax.hist(
                adapted_probability,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.6,
                label=
                    labels[
                        method
                    ],
            )


            ax.axvline(
                THRESHOLD,
                linestyle="--",
                linewidth=1,
            )


            ax.set_title(
                direction_name(
                    source,
                    target,
                ),
                fontsize=9,
            )


            ax.set_xlim(
                0,
                1,
            )


            ax.grid(
                alpha=0.15
            )


        for row in range(
            4
        ):

            axes[
                row
                *
                3
            ].set_ylabel(
                "Density"
            )


        for col in range(
            3
        ):

            axes[
                9
                +
                col
            ].set_xlabel(
                "Across-seed mean probability"
            )


        handles, names = (
            axes[
                0
            ]
            .get_legend_handles_labels()
        )


        fig.legend(
            handles,
            names,
            loc="lower center",
            ncol=2,
            frameon=False,
        )


        fig.suptitle(
            (
                "Probability distributions: "
                f"{labels[source_method]} vs "
                f"{labels[method]}"
            ),
            fontsize=14,
        )


        fig.subplots_adjust(
            bottom=0.07
        )


        save_figure(
            fig,
            (
                f"{prefix}_probability_"
                f"{source_method}_vs_{method}"
            ),
        )


# ============================================================
# Metric boxplots
# ============================================================

def plot_metric_boxplot(
    metrics,
    methods,
    labels,
    metric,
    ylabel,
    stem,
    title,
):

    data = [
        metrics[
            metrics[
                "method"
            ]
            == method
        ][
            metric
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )

        for method
        in methods
    ]


    fig, ax = plt.subplots(
        figsize=(
            9,
            5.5,
        )
    )


    ax.boxplot(
        data,
        tick_labels=[
            labels[
                method
            ]
            for method
            in methods
        ],
        showfliers=False,
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        title
    )


    ax.tick_params(
        axis="x",
        rotation=20,
    )


    ax.grid(
        axis="y",
        alpha=0.2,
    )


    save_figure(
        fig,
        stem,
    )


# ============================================================
# Single-head diagnostic pipeline
# ============================================================

def run_single_head():

    print(
        "\n"
        + "=" * 100
    )


    print(
        "SINGLE-HEAD DIAGNOSTICS"
    )


    print(
        "=" * 100
    )


    df = pd.read_csv(
        SINGLE_FILE
    )


    validate_patient_file(
        df,
        SINGLE_METHODS,
        "single-head",
    )


    metrics = (
        compute_seed_metrics(
            df,
            SINGLE_METHODS,
            "single_head",
        )
    )


    metrics.to_csv(
        RESULT_DIR
        / "single_head_seed_diagnostic_metrics.csv",
        index=False,
    )


    verify_single_metrics(
        metrics
    )


    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    plot_roc_grid(
        df=
            df,

        methods=
            SINGLE_METHODS,

        labels=
            SINGLE_METHOD_LABELS,

        stem=
            "single_head_roc_all_directions",

        title=
            (
                "ROC curves across "
                "cross-hospital directions\n"
                "(mean curve across 10 source seeds)"
            ),
    )


    # --------------------------------------------------------
    # Precision-recall
    # --------------------------------------------------------

    plot_pr_grid(
        df=
            df,

        methods=
            SINGLE_METHODS,

        labels=
            SINGLE_METHOD_LABELS,

        stem=
            "single_head_pr_all_directions",

        title=
            (
                "Precision-recall curves across "
                "cross-hospital directions\n"
                "(mean curve across 10 source seeds)"
            ),
    )


    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    plot_calibration_grid(
        df=
            df,

        methods=
            SINGLE_METHODS,

        labels=
            SINGLE_METHOD_LABELS,

        stem=
            "single_head_calibration_all_directions",

        title=
            (
                "Calibration curves across "
                "cross-hospital directions\n"
                "(10 fixed bins; mean across seeds)"
            ),
    )


    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    plot_confusion_grids(
        df=
            df,

        methods=
            SINGLE_METHODS,

        labels=
            SINGLE_METHOD_LABELS,

        prefix=
            "single_head",
    )


    # --------------------------------------------------------
    # Probability distributions
    # --------------------------------------------------------

    plot_probability_distributions(
        df=
            df,

        source_method=
            "source_only",

        adapted_methods=
            SINGLE_METHODS[
                1:
            ],

        labels=
            SINGLE_METHOD_LABELS,

        prefix=
            "single_head",
    )


    # --------------------------------------------------------
    # ECE
    # --------------------------------------------------------

    plot_metric_boxplot(
        metrics=
            metrics,

        methods=
            SINGLE_METHODS,

        labels=
            SINGLE_METHOD_LABELS,

        metric=
            "ece10",

        ylabel=
            "ECE (10 fixed bins)",

        stem=
            "single_head_ece_by_method",

        title=
            (
                "Calibration error across "
                "seed-direction evaluations"
            ),
    )


    # --------------------------------------------------------
    # Brier
    # --------------------------------------------------------

    plot_metric_boxplot(
        metrics=
            metrics,

        methods=
            SINGLE_METHODS,

        labels=
            SINGLE_METHOD_LABELS,

        metric=
            "brier",

        ylabel=
            "Brier score (lower is better)",

        stem=
            "single_head_brier_by_method",

        title=
            (
                "Brier score across "
                "seed-direction evaluations"
            ),
    )


    print(
        "Single-head diagnostic rows:",
        len(
            metrics
        )
    )


    if (
        len(
            metrics
        )
        != 600
    ):

        raise RuntimeError(
            "Expected 600 single-head "
            "diagnostic rows."
        )


# ============================================================
# DLAR-LCL diagnostic pipeline
# ============================================================

def run_dlar():

    if not DLAR_FILE.exists():

        print(
            "\nDLAR-LCL patient prediction file "
            "not found; skipping DLAR diagnostics:"
        )

        print(
            DLAR_FILE
        )

        return None


    print(
        "\n"
        + "=" * 100
    )


    print(
        "DLAR-LCL DIAGNOSTICS"
    )


    print(
        "=" * 100
    )


    df = pd.read_csv(
        DLAR_FILE
    )


    validate_patient_file(
        df,
        DLAR_METHODS,
        "DLAR-LCL",
    )


    metrics = (
        compute_seed_metrics(
            df,
            DLAR_METHODS,
            "dual_head",
        )
    )


    metrics.to_csv(
        RESULT_DIR
        / "dlar_lcl_seed_diagnostic_metrics.csv",
        index=False,
    )


    plot_roc_grid(
        df=
            df,

        methods=
            DLAR_METHODS,

        labels=
            DLAR_METHOD_LABELS,

        stem=
            "dlar_lcl_roc_all_directions",

        title=
            (
                "DLAR-LCL ROC curves\n"
                "(architecture-matched dual-head comparison)"
            ),
    )


    plot_pr_grid(
        df=
            df,

        methods=
            DLAR_METHODS,

        labels=
            DLAR_METHOD_LABELS,

        stem=
            "dlar_lcl_pr_all_directions",

        title=
            (
                "DLAR-LCL precision-recall curves\n"
                "(architecture-matched dual-head comparison)"
            ),
    )


    plot_calibration_grid(
        df=
            df,

        methods=
            DLAR_METHODS,

        labels=
            DLAR_METHOD_LABELS,

        stem=
            "dlar_lcl_calibration_all_directions",

        title=
            (
                "DLAR-LCL calibration curves\n"
                "(architecture-matched dual-head comparison)"
            ),
    )


    plot_confusion_grids(
        df=
            df,

        methods=
            DLAR_METHODS,

        labels=
            DLAR_METHOD_LABELS,

        prefix=
            "dlar_lcl",
    )


    plot_probability_distributions(
        df=
            df,

        source_method=
            "dual_head_source_only",

        adapted_methods=[
            "dlar_lcl",
        ],

        labels=
            DLAR_METHOD_LABELS,

        prefix=
            "dlar_lcl",
    )


    plot_metric_boxplot(
        metrics=
            metrics,

        methods=
            DLAR_METHODS,

        labels=
            DLAR_METHOD_LABELS,

        metric=
            "ece10",

        ylabel=
            "ECE (10 fixed bins)",

        stem=
            "dlar_lcl_ece",

        title=
            "DLAR-LCL calibration error",
    )


    plot_metric_boxplot(
        metrics=
            metrics,

        methods=
            DLAR_METHODS,

        labels=
            DLAR_METHOD_LABELS,

        metric=
            "brier",

        ylabel=
            "Brier score (lower is better)",

        stem=
            "dlar_lcl_brier",

        title=
            "DLAR-LCL Brier score",
    )


    print(
        "DLAR-LCL diagnostic rows:",
        len(
            metrics
        )
    )


    if (
        len(
            metrics
        )
        != 240
    ):

        raise RuntimeError(
            "Expected 240 DLAR-LCL "
            "diagnostic rows."
        )


    return metrics


# ============================================================
# Manifest
# ============================================================

def write_manifest():

    manifest = {
        "analysis_type":
            "frozen_prediction_diagnostics",

        "model_training_performed":
            False,

        "adaptation_performed":
            False,

        "threshold":
            THRESHOLD,

        "calibration_bins":
            N_CALIBRATION_BINS,

        "curve_grid_size":
            CURVE_GRID_SIZE,

        "metric_reproduction_tolerance":
            METRIC_TOLERANCE,

        "metric_reproduction_note":
            (
                "Metrics were recomputed from serialized "
                "patient-level probability CSVs. "
                "Differences below 1e-6 are treated as "
                "floating-point serialization precision, "
                "not substantive result differences."
            ),

        "roc_pr_curve_summary":
            (
                "Each source seed is evaluated independently. "
                "Curves are interpolated to a common grid and "
                "then averaged across the 10 source seeds. "
                "No probability ensemble is used for ROC/PR."
            ),

        "calibration_curve_summary":
            (
                "Ten fixed probability bins are computed "
                "within each source seed and summarized "
                "across seeds."
            ),

        "confusion_matrix_summary":
            (
                "Row-normalized confusion matrices are "
                "computed independently for each source seed "
                "at threshold 0.5 and averaged across seeds."
            ),

        "probability_distribution_summary":
            (
                "For visualization only, each target patient's "
                "predicted probability is averaged across "
                "the 10 source seeds before plotting."
            ),

        "single_head_methods":
            SINGLE_METHODS,

        "dlar_comparison":
            (
                "DLAR-LCL is compared only against its "
                "architecture-matched dual-head source-only "
                "reference. Absolute single-head and "
                "dual-head predictions are not treated as "
                "direct method comparisons."
            ),

        "switzerland_limitation":
            (
                "Switzerland has very high outcome prevalence "
                "and only eight negative cases. PR curves, "
                "specificity, calibration, and thresholded "
                "diagnostics involving Switzerland should "
                "therefore be interpreted cautiously."
            ),
    }


    with open(
        RESULT_DIR
        / "diagnostic_figure_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )


# ============================================================
# Main
# ============================================================

def main():

    ensure_dirs()


    run_single_head()


    run_dlar()


    write_manifest()


    print(
        "\n"
        + "=" * 100
    )


    print(
        "FINAL DIAGNOSTIC FIGURES COMPLETE"
    )


    print(
        "=" * 100
    )


    print(
        "Results:",
        RESULT_DIR
    )


    print(
        "Figures:",
        FIGURE_DIR
    )


if __name__ == "__main__":

    main()