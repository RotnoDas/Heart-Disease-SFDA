from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SINGLE_PREDICTIONS = (
    ROOT
    / "results"
    / "cross_source"
    / "cross_source_patient_predictions.csv"
)

DLAR_PREDICTIONS = (
    ROOT
    / "results"
    / "dlar_lcl"
    / "dlar_lcl_patient_predictions.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "final"
    / "performance_tables"
)


# ============================================================
# Settings
# ============================================================

THRESHOLD = 0.5
ECE_BINS = 10


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
    "source_only": "Source-only",
    "pseudo_label_sfda": "Pseudo-label",
    "entropy_sfda": "Entropy",
    "reliability_gated_sfda": "Reliability-gated",
    "conservative_candidate": "Conservative",
}


DLAR_METHODS = [
    "dual_head_source_only",
    "dlar_lcl",
]


DLAR_METHOD_LABELS = {
    "dual_head_source_only": "Dual-head source-only",
    "dlar_lcl": "DLAR-LCL",
}


PERFORMANCE_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
    "brier",
    "ece10",
    "predicted_positive_rate",
]


CONFUSION_METRICS = [
    "tn",
    "fp",
    "fn",
    "tp",
]


# ============================================================
# Helpers
# ============================================================

def make_output_dir():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def direction_label(
    source,
    target,
):

    return (
        f"{DOMAIN_LABELS[source]}"
        f" -> "
        f"{DOMAIN_LABELS[target]}"
    )


def expected_calibration_error(
    y_true,
    probability,
    n_bins=ECE_BINS,
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
        0.0,
        1.0,
        n_bins + 1,
    )


    bin_id = np.digitize(
        probability,
        edges[1:-1],
        right=False,
    )


    ece = 0.0


    for b in range(
        n_bins
    ):

        mask = (
            bin_id
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
            ].mean()
        )


        predicted = float(
            probability[
                mask
            ].mean()
        )


        ece += (
            n
            /
            len(y_true)
        ) * abs(
            observed
            -
            predicted
        )


    return float(
        ece
    )


# ============================================================
# Validation
# ============================================================

def validate_prediction_file(
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
        set(df.columns)
    )


    if missing:

        raise RuntimeError(
            f"{name}: missing columns: "
            f"{sorted(missing)}"
        )


    directions = (
        df[
            [
                "source",
                "target",
            ]
        ]
        .drop_duplicates()
    )


    if len(directions) != 12:

        raise RuntimeError(
            f"{name}: expected 12 directions, "
            f"found {len(directions)}"
        )


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
                f"{name}: probabilities outside "
                f"[0,1] in {method}"
            )


# ============================================================
# Seed-level metrics
# ============================================================

def calculate_seed_metrics(
    df,
    methods,
    method_labels,
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


            specificity_denominator = (
                tn
                +
                fp
            )


            specificity = (
                tn
                /
                specificity_denominator
                if specificity_denominator > 0
                else np.nan
            )


            row = {
                "architecture":
                    architecture,

                "source":
                    source,

                "target":
                    target,

                "direction":
                    direction_label(
                        source,
                        target,
                    ),

                "seed":
                    int(seed),

                "method":
                    method,

                "method_label":
                    method_labels[
                        method
                    ],

                "n":
                    int(
                        len(y_true)
                    ),

                "prevalence":
                    float(
                        y_true.mean()
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

                "precision":
                    float(
                        precision_score(
                            y_true,
                            y_pred,
                            zero_division=0,
                        )
                    ),

                "recall":
                    float(
                        recall_score(
                            y_true,
                            y_pred,
                            zero_division=0,
                        )
                    ),

                "specificity":
                    float(
                        specificity
                    ),

                "f1":
                    float(
                        f1_score(
                            y_true,
                            y_pred,
                            zero_division=0,
                        )
                    ),

                "mcc":
                    float(
                        matthews_corrcoef(
                            y_true,
                            y_pred,
                        )
                    ),

                "roc_auc":
                    float(
                        roc_auc_score(
                            y_true,
                            probability,
                        )
                    ),

                # Consistent with previous experiment:
                # PR-AUC is Average Precision.
                "pr_auc":
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

                "predicted_positive_rate":
                    float(
                        y_pred.mean()
                    ),

                "mean_probability":
                    float(
                        probability.mean()
                    ),

                "tn":
                    int(tn),

                "fp":
                    int(fp),

                "fn":
                    int(fn),

                "tp":
                    int(tp),
            }


            rows.append(
                row
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# Direction × method aggregation
# ============================================================

def aggregate_by_direction(
    seed_metrics,
):

    metrics = (
        PERFORMANCE_METRICS
        +
        CONFUSION_METRICS
    )


    grouped = (
        seed_metrics
        .groupby(
            [
                "architecture",
                "source",
                "target",
                "direction",
                "method",
                "method_label",
            ],
            sort=True,
        )[metrics]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )


    flattened = []


    for column in grouped.columns:

        if isinstance(
            column,
            tuple,
        ):

            first = str(
                column[0]
            )

            second = str(
                column[1]
            )


            if second == "":
                flattened.append(
                    first
                )

            else:
                flattened.append(
                    f"{first}_{second}"
                )

        else:

            flattened.append(
                str(column)
            )


    grouped.columns = flattened


    seed_counts = (
        seed_metrics
        .groupby(
            [
                "architecture",
                "source",
                "target",
                "direction",
                "method",
                "method_label",
            ]
        )
        .size()
        .reset_index(
            name="n_seeds"
        )
    )


    grouped = grouped.merge(
        seed_counts,

        on=[
            "architecture",
            "source",
            "target",
            "direction",
            "method",
            "method_label",
        ],

        how="left",

        validate="one_to_one",
    )


    return grouped


# ============================================================
# Manuscript-friendly formatted table
# ============================================================

def format_mean_sd(
    mean,
    std,
    decimals=3,
):

    if pd.isna(mean):

        return ""


    if pd.isna(std):

        return (
            f"{mean:.{decimals}f}"
        )


    return (
        f"{mean:.{decimals}f} ± "
        f"{std:.{decimals}f}"
    )


def make_formatted_direction_table(
    aggregated,
):

    output = aggregated[
        [
            "source",
            "target",
            "direction",
            "method",
            "method_label",
            "n_seeds",
        ]
    ].copy()


    for metric in PERFORMANCE_METRICS:

        output[
            metric
        ] = [
            format_mean_sd(
                mean,
                std,
                decimals=3,
            )

            for mean, std in zip(
                aggregated[
                    f"{metric}_mean"
                ],
                aggregated[
                    f"{metric}_std"
                ],
            )
        ]


    return output


# ============================================================
# Confusion matrix summary
# ============================================================

def make_confusion_summary(
    aggregated,
):

    output = aggregated[
        [
            "source",
            "target",
            "direction",
            "method",
            "method_label",
            "n_seeds",
        ]
    ].copy()


    for metric in CONFUSION_METRICS:

        output[
            metric
        ] = [
            format_mean_sd(
                mean,
                std,
                decimals=1,
            )

            for mean, std in zip(
                aggregated[
                    f"{metric}_mean"
                ],
                aggregated[
                    f"{metric}_std"
                ],
            )
        ]


    return output


# ============================================================
# Pair-balanced method summary
# ============================================================

def make_pair_balanced_summary(
    seed_metrics,
):

    # First average the 10 seeds within each direction.
    #
    # Then summarize the 12 directions equally.
    #
    # This prevents large target cohorts from receiving
    # greater weight merely because they contain more patients.

    direction_means = (
        seed_metrics
        .groupby(
            [
                "architecture",
                "source",
                "target",
                "method",
                "method_label",
            ],
            sort=True,
        )[PERFORMANCE_METRICS]
        .mean()
        .reset_index()
    )


    rows = []


    for (
        architecture,
        method,
        method_label,
    ), group in direction_means.groupby(
        [
            "architecture",
            "method",
            "method_label",
        ],
        sort=True,
    ):

        row = {
            "architecture":
                architecture,

            "method":
                method,

            "method_label":
                method_label,

            "n_directions":
                int(
                    len(group)
                ),
        }


        for metric in PERFORMANCE_METRICS:

            values = (
                group[
                    metric
                ]
                .to_numpy(
                    dtype=float
                )
            )


            row[
                f"{metric}_mean"
            ] = float(
                np.mean(values)
            )


            row[
                f"{metric}_sd"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
            )


            row[
                metric
            ] = format_mean_sd(
                np.mean(values),
                np.std(
                    values,
                    ddof=1,
                ),
                decimals=3,
            )


        rows.append(
            row
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# Save outputs
# ============================================================

def save_analysis(
    prediction_file,
    methods,
    method_labels,
    architecture,
    prefix,
    expected_seed_rows,
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        prefix.upper()
    )

    print(
        "=" * 100
    )


    df = pd.read_csv(
        prediction_file
    )


    validate_prediction_file(
        df,
        methods,
        prefix,
    )


    seed_metrics = (
        calculate_seed_metrics(
            df,
            methods,
            method_labels,
            architecture,
        )
    )


    if (
        len(seed_metrics)
        != expected_seed_rows
    ):

        raise RuntimeError(
            f"{prefix}: expected "
            f"{expected_seed_rows} seed-level rows, "
            f"found {len(seed_metrics)}"
        )


    # --------------------------------------------------------
    # Seed-level table
    # --------------------------------------------------------

    seed_path = (
        OUT_DIR
        / f"{prefix}_seed_level_metrics.csv"
    )


    seed_metrics.to_csv(
        seed_path,
        index=False,
        encoding="utf-8-sig",
    )


    # --------------------------------------------------------
    # Direction-level numeric table
    # --------------------------------------------------------

    aggregated = (
        aggregate_by_direction(
            seed_metrics
        )
    )


    numeric_path = (
        OUT_DIR
        / f"{prefix}_direction_performance_numeric.csv"
    )


    aggregated.to_csv(
        numeric_path,
        index=False,
        encoding="utf-8-sig",
    )


    # --------------------------------------------------------
    # Direction-level manuscript table
    # --------------------------------------------------------

    formatted = (
        make_formatted_direction_table(
            aggregated
        )
    )


    formatted_path = (
        OUT_DIR
        / f"{prefix}_direction_performance_formatted.csv"
    )


    formatted.to_csv(
        formatted_path,
        index=False,
        encoding="utf-8-sig",
    )


    # --------------------------------------------------------
    # Confusion matrix counts
    # --------------------------------------------------------

    confusion = (
        make_confusion_summary(
            aggregated
        )
    )


    confusion_path = (
        OUT_DIR
        / f"{prefix}_direction_confusion_summary.csv"
    )


    confusion.to_csv(
        confusion_path,
        index=False,
        encoding="utf-8-sig",
    )


    # --------------------------------------------------------
    # Pair-balanced method-level summary
    # --------------------------------------------------------

    pair_balanced = (
        make_pair_balanced_summary(
            seed_metrics
        )
    )


    pair_path = (
        OUT_DIR
        / f"{prefix}_pair_balanced_method_summary.csv"
    )


    pair_balanced.to_csv(
        pair_path,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "Seed-level rows:",
        len(seed_metrics)
    )


    print(
        "Direction-method rows:",
        len(aggregated)
    )


    print(
        "Directions:",
        seed_metrics[
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
        seed_metrics[
            "method"
        ]
        .nunique()
    )


    print(
        "\nSaved:"
    )


    print(
        seed_path
    )


    print(
        numeric_path
    )


    print(
        formatted_path
    )


    print(
        confusion_path
    )


    print(
        pair_path
    )


    return (
        seed_metrics,
        aggregated,
        pair_balanced,
    )


# ============================================================
# Console summary
# ============================================================

def print_pair_balanced_summary(
    summary,
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PAIR-BALANCED DESCRIPTIVE SUMMARY"
    )

    print(
        "=" * 100
    )


    show_metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier",
    ]


    for _, row in summary.iterrows():

        print(
            f"\n{row['method_label']}"
        )


        for metric in show_metrics:

            print(
                f"  {metric:24s} "
                f"{row[metric]}"
            )


# ============================================================
# Main
# ============================================================

def main():

    make_output_dir()


    # ========================================================
    # Single-head
    # ========================================================

    (
        single_seed,
        single_direction,
        single_summary,
    ) = save_analysis(
        prediction_file=
            SINGLE_PREDICTIONS,

        methods=
            SINGLE_METHODS,

        method_labels=
            SINGLE_METHOD_LABELS,

        architecture=
            "single_head",

        prefix=
            "single_head",

        expected_seed_rows=
            600,
    )


    print_pair_balanced_summary(
        single_summary
    )


    # ========================================================
    # DLAR-LCL
    # ========================================================

    if DLAR_PREDICTIONS.exists():

        (
            dlar_seed,
            dlar_direction,
            dlar_summary,
        ) = save_analysis(
            prediction_file=
                DLAR_PREDICTIONS,

            methods=
                DLAR_METHODS,

            method_labels=
                DLAR_METHOD_LABELS,

            architecture=
                "dual_head",

            prefix=
                "dlar_lcl",

            expected_seed_rows=
                240,
        )


        print_pair_balanced_summary(
            dlar_summary
        )


    else:

        print(
            "\nDLAR-LCL patient prediction file "
            "not found; skipped."
        )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL PERFORMANCE TABLES COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "Output directory:"
    )

    print(
        OUT_DIR
    )


if __name__ == "__main__":

    main()