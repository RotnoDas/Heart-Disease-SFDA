import numpy as np

from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


def stratified_bootstrap_indices(
    y,
    rng,
):
    """
    Sample positives and negatives separately.

    Preserves observed class counts and guarantees
    both classes are present.
    """

    y = np.asarray(y)

    negative = np.where(
        y == 0
    )[0]

    positive = np.where(
        y == 1
    )[0]

    negative_sample = rng.choice(
        negative,
        size=len(negative),
        replace=True,
    )

    positive_sample = rng.choice(
        positive,
        size=len(positive),
        replace=True,
    )

    indices = np.concatenate(
        [
            negative_sample,
            positive_sample,
        ]
    )

    rng.shuffle(indices)

    return indices


def metric_values(
    y,
    probability,
):
    prediction = (
        probability >= 0.5
    ).astype(int)

    return {
        "roc_auc":
            roc_auc_score(
                y,
                probability,
            ),

        "brier":
            brier_score_loss(
                y,
                probability,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                prediction,
            ),
    }


def paired_bootstrap_comparison(
    y,
    source_probability,
    adapted_probability,
    iterations=5000,
    seed=42,
):
    """
    Paired bootstrap.

    AUROC delta:
        adapted - source
        positive = improvement

    Brier improvement:
        source - adapted
        positive = improvement

    Balanced accuracy delta:
        adapted - source
        positive = improvement
    """

    y = np.asarray(y)

    source_probability = np.asarray(
        source_probability
    )

    adapted_probability = np.asarray(
        adapted_probability
    )

    source_observed = metric_values(
        y,
        source_probability,
    )

    adapted_observed = metric_values(
        y,
        adapted_probability,
    )

    observed = {
        "delta_roc_auc":
            adapted_observed["roc_auc"]
            -
            source_observed["roc_auc"],

        "brier_improvement":
            source_observed["brier"]
            -
            adapted_observed["brier"],

        "delta_balanced_accuracy":
            adapted_observed[
                "balanced_accuracy"
            ]
            -
            source_observed[
                "balanced_accuracy"
            ],
    }

    rng = np.random.default_rng(
        seed
    )

    bootstrap = {
        "delta_roc_auc": [],
        "brier_improvement": [],
        "delta_balanced_accuracy": [],
    }

    for _ in range(iterations):

        indices = (
            stratified_bootstrap_indices(
                y,
                rng,
            )
        )

        y_boot = y[indices]

        source_boot = (
            source_probability[
                indices
            ]
        )

        adapted_boot = (
            adapted_probability[
                indices
            ]
        )

        source_metrics = metric_values(
            y_boot,
            source_boot,
        )

        adapted_metrics = metric_values(
            y_boot,
            adapted_boot,
        )

        bootstrap[
            "delta_roc_auc"
        ].append(
            adapted_metrics["roc_auc"]
            -
            source_metrics["roc_auc"]
        )

        bootstrap[
            "brier_improvement"
        ].append(
            source_metrics["brier"]
            -
            adapted_metrics["brier"]
        )

        bootstrap[
            "delta_balanced_accuracy"
        ].append(
            adapted_metrics[
                "balanced_accuracy"
            ]
            -
            source_metrics[
                "balanced_accuracy"
            ]
        )

    results = {}

    for metric, values in (
        bootstrap.items()
    ):

        values = np.asarray(
            values
        )

        results[metric] = {
            "observed":
                observed[metric],

            "ci_lower":
                float(
                    np.percentile(
                        values,
                        2.5,
                    )
                ),

            "ci_upper":
                float(
                    np.percentile(
                        values,
                        97.5,
                    )
                ),
        }

    return results