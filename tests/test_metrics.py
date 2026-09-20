import numpy as np

from heart_sfda.evaluation.metrics import (
    compute_binary_metrics,
)


def test_perfect_predictions():

    y_true = np.array(
        [0, 0, 1, 1]
    )

    probability = np.array(
        [0.01, 0.10, 0.90, 0.99]
    )

    metrics = compute_binary_metrics(
        y_true,
        probability,
    )

    assert np.isclose(
        metrics["roc_auc"],
        1.0,
    )

    assert np.isclose(
        metrics["pr_auc"],
        1.0,
    )

    assert (
        metrics[
            "balanced_accuracy"
        ]
        == 1.0
    )

    assert (
        metrics["brier"]
        < 0.01
    )


def test_reversed_predictions_are_bad():

    y_true = np.array(
        [0, 0, 1, 1]
    )

    probability = np.array(
        [0.90, 0.80, 0.20, 0.10]
    )

    metrics = compute_binary_metrics(
        y_true,
        probability,
    )

    assert np.isclose(
        metrics["roc_auc"],
        0.0,
    )

    assert (
        metrics[
            "balanced_accuracy"
        ]
        == 0.0
    )

    assert (
        metrics["brier"]
        > 0.5
    )