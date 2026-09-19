import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true,
    y_probability,
    threshold=0.5,
):

    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "sensitivity": sensitivity,

        "specificity": specificity,

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "mcc":
            matthews_corrcoef(
                y_true,
                y_pred,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_probability,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                y_probability,
            ),

        "brier":
            brier_score_loss(
                y_true,
                y_probability,
            ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }