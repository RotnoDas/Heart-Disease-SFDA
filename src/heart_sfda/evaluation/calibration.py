import numpy as np

from scipy.optimize import minimize
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)


EPS = 1e-6


def expected_calibration_error(
    y_true,
    probability,
    n_bins=10,
):
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_ids = np.digitize(
        probability,
        bins[1:-1],
        right=False,
    )

    ece = 0.0

    for bin_id in range(n_bins):

        mask = (
            bin_ids == bin_id
        )

        n_bin = int(mask.sum())

        if n_bin == 0:
            continue

        observed_frequency = float(
            y_true[mask].mean()
        )

        mean_probability = float(
            probability[mask].mean()
        )

        ece += (
            n_bin / len(y_true)
        ) * abs(
            observed_frequency
            - mean_probability
        )

    return float(ece)


def probability_to_logit(
    probability,
):
    probability = np.clip(
        np.asarray(probability),
        EPS,
        1.0 - EPS,
    )

    return np.log(
        probability
        / (1.0 - probability)
    )


def calibration_intercept_slope(
    y_true,
    probability,
):
    """
    Fits:

    logit(P(Y=1)) =
        intercept + slope * logit(predicted_probability)

    Ideal:
        intercept = 0
        slope = 1
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    predicted_logit = (
        probability_to_logit(
            probability
        )
    )


    def negative_log_likelihood(
        parameters,
    ):
        intercept = parameters[0]
        slope = parameters[1]

        linear_predictor = (
            intercept
            +
            slope
            * predicted_logit
        )

        predicted = (
            1.0
            /
            (
                1.0
                +
                np.exp(
                    -np.clip(
                        linear_predictor,
                        -30,
                        30,
                    )
                )
            )
        )

        predicted = np.clip(
            predicted,
            EPS,
            1.0 - EPS,
        )

        nll = -np.sum(
            y_true
            * np.log(predicted)
            +
            (1.0 - y_true)
            * np.log(
                1.0 - predicted
            )
        )

        return float(nll)


    result = minimize(
        negative_log_likelihood,
        x0=np.array(
            [0.0, 1.0]
        ),
        method="BFGS",
    )


    return {
        "calibration_intercept":
            float(
                result.x[0]
            ),

        "calibration_slope":
            float(
                result.x[1]
            ),

        "calibration_fit_success":
            bool(
                result.success
            ),
    }


def compute_calibration_metrics(
    y_true,
    probability,
):
    probability = np.clip(
        np.asarray(probability),
        EPS,
        1.0 - EPS,
    )

    calibration_fit = (
        calibration_intercept_slope(
            y_true,
            probability,
        )
    )

    return {
        "brier":
            float(
                brier_score_loss(
                    y_true,
                    probability,
                )
            ),

        "log_loss":
            float(
                log_loss(
                    y_true,
                    probability,
                    labels=[0, 1],
                )
            ),

        "ece_5":
            expected_calibration_error(
                y_true,
                probability,
                n_bins=5,
            ),

        "ece_10":
            expected_calibration_error(
                y_true,
                probability,
                n_bins=10,
            ),

        **calibration_fit,
    }