import copy

import numpy as np
import torch
import torch.nn.functional as F

from heart_sfda.adaptation.reliability_gate import (
    binary_entropy_from_logits,
    freeze_classifier,
    predict_probabilities,
    prediction_statistics,
    select_class_balanced_pseudo_labels,
)


def compute_reliability(
    mean_selected_confidence,
    class_support_score,
    missingness_factor,
    mode="full",
):
    """
    Compute label-free adaptation reliability.

    Modes
    -----
    full:
        confidence * class_support * missingness

    unit:
        reliability = 1

    no_missingness:
        confidence * class_support

    no_class_support:
        confidence * missingness

    no_confidence:
        class_support * missingness
    """

    if mode == "full":

        reliability = (
            mean_selected_confidence
            * class_support_score
            * missingness_factor
        )

    elif mode == "unit":

        reliability = 1.0

    elif mode == "no_missingness":

        reliability = (
            mean_selected_confidence
            * class_support_score
        )

    elif mode == "no_class_support":

        reliability = (
            mean_selected_confidence
            * missingness_factor
        )

    elif mode == "no_confidence":

        reliability = (
            class_support_score
            * missingness_factor
        )

    else:

        raise ValueError(
            f"Unknown reliability mode: {mode}"
        )

    return float(
        np.clip(
            reliability,
            0.0,
            1.0,
        )
    )


def adapt_conservative(
    source_model,
    X_target_unlabeled,
    device,
    missingness_rate=0.0,
    reliability_mode="full",
    selection_fraction=0.40,
    minimum_per_class=8,
    learning_rate=5e-5,
    weight_decay=1e-4,
    epochs=30,
    lambda_entropy=0.05,
    lambda_anchor=1.0,
    lambda_prior=0.50,
):
    """
    Pseudo-label-free conservative SFDA.

    Uses only:
        - predictive entropy
        - frozen source-teacher anchoring
        - predicted-prior preservation
        - label-free reliability scaling

    No target ground-truth labels are used.
    """

    # ======================================================
    # Frozen source teacher
    # ======================================================

    teacher = copy.deepcopy(
        source_model
    ).to(device)

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False


    # ======================================================
    # Student
    # ======================================================

    student = copy.deepcopy(
        source_model
    ).to(device)

    freeze_classifier(
        student
    )


    # ======================================================
    # Source-teacher target predictions
    # ======================================================

    teacher_probabilities = (
        predict_probabilities(
            teacher,
            X_target_unlabeled,
            device,
        )
    )

    before_stats = (
        prediction_statistics(
            teacher_probabilities
        )
    )


    # ======================================================
    # Balanced confidence subset
    #
    # IMPORTANT:
    # Labels returned here are NOT used for training.
    # Selection is used only to estimate reliability.
    # ======================================================

    selection = (
        select_class_balanced_pseudo_labels(
            teacher_probabilities,
            selection_fraction=
                selection_fraction,
        )
    )

    selected_indices = (
        selection["indices"]
    )

    selected_per_class = (
        selection["selected_per_class"]
    )


    # ======================================================
    # Reliability factors
    # ======================================================

    teacher_positive_rate = (
        before_stats[
            "predicted_positive_rate"
        ]
    )

    class_support_score = (
        2.0
        * min(
            teacher_positive_rate,
            1.0
            - teacher_positive_rate,
        )
    )


    if len(selected_indices) > 0:

        confidence = np.maximum(
            teacher_probabilities[
                selected_indices
            ],
            1.0
            - teacher_probabilities[
                selected_indices
            ],
        )

        mean_selected_confidence = float(
            confidence.mean()
        )

    else:

        mean_selected_confidence = 0.0


    missingness_factor = float(
        max(
            0.0,
            1.0
            - missingness_rate,
        )
    )


    reliability_score = (
        compute_reliability(
            mean_selected_confidence=
                mean_selected_confidence,

            class_support_score=
                class_support_score,

            missingness_factor=
                missingness_factor,

            mode=
                reliability_mode,
        )
    )


    stats = {
        "reliability_mode":
            reliability_mode,

        "selected_per_class":
            selected_per_class,

        "selected_n":
            int(
                len(
                    selected_indices
                )
            ),

        "teacher_positive_rate":
            teacher_positive_rate,

        "mean_selected_confidence":
            mean_selected_confidence,

        "class_support_score":
            class_support_score,

        "missingness_rate":
            float(
                missingness_rate
            ),

        "missingness_factor":
            missingness_factor,

        "reliability_score":
            reliability_score,

        "adaptation_skipped":
            False,
    }


    # ======================================================
    # Safety condition
    # ======================================================

    if (
        selected_per_class
        < minimum_per_class
    ):

        stats[
            "adaptation_skipped"
        ] = True

        stats.update(
            {
                "entropy_before":
                    before_stats[
                        "mean_entropy"
                    ],

                "entropy_after":
                    before_stats[
                        "mean_entropy"
                    ],

                "positive_rate_before":
                    before_stats[
                        "predicted_positive_rate"
                    ],

                "positive_rate_after":
                    before_stats[
                        "predicted_positive_rate"
                    ],

                "teacher_drift":
                    0.0,
            }
        )

        return student, stats


    # ======================================================
    # Tensors
    # ======================================================

    X_tensor = torch.tensor(
        X_target_unlabeled,
        dtype=torch.float32,
        device=device,
    )

    teacher_probability_tensor = (
        torch.tensor(
            teacher_probabilities,
            dtype=torch.float32,
            device=device,
        )
    )


    # ======================================================
    # Optimizer
    # ======================================================

    trainable_parameters = [
        parameter
        for parameter in student.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


    # ======================================================
    # Adaptation
    # ======================================================

    for _ in range(epochs):

        student.train()

        optimizer.zero_grad()


        logits = student(
            X_tensor
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
        )


        # --------------------------------------------------
        # Entropy adaptation
        # --------------------------------------------------

        entropy_loss = (
            binary_entropy_from_logits(
                logits
            )
        )


        # --------------------------------------------------
        # Teacher prediction anchor
        # --------------------------------------------------

        anchor_loss = (
            F.binary_cross_entropy_with_logits(
                logits,
                teacher_probability_tensor,
            )
        )


        # --------------------------------------------------
        # Prediction-prior preservation
        # --------------------------------------------------

        prior_loss = (
            probabilities.mean()
            -
            teacher_probability_tensor.mean()
        ) ** 2


        total_loss = (

            reliability_score
            * lambda_entropy
            * entropy_loss

            +

            lambda_anchor
            * anchor_loss

            +

            lambda_prior
            * prior_loss
        )


        total_loss.backward()

        optimizer.step()


    # ======================================================
    # Diagnostics
    # ======================================================

    adapted_probabilities = (
        predict_probabilities(
            student,
            X_target_unlabeled,
            device,
        )
    )


    after_stats = (
        prediction_statistics(
            adapted_probabilities
        )
    )


    stats.update(
        {
            "entropy_before":
                before_stats[
                    "mean_entropy"
                ],

            "entropy_after":
                after_stats[
                    "mean_entropy"
                ],

            "positive_rate_before":
                before_stats[
                    "predicted_positive_rate"
                ],

            "positive_rate_after":
                after_stats[
                    "predicted_positive_rate"
                ],

            "teacher_drift":
                float(
                    np.mean(
                        np.abs(
                            adapted_probabilities
                            -
                            teacher_probabilities
                        )
                    )
                ),
        }
    )


    return student, stats