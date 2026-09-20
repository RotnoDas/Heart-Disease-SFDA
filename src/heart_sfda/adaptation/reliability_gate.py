import copy

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-7


def freeze_classifier(model):
    """
    Freeze the final classification layer.

    We adapt the representation while preserving
    the source classifier.
    """

    for parameter in model.network[-1].parameters():
        parameter.requires_grad = False


def predict_probabilities(
    model,
    X,
    device,
):
    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        logits = model(X_tensor)

        probabilities = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

    return probabilities


def binary_entropy_from_logits(
    logits,
):
    probabilities = torch.sigmoid(
        logits
    )

    probabilities = torch.clamp(
        probabilities,
        EPS,
        1.0 - EPS,
    )

    entropy = -(
        probabilities
        * torch.log(probabilities)
        +
        (1.0 - probabilities)
        * torch.log(
            1.0 - probabilities
        )
    )

    return entropy.mean()


def prediction_statistics(
    probabilities,
):

    probabilities = np.asarray(
        probabilities
    )

    clipped = np.clip(
        probabilities,
        EPS,
        1.0 - EPS,
    )

    entropy = -(
        clipped
        * np.log(clipped)
        +
        (1.0 - clipped)
        * np.log(
            1.0 - clipped
        )
    )

    return {
        "mean_probability":
            float(
                probabilities.mean()
            ),

        "predicted_positive_rate":
            float(
                (
                    probabilities
                    >= 0.5
                ).mean()
            ),

        "mean_confidence":
            float(
                np.maximum(
                    probabilities,
                    1.0 - probabilities,
                ).mean()
            ),

        "mean_entropy":
            float(
                entropy.mean()
            ),
    }


def select_class_balanced_pseudo_labels(
    probabilities,
    selection_fraction=0.40,
):
    """
    Select an equal number of source-model predicted
    positive and negative target samples.

    Selection uses confidence ranking ONLY.
    Target ground-truth labels are never used.
    """

    probabilities = np.asarray(
        probabilities
    )

    predicted_labels = (
        probabilities >= 0.5
    ).astype(int)

    confidence = np.maximum(
        probabilities,
        1.0 - probabilities,
    )

    positive_indices = np.where(
        predicted_labels == 1
    )[0]

    negative_indices = np.where(
        predicted_labels == 0
    )[0]

    desired_per_class = max(
        1,
        int(
            round(
                len(probabilities)
                * selection_fraction
                / 2.0
            )
        ),
    )

    selected_per_class = min(
        desired_per_class,
        len(positive_indices),
        len(negative_indices),
    )

    if selected_per_class == 0:

        return {
            "indices":
                np.array(
                    [],
                    dtype=int,
                ),

            "labels":
                np.array(
                    [],
                    dtype=np.float32,
                ),

            "weights":
                np.array(
                    [],
                    dtype=np.float32,
                ),

            "selected_per_class":
                0,

            "available_positive":
                len(
                    positive_indices
                ),

            "available_negative":
                len(
                    negative_indices
                ),
        }

    positive_ranked = (
        positive_indices[
            np.argsort(
                confidence[
                    positive_indices
                ]
            )[::-1]
        ]
    )

    negative_ranked = (
        negative_indices[
            np.argsort(
                confidence[
                    negative_indices
                ]
            )[::-1]
        ]
    )

    selected_positive = (
        positive_ranked[
            :selected_per_class
        ]
    )

    selected_negative = (
        negative_ranked[
            :selected_per_class
        ]
    )

    selected_indices = np.concatenate(
        [
            selected_positive,
            selected_negative,
        ]
    )

    selected_labels = (
        predicted_labels[
            selected_indices
        ]
        .astype(
            np.float32
        )
    )

    selected_confidence = (
        confidence[
            selected_indices
        ]
    )

    # --------------------------------------------------
    # Convert confidence from:
    #
    # 0.50 -> 0
    # 1.00 -> 1
    #
    # Then prevent exactly-zero weights.
    # --------------------------------------------------

    weights = (
        2.0
        * (
            selected_confidence
            - 0.5
        )
    )

    weights = np.clip(
        weights,
        0.05,
        1.0,
    ).astype(
        np.float32
    )

    return {
        "indices":
            selected_indices,

        "labels":
            selected_labels,

        "weights":
            weights,

        "selected_per_class":
            int(
                selected_per_class
            ),

        "available_positive":
            int(
                len(
                    positive_indices
                )
            ),

        "available_negative":
            int(
                len(
                    negative_indices
                )
            ),
    }


def adapt_reliability_gated(
    source_model,
    X_target_unlabeled,
    device,
    missingness_rate=0.0,
    selection_fraction=0.40,
    minimum_per_class=8,
    learning_rate=5e-5,
    weight_decay=1e-4,
    epochs=30,
    lambda_pseudo=1.0,
    lambda_entropy=0.05,
    lambda_anchor=1.0,
    lambda_prior=0.50,
):
    """
    Reliability-gated source-free adaptation.

    Uses:
    - class-balanced pseudo-label selection
    - confidence weighting
    - entropy minimization
    - teacher/source prediction anchoring
    - predicted-prior preservation
    - target missingness penalty

    No target labels are used.
    """

    # ==================================================
    # Teacher
    # ==================================================

    teacher = copy.deepcopy(
        source_model
    ).to(device)

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False


    # ==================================================
    # Student
    # ==================================================

    student = copy.deepcopy(
        source_model
    ).to(device)

    freeze_classifier(
        student
    )


    # ==================================================
    # Source-model predictions on unlabeled target
    # ==================================================

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


    # ==================================================
    # Class-balanced pseudo labels
    # ==================================================

    selection = (
        select_class_balanced_pseudo_labels(
            teacher_probabilities,
            selection_fraction=
                selection_fraction,
        )
    )


    selected_indices = (
        selection[
            "indices"
        ]
    )


    selected_n = len(
        selected_indices
    )


    selected_per_class = (
        selection[
            "selected_per_class"
        ]
    )


    # ==================================================
    # Reliability calculation
    # ==================================================

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


    if selected_n > 0:

        selected_confidence = (
            np.maximum(
                teacher_probabilities[
                    selected_indices
                ],
                1.0
                - teacher_probabilities[
                    selected_indices
                ],
            )
        )

        mean_selected_confidence = (
            float(
                selected_confidence.mean()
            )
        )

    else:

        mean_selected_confidence = 0.0


    missingness_factor = (
        max(
            0.0,
            1.0
            - float(
                missingness_rate
            ),
        )
    )


    reliability_score = (
        mean_selected_confidence
        * class_support_score
        * missingness_factor
    )


    reliability_score = float(
        np.clip(
            reliability_score,
            0.0,
            1.0,
        )
    )


    coverage = (
        selected_n
        / len(
            X_target_unlabeled
        )
        if len(
            X_target_unlabeled
        ) > 0
        else 0.0
    )


    stats = {
        "selected_n":
            selected_n,

        "selected_per_class":
            selected_per_class,

        "coverage":
            coverage,

        "available_positive":
            selection[
                "available_positive"
            ],

        "available_negative":
            selection[
                "available_negative"
            ],

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

        "reliability_score":
            reliability_score,

        "adaptation_skipped":
            False,
    }


    # ==================================================
    # Safety gate
    # ==================================================

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

                "mean_probability_before":
                    before_stats[
                        "mean_probability"
                    ],

                "mean_probability_after":
                    before_stats[
                        "mean_probability"
                    ],

                "mean_abs_teacher_drift":
                    0.0,
            }
        )

        return student, stats


    # ==================================================
    # Tensors
    # ==================================================

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


    selected_index_tensor = (
        torch.tensor(
            selected_indices,
            dtype=torch.long,
            device=device,
        )
    )


    pseudo_label_tensor = (
        torch.tensor(
            selection[
                "labels"
            ],
            dtype=torch.float32,
            device=device,
        )
    )


    pseudo_weight_tensor = (
        torch.tensor(
            selection[
                "weights"
            ],
            dtype=torch.float32,
            device=device,
        )
    )


    # ==================================================
    # Optimizer
    # ==================================================

    trainable_parameters = [
        parameter
        for parameter
        in student.parameters()
        if parameter.requires_grad
    ]


    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


    # ==================================================
    # Adaptation
    # ==================================================

    for _ in range(epochs):

        student.train()

        optimizer.zero_grad()


        logits = student(
            X_tensor
        )


        probabilities = torch.sigmoid(
            logits
        )


        # ----------------------------------------------
        # 1. Balanced pseudo-label loss
        # ----------------------------------------------

        selected_logits = (
            logits[
                selected_index_tensor
            ]
        )


        pseudo_losses = (
            F.binary_cross_entropy_with_logits(
                selected_logits,
                pseudo_label_tensor,
                reduction="none",
            )
        )


        pseudo_loss = (
            pseudo_losses
            * pseudo_weight_tensor
        ).mean()


        # ----------------------------------------------
        # 2. Entropy loss
        # ----------------------------------------------

        entropy_loss = (
            binary_entropy_from_logits(
                logits
            )
        )


        # ----------------------------------------------
        # 3. Source-teacher anchor
        # ----------------------------------------------

        anchor_loss = (
            F.binary_cross_entropy_with_logits(
                logits,
                teacher_probability_tensor,
            )
        )


        # ----------------------------------------------
        # 4. Predicted-prior preservation
        # ----------------------------------------------

        prior_loss = (
            probabilities.mean()
            -
            teacher_probability_tensor.mean()
        ) ** 2


        # ----------------------------------------------
        # Reliability-gated target adaptation
        # ----------------------------------------------

        adaptation_loss = (
            lambda_pseudo
            * pseudo_loss
            +
            lambda_entropy
            * entropy_loss
        )


        preservation_loss = (
            lambda_anchor
            * anchor_loss
            +
            lambda_prior
            * prior_loss
        )


        total_loss = (
            reliability_score
            * adaptation_loss
            +
            preservation_loss
        )


        total_loss.backward()

        optimizer.step()


    # ==================================================
    # Post-adaptation diagnostics
    # ==================================================

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

            "mean_probability_before":
                before_stats[
                    "mean_probability"
                ],

            "mean_probability_after":
                after_stats[
                    "mean_probability"
                ],

            "mean_abs_teacher_drift":
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