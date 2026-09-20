import copy

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-9


def _as_tensor(
    X,
    device,
):
    return torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )


def _entropy(
    probability,
):
    probability = probability.clamp_min(
        EPS
    )

    return -torch.sum(
        probability
        * torch.log(
            probability
        ),
        dim=1,
    )


def _dual_forward(
    model,
    X_tensor,
):
    (
        features,
        logits_1,
        logits_2,
    ) = model(
        X_tensor
    )

    probability_1 = F.softmax(
        logits_1,
        dim=1,
    )

    probability_2 = F.softmax(
        logits_2,
        dim=1,
    )

    mean_probability = (
        probability_1
        + probability_2
    ) / 2.0

    return (
        features,
        logits_1,
        logits_2,
        probability_1,
        probability_2,
        mean_probability,
    )


def predict_dlar_lcl_probability(
    model,
    X,
    device,
):
    """
    Return probability of class 1 using the mean
    of the two classifier softmax outputs.
    """

    model.eval()

    X_tensor = _as_tensor(
        X,
        device,
    )

    with torch.no_grad():

        (
            _,
            _,
            _,
            probability_1,
            probability_2,
            mean_probability,
        ) = _dual_forward(
            model,
            X_tensor,
        )

    return (
        mean_probability[:, 1]
        .detach()
        .cpu()
        .numpy()
    )


def compute_target_statistics(
    model,
    X_target,
    device,
):
    """
    Compute the target-domain quantities used before DLAR:

        - soft class centroids
        - mean classifier discrepancy

    This follows the algorithmic definition from the paper:
    each feature contributes to class k's centroid weighted
    by the averaged softmax probability for class k.
    """

    model.eval()

    X_tensor = _as_tensor(
        X_target,
        device,
    )

    with torch.no_grad():

        (
            features,
            _,
            _,
            probability_1,
            probability_2,
            mean_probability,
        ) = _dual_forward(
            model,
            X_tensor,
        )

        # ----------------------------------------------------
        # Soft target centroids
        # ----------------------------------------------------

        class_masses = (
            mean_probability.sum(
                dim=0
            )
            .clamp_min(EPS)
        )

        centers = []

        for class_index in range(2):

            weights = (
                mean_probability[
                    :,
                    class_index
                ]
                .unsqueeze(1)
            )

            center = (
                (
                    features
                    * weights
                )
                .sum(
                    dim=0
                )
                /
                class_masses[
                    class_index
                ]
            )

            centers.append(
                center
            )

        centers = torch.stack(
            centers,
            dim=0,
        )


        # ----------------------------------------------------
        # Mean Euclidean classifier discrepancy
        # ----------------------------------------------------

        classifier_discrepancy = (
            torch.norm(
                probability_1
                - probability_2,
                p=2,
                dim=1,
            )
        )

        mean_classifier_discrepancy = (
            classifier_discrepancy.mean()
        )


        source_positive_rate = (
            mean_probability[
                :,
                1
            ]
            .mean()
        )


        source_predicted_positive_rate = (
            (
                mean_probability[
                    :,
                    1
                ]
                >= 0.5
            )
            .float()
            .mean()
        )


        mean_entropy = (
            _entropy(
                mean_probability
            )
            .mean()
        )


    return {
        "centers":
            centers.detach(),

        "mean_classifier_discrepancy":
            float(
                mean_classifier_discrepancy.item()
            ),

        "mean_probability_positive":
            float(
                source_positive_rate.item()
            ),

        "predicted_positive_rate":
            float(
                source_predicted_positive_rate.item()
            ),

        "mean_entropy":
            float(
                mean_entropy.item()
            ),

        "centroid_distance":
            float(
                torch.norm(
                    centers[0]
                    - centers[1]
                ).item()
            ),
    }


def _nearest_centroid_labels(
    features,
    centers,
):
    distances = torch.cdist(
        features,
        centers,
        p=2,
    )

    return torch.argmin(
        distances,
        dim=1,
    )


def _lcl_loss(
    features,
    probabilities,
    k=5,
):
    """
    Localized Consistency Learning.

    Reliable neighbors are those appearing in BOTH:
        - kNN in feature space
        - kNN in prediction-probability space

    Pair consistency:
        -log(p_i dot p_j)

    Returns:
        loss
        mean common-neighbor count
        number of contributing pairs
    """

    n_samples = (
        features.shape[0]
    )

    if n_samples <= 1:

        zero = (
            features.sum()
            * 0.0
        )

        return (
            zero,
            0.0,
            0,
        )


    k_effective = min(
        k,
        n_samples - 1,
    )


    # --------------------------------------------------------
    # Feature-space neighbors
    # --------------------------------------------------------

    feature_distances = torch.cdist(
        features,
        features,
        p=2,
    )

    feature_distances.fill_diagonal_(
        float("inf")
    )

    feature_neighbors = torch.topk(
        feature_distances,
        k=k_effective,
        dim=1,
        largest=False,
    ).indices


    # --------------------------------------------------------
    # Probability-space neighbors
    # --------------------------------------------------------

    probability_distances = torch.cdist(
        probabilities,
        probabilities,
        p=2,
    )

    probability_distances.fill_diagonal_(
        float("inf")
    )

    probability_neighbors = torch.topk(
        probability_distances,
        k=k_effective,
        dim=1,
        largest=False,
    ).indices


    pair_losses = []

    common_neighbor_counts = []


    for i in range(
        n_samples
    ):

        feature_set = set(
            feature_neighbors[
                i
            ]
            .detach()
            .cpu()
            .tolist()
        )

        probability_set = set(
            probability_neighbors[
                i
            ]
            .detach()
            .cpu()
            .tolist()
        )

        common = sorted(
            feature_set.intersection(
                probability_set
            )
        )


        common_neighbor_counts.append(
            len(common)
        )


        for j in common:

            similarity = torch.sum(
                probabilities[i]
                * probabilities[j]
            )

            similarity = (
                similarity
                .clamp_min(
                    EPS
                )
            )

            pair_losses.append(
                -torch.log(
                    similarity
                )
            )


    if len(pair_losses) == 0:

        zero = (
            features.sum()
            * 0.0
        )

        return (
            zero,
            float(
                np.mean(
                    common_neighbor_counts
                )
            ),
            0,
        )


    # Original implementation accumulates pair loss
    # and normalizes by number of samples rather than
    # number of pairs. We preserve that convention.
    loss = (
        torch.stack(
            pair_losses
        )
        .sum()
        /
        float(
            n_samples
        )
    )


    return (
        loss,
        float(
            np.mean(
                common_neighbor_counts
            )
        ),
        len(
            pair_losses
        ),
    )


def adapt_dlar_lcl(
    source_model,
    X_target_unlabeled,
    device,
    learning_rate=1e-4,
    weight_decay=5e-4,
    dlar_epochs=5,
    lcl_epochs=10,
    beta=1.0,
    gamma=1.0,
    k_neighbors=5,
):
    """
    Tabular port of the DLAR + LCL target adaptation pipeline.

    IMPORTANT
    ---------
    Target labels are never accepted by this function.

    Source data are also not required.

    Inputs:
        pretrained source model
        unlabeled target feature matrix

    Paper/repository defaults:
        target LR = 1e-4
        weight decay = 5e-4
        DLAR epochs = 5
        LCL epochs = 10
        beta = 1
        gamma = 1
        k = 5
    """

    student = copy.deepcopy(
        source_model
    ).to(device)


    # --------------------------------------------------------
    # Original repository keeps the loaded source network
    # in eval mode during adaptation.
    #
    # Gradients still work in eval mode; this merely disables
    # dropout / running-stat updates.
    # --------------------------------------------------------

    student.eval()


    X_tensor = _as_tensor(
        X_target_unlabeled,
        device,
    )


    # ========================================================
    # Pre-adaptation target computation stage
    # ========================================================

    target_statistics = (
        compute_target_statistics(
            student,
            X_target_unlabeled,
            device,
        )
    )


    centers = (
        target_statistics[
            "centers"
        ]
        .to(device)
    )


    mean_discrepancy = float(
        target_statistics[
            "mean_classifier_discrepancy"
        ]
    )


    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


    dlar_confident_fractions = []

    dlar_pseudo_positive_fractions = []

    dlar_losses = []


    # ========================================================
    # Stage 1 — DLAR
    # ========================================================

    for _ in range(
        dlar_epochs
    ):

        optimizer.zero_grad()


        (
            features,
            _,
            _,
            probability_1,
            probability_2,
            mean_probability,
        ) = _dual_forward(
            student,
            X_tensor,
        )


        # ----------------------------------------------------
        # Classifier discrepancy
        # ----------------------------------------------------

        discrepancy_per_sample = (
            torch.norm(
                probability_1
                - probability_2,
                p=2,
                dim=1,
            )
        )


        # ----------------------------------------------------
        # Entropy-based confidence
        # ----------------------------------------------------

        entropy_per_sample = (
            _entropy(
                mean_probability
            )
        )


        entropy_threshold = (
            entropy_per_sample.mean()
        )


        confident_mask = (
            (
                entropy_per_sample
                < entropy_threshold
            )
            &
            (
                discrepancy_per_sample
                < mean_discrepancy
            )
        )


        confident_fraction = float(
            confident_mask
            .float()
            .mean()
            .item()
        )


        # ----------------------------------------------------
        # Classifier-agreement loss on confident samples
        #
        # The released implementation can generate NaN if
        # there are no confident samples. We explicitly return
        # zero in that edge case.
        # ----------------------------------------------------

        if (
            confident_mask.any()
        ):

            classifier_loss = (
                torch.norm(
                    probability_1[
                        confident_mask
                    ]
                    -
                    probability_2[
                        confident_mask
                    ],
                    p=2,
                    dim=1,
                )
                .mean()
            )

        else:

            classifier_loss = (
                mean_probability.sum()
                * 0.0
            )


        # ----------------------------------------------------
        # Nearest-soft-centroid pseudo-labels
        # ----------------------------------------------------

        pseudo_labels = (
            _nearest_centroid_labels(
                features,
                centers,
            )
        )


        pseudo_positive_fraction = float(
            (
                pseudo_labels
                == 1
            )
            .float()
            .mean()
            .item()
        )


        # ----------------------------------------------------
        # Paper equation uses cross-entropy between
        # pseudo-labels and predicted probabilities.
        #
        # Using NLL(log p) is the numerically correct
        # implementation of that equation.
        # ----------------------------------------------------

        pseudo_label_loss = (
            F.nll_loss(
                torch.log(
                    mean_probability
                    .clamp_min(
                        EPS
                    )
                ),
                pseudo_labels,
            )
        )


        total_loss = (
            gamma
            * pseudo_label_loss
            +
            beta
            * classifier_loss
        )


        if not torch.isfinite(
            total_loss
        ):

            raise RuntimeError(
                "Non-finite DLAR loss."
            )


        total_loss.backward()


        torch.nn.utils.clip_grad_norm_(
            student.feature_extractor.parameters(),
            max_norm=1.0,
        )


        optimizer.step()


        dlar_confident_fractions.append(
            confident_fraction
        )

        dlar_pseudo_positive_fractions.append(
            pseudo_positive_fraction
        )

        dlar_losses.append(
            float(
                total_loss.item()
            )
        )


    # ========================================================
    # Stage 2 — LCL
    # ========================================================

    lcl_losses = []

    lcl_common_neighbors = []

    lcl_pair_counts = []


    for _ in range(
        lcl_epochs
    ):

        optimizer.zero_grad()


        (
            features,
            _,
            _,
            _,
            _,
            mean_probability,
        ) = _dual_forward(
            student,
            X_tensor,
        )


        (
            lcl_loss,
            mean_common_neighbors,
            pair_count,
        ) = _lcl_loss(
            features=
                features,

            probabilities=
                mean_probability,

            k=
                k_neighbors,
        )


        if (
            pair_count > 0
        ):

            if not torch.isfinite(
                lcl_loss
            ):

                raise RuntimeError(
                    "Non-finite LCL loss."
                )


            lcl_loss.backward()


            torch.nn.utils.clip_grad_norm_(
                student.feature_extractor.parameters(),
                max_norm=1.0,
            )


            optimizer.step()


        lcl_losses.append(
            float(
                lcl_loss.item()
            )
        )

        lcl_common_neighbors.append(
            mean_common_neighbors
        )

        lcl_pair_counts.append(
            pair_count
        )


    # ========================================================
    # Post-adaptation diagnostics
    # ========================================================

    student.eval()


    with torch.no_grad():

        (
            _,
            _,
            _,
            _,
            _,
            final_probability,
        ) = _dual_forward(
            student,
            X_tensor,
        )


        final_positive_probability = (
            final_probability[
                :,
                1
            ]
        )


        final_positive_rate = float(
            (
                final_positive_probability
                >= 0.5
            )
            .float()
            .mean()
            .item()
        )


        final_mean_entropy = float(
            _entropy(
                final_probability
            )
            .mean()
            .item()
        )


        # Original source model probability on same
        # adaptation samples for drift calculation.
        source_model.eval()

        (
            _,
            _,
            _,
            _,
            _,
            original_probability,
        ) = _dual_forward(
            source_model,
            X_tensor,
        )


        probability_drift = float(
            torch.mean(
                torch.abs(
                    final_probability[
                        :,
                        1
                    ]
                    -
                    original_probability[
                        :,
                        1
                    ]
                )
            ).item()
        )


    diagnostics = {

        "target_n":
            int(
                len(
                    X_target_unlabeled
                )
            ),

        "mean_classifier_discrepancy":
            mean_discrepancy,

        "centroid_distance":
            target_statistics[
                "centroid_distance"
            ],

        "positive_rate_before":
            target_statistics[
                "predicted_positive_rate"
            ],

        "positive_rate_after":
            final_positive_rate,

        "entropy_before":
            target_statistics[
                "mean_entropy"
            ],

        "entropy_after":
            final_mean_entropy,

        "probability_drift":
            probability_drift,

        "mean_dlar_confident_fraction":
            float(
                np.mean(
                    dlar_confident_fractions
                )
            ),

        "mean_pseudo_positive_fraction":
            float(
                np.mean(
                    dlar_pseudo_positive_fractions
                )
            ),

        "mean_dlar_loss":
            float(
                np.mean(
                    dlar_losses
                )
            ),

        "mean_lcl_loss":
            float(
                np.mean(
                    lcl_losses
                )
            ),

        "mean_common_neighbors":
            float(
                np.mean(
                    lcl_common_neighbors
                )
            ),

        "mean_lcl_pair_count":
            float(
                np.mean(
                    lcl_pair_counts
                )
            ),
    }


    return (
        student,
        diagnostics,
    )