import copy

import numpy as np
import torch

from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


def get_pseudo_labels(
    model,
    X,
    device,
    positive_threshold=0.90,
    negative_threshold=0.10,
):
    """
    Generate high-confidence pseudo-labels.

    IMPORTANT:
    No ground-truth target labels are used.
    """

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

    positive_mask = (
        probabilities >= positive_threshold
    )

    negative_mask = (
        probabilities <= negative_threshold
    )

    selected_mask = (
        positive_mask | negative_mask
    )

    pseudo_labels = np.zeros(
        len(probabilities),
        dtype=np.float32,
    )

    pseudo_labels[
        positive_mask
    ] = 1.0

    confidence = np.maximum(
        probabilities,
        1.0 - probabilities,
    )

    return {
        "probabilities": probabilities,
        "selected_mask": selected_mask,
        "pseudo_labels": pseudo_labels,
        "confidence": confidence,
    }


def adapt_with_pseudo_labels(
    source_model,
    X_target_unlabeled,
    device,
    positive_threshold=0.90,
    negative_threshold=0.10,
    learning_rate=1e-4,
    weight_decay=1e-4,
    epochs=20,
    batch_size=16,
    min_selected=10,
):
    """
    Simple source-free pseudo-label adaptation.

    This is intentionally a baseline.

    No target ground-truth labels are accepted
    by this function.
    """

    model = copy.deepcopy(
        source_model
    ).to(device)

    pseudo_result = get_pseudo_labels(
        model=model,
        X=X_target_unlabeled,
        device=device,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
    )

    selected_mask = (
        pseudo_result["selected_mask"]
    )

    pseudo_labels = (
        pseudo_result["pseudo_labels"]
    )

    confidence = (
        pseudo_result["confidence"]
    )

    selected_n = int(
        selected_mask.sum()
    )

    coverage = (
        selected_n
        / len(X_target_unlabeled)
        if len(X_target_unlabeled) > 0
        else 0.0
    )

    if selected_n > 0:

        selected_labels = (
            pseudo_labels[
                selected_mask
            ]
        )

        pseudo_positive_n = int(
            selected_labels.sum()
        )

        pseudo_negative_n = int(
            selected_n
            - pseudo_positive_n
        )

        mean_confidence = float(
            confidence[
                selected_mask
            ].mean()
        )

    else:

        pseudo_positive_n = 0
        pseudo_negative_n = 0
        mean_confidence = np.nan


    stats = {
        "selected_n": selected_n,
        "coverage": coverage,
        "pseudo_positive_n":
            pseudo_positive_n,
        "pseudo_negative_n":
            pseudo_negative_n,
        "mean_selected_confidence":
            mean_confidence,
        "adaptation_skipped":
            False,
    }


    # --------------------------------------------------
    # Minimum viability condition
    # --------------------------------------------------

    if selected_n < min_selected:

        stats[
            "adaptation_skipped"
        ] = True

        return model, stats


    # --------------------------------------------------
    # Build pseudo-labeled dataset
    # --------------------------------------------------

    X_selected = (
        X_target_unlabeled[
            selected_mask
        ]
    )

    y_selected = (
        pseudo_labels[
            selected_mask
        ]
    )


    X_tensor = torch.tensor(
        X_selected,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y_selected,
        dtype=torch.float32,
    )


    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )


    loader = DataLoader(
        dataset,
        batch_size=min(
            batch_size,
            len(dataset),
        ),
        shuffle=True,
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


    criterion = nn.BCEWithLogitsLoss()


    # --------------------------------------------------
    # Fine-tune on pseudo labels
    # --------------------------------------------------

    for _ in range(epochs):

        model.train()

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )


            optimizer.zero_grad()


            logits = model(
                X_batch
            )


            loss = criterion(
                logits,
                y_batch,
            )


            loss.backward()

            optimizer.step()


    return model, stats