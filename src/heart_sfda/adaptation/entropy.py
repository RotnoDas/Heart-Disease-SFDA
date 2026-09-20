import copy

import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset


EPS = 1e-7


def binary_entropy_from_logits(logits):
    """
    Mean binary predictive entropy.

    H(p) = -p log(p) - (1-p) log(1-p)
    """

    probabilities = torch.sigmoid(logits)

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


def predict_statistics(
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

    clipped = np.clip(
        probabilities,
        EPS,
        1.0 - EPS,
    )

    entropy = -(
        clipped * np.log(clipped)
        +
        (1.0 - clipped)
        * np.log(1.0 - clipped)
    )

    return {
        "mean_entropy":
            float(entropy.mean()),

        "mean_probability":
            float(probabilities.mean()),

        "predicted_positive_rate":
            float(
                (probabilities >= 0.5)
                .mean()
            ),

        "mean_confidence":
            float(
                np.maximum(
                    probabilities,
                    1.0 - probabilities,
                ).mean()
            ),
    }


def freeze_classifier(model):
    """
    Freeze final classification layer.

    HeartMLP.network[-1] is the output Linear layer.
    """

    for parameter in (
        model.network[-1].parameters()
    ):
        parameter.requires_grad = False


def adapt_with_entropy(
    source_model,
    X_target_unlabeled,
    device,
    learning_rate=1e-4,
    weight_decay=1e-4,
    epochs=20,
    batch_size=32,
):
    """
    Source-free entropy-minimization baseline.

    Target ground-truth labels are never used.

    Final classifier is frozen.
    Earlier representation layers are updated.
    """

    model = copy.deepcopy(
        source_model
    ).to(device)

    freeze_classifier(model)

    before_stats = predict_statistics(
        model,
        X_target_unlabeled,
        device,
    )

    X_tensor = torch.tensor(
        X_target_unlabeled,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=min(
            batch_size,
            len(dataset),
        ),
        shuffle=True,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    for _ in range(epochs):

        model.train()

        for (X_batch,) in loader:

            X_batch = X_batch.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = (
                binary_entropy_from_logits(
                    logits
                )
            )

            loss.backward()

            optimizer.step()

    after_stats = predict_statistics(
        model,
        X_target_unlabeled,
        device,
    )

    stats = {

        "entropy_before":
            before_stats[
                "mean_entropy"
            ],

        "entropy_after":
            after_stats[
                "mean_entropy"
            ],

        "mean_probability_before":
            before_stats[
                "mean_probability"
            ],

        "mean_probability_after":
            after_stats[
                "mean_probability"
            ],

        "positive_rate_before":
            before_stats[
                "predicted_positive_rate"
            ],

        "positive_rate_after":
            after_stats[
                "predicted_positive_rate"
            ],

        "confidence_before":
            before_stats[
                "mean_confidence"
            ],

        "confidence_after":
            after_stats[
                "mean_confidence"
            ],
    }

    return model, stats