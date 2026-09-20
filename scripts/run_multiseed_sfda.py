from pathlib import Path
import copy
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from heart_sfda.data.preprocessing import CORE8
from heart_sfda.models.mlp import HeartMLP

from heart_sfda.adaptation.pseudo_label import (
    adapt_with_pseudo_labels,
)

from heart_sfda.adaptation.entropy import (
    adapt_with_entropy,
)

from heart_sfda.adaptation.reliability_gate import (
    adapt_reliability_gated,
)

from heart_sfda.evaluation.metrics import (
    compute_binary_metrics,
)

from heart_sfda.utils.device import get_device
from heart_sfda.utils.seed import set_seed


# ============================================================
# Fixed experimental configuration
# ============================================================

SEEDS = list(range(10))

SOURCE_EPOCHS = 22

SOURCE_LR = 1e-3
SOURCE_WEIGHT_DECAY = 1e-4
SOURCE_BATCH_SIZE = 32

N_FOLDS = 5
FOLD_RANDOM_STATE = 42


# Pseudo-label baseline
PSEUDO_POSITIVE_THRESHOLD = 0.90
PSEUDO_NEGATIVE_THRESHOLD = 0.10
PSEUDO_LR = 1e-4
PSEUDO_WEIGHT_DECAY = 1e-4
PSEUDO_EPOCHS = 20
PSEUDO_BATCH_SIZE = 16
PSEUDO_MIN_SELECTED = 10


# Entropy baseline
ENTROPY_LR = 1e-4
ENTROPY_WEIGHT_DECAY = 1e-4
ENTROPY_EPOCHS = 20
ENTROPY_BATCH_SIZE = 32


# Reliability-gated v1
RG_SELECTION_FRACTION = 0.40
RG_MINIMUM_PER_CLASS = 8
RG_LR = 5e-5
RG_WEIGHT_DECAY = 1e-4
RG_EPOCHS = 30

RG_LAMBDA_PSEUDO = 1.0
RG_LAMBDA_ENTROPY = 0.05
RG_LAMBDA_ANCHOR = 1.0
RG_LAMBDA_PRIOR = 0.50


DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)

SOURCE_MODEL_DIR = (
    ROOT
    / "models"
    / "source"
    / "mlp"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "statistics"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGETS = {
    "hungary": "hungary.csv",
    "switzerland": "switzerland.csv",
    "va_long_beach": "va_long_beach.csv",
}


# ============================================================
# Utilities
# ============================================================

def load_dataset(filename):
    return pd.read_csv(
        DATA_DIR / filename
    )


def make_loader(
    X,
    y,
    batch_size,
):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )


def train_source_model(
    X,
    y,
    seed,
    device,
):

    set_seed(seed)

    model = HeartMLP(
        input_dim=X.shape[1],
        hidden_dims=(32, 16),
        dropout=0.20,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SOURCE_LR,
        weight_decay=SOURCE_WEIGHT_DECAY,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    loader = make_loader(
        X,
        y,
        SOURCE_BATCH_SIZE,
    )

    for _ in range(
        SOURCE_EPOCHS
    ):

        model.train()

        for (
            X_batch,
            y_batch,
        ) in loader:

            X_batch = (
                X_batch.to(device)
            )

            y_batch = (
                y_batch.to(device)
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

    model.eval()

    return model


def predict_probability(
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

        probability = (
            torch.sigmoid(
                model(X_tensor)
            )
            .cpu()
            .numpy()
        )

    return probability


# ============================================================
# Cross-fitted adaptation
# ============================================================

def run_target_method(
    method,
    source_model,
    X_target,
    X_target_raw,
    device,
    seed,
):

    kfold = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=FOLD_RANDOM_STATE,
    )

    probabilities = np.zeros(
        len(X_target),
        dtype=float,
    )

    for fold, (
        adapt_indices,
        test_indices,
    ) in enumerate(
        kfold.split(X_target),
        start=1,
    ):

        # Fresh copy of exactly the same source model.
        fold_source_model = copy.deepcopy(
            source_model
        ).to(device)

        X_adapt = (
            X_target[
                adapt_indices
            ]
        )

        X_test = (
            X_target[
                test_indices
            ]
        )

        # Deterministic but distinct seed
        # for each seed/method/fold run.
        method_offset = {
            "pseudo_label_sfda": 1000,
            "entropy_sfda": 2000,
            "reliability_gated_sfda": 3000,
        }[method]

        set_seed(
            seed * 10000
            + method_offset
            + fold
        )

        if (
            method
            == "pseudo_label_sfda"
        ):

            adapted_model, _ = (
                adapt_with_pseudo_labels(
                    source_model=
                        fold_source_model,

                    X_target_unlabeled=
                        X_adapt,

                    device=device,

                    positive_threshold=
                        PSEUDO_POSITIVE_THRESHOLD,

                    negative_threshold=
                        PSEUDO_NEGATIVE_THRESHOLD,

                    learning_rate=
                        PSEUDO_LR,

                    weight_decay=
                        PSEUDO_WEIGHT_DECAY,

                    epochs=
                        PSEUDO_EPOCHS,

                    batch_size=
                        PSEUDO_BATCH_SIZE,

                    min_selected=
                        PSEUDO_MIN_SELECTED,
                )
            )

        elif (
            method
            == "entropy_sfda"
        ):

            adapted_model, _ = (
                adapt_with_entropy(
                    source_model=
                        fold_source_model,

                    X_target_unlabeled=
                        X_adapt,

                    device=device,

                    learning_rate=
                        ENTROPY_LR,

                    weight_decay=
                        ENTROPY_WEIGHT_DECAY,

                    epochs=
                        ENTROPY_EPOCHS,

                    batch_size=
                        ENTROPY_BATCH_SIZE,
                )
            )

        elif (
            method
            == "reliability_gated_sfda"
        ):

            missingness_rate = float(
                X_target_raw
                .iloc[
                    adapt_indices
                ]
                .isna()
                .mean()
                .mean()
            )

            adapted_model, _ = (
                adapt_reliability_gated(
                    source_model=
                        fold_source_model,

                    X_target_unlabeled=
                        X_adapt,

                    device=device,

                    missingness_rate=
                        missingness_rate,

                    selection_fraction=
                        RG_SELECTION_FRACTION,

                    minimum_per_class=
                        RG_MINIMUM_PER_CLASS,

                    learning_rate=
                        RG_LR,

                    weight_decay=
                        RG_WEIGHT_DECAY,

                    epochs=
                        RG_EPOCHS,

                    lambda_pseudo=
                        RG_LAMBDA_PSEUDO,

                    lambda_entropy=
                        RG_LAMBDA_ENTROPY,

                    lambda_anchor=
                        RG_LAMBDA_ANCHOR,

                    lambda_prior=
                        RG_LAMBDA_PRIOR,
                )
            )

        else:
            raise ValueError(
                f"Unknown method: {method}"
            )

        probabilities[
            test_indices
        ] = predict_probability(
            adapted_model,
            X_test,
            device,
        )

    return probabilities


# ============================================================
# Main
# ============================================================

def main():

    device = get_device()

    preprocessor = joblib.load(
        SOURCE_MODEL_DIR
        / "cleveland_core8_preprocessor.joblib"
    )


    # --------------------------------------------------------
    # Source data
    # --------------------------------------------------------

    source_df = load_dataset(
        "cleveland.csv"
    )

    X_source = (
        preprocessor
        .transform(
            source_df[
                CORE8
            ]
        )
        .astype(
            np.float32
        )
    )

    y_source = (
        source_df[
            "target"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    rows = []


    print(
        "\n"
        + "=" * 110
    )

    print(
        "MULTI-SEED SFDA ROBUSTNESS"
    )

    print(
        "=" * 110
    )

    print(
        "\nSeeds:",
        SEEDS
    )

    print(
        "Source epochs:",
        SOURCE_EPOCHS
    )


    for seed in SEEDS:

        print(
            "\n"
            + "#" * 110
        )

        print(
            f"SEED {seed}"
        )

        print(
            "#" * 110
        )


        source_model = (
            train_source_model(
                X_source,
                y_source,
                seed,
                device,
            )
        )


        for (
            target,
            filename,
        ) in TARGETS.items():

            print(
                f"\nSeed {seed} | "
                f"Target {target}"
            )

            target_df = load_dataset(
                filename
            )

            X_target_raw = (
                target_df[
                    CORE8
                ]
            )

            X_target = (
                preprocessor
                .transform(
                    X_target_raw
                )
                .astype(
                    np.float32
                )
            )

            y_target = (
                target_df[
                    "target"
                ]
                .to_numpy()
            )


            # ------------------------------------------------
            # Source only
            # ------------------------------------------------

            source_probability = (
                predict_probability(
                    source_model,
                    X_target,
                    device,
                )
            )

            source_metrics = (
                compute_binary_metrics(
                    y_target,
                    source_probability,
                )
            )


            rows.append(
                {
                    "seed":
                        seed,

                    "target":
                        target,

                    "method":
                        "source_only",

                    **source_metrics,
                }
            )


            print(
                "  source_only"
                f" AUROC={source_metrics['roc_auc']:.4f}"
                f" Brier={source_metrics['brier']:.4f}"
            )


            # ------------------------------------------------
            # Adaptation methods
            # ------------------------------------------------

            for method in [
                "pseudo_label_sfda",
                "entropy_sfda",
                "reliability_gated_sfda",
            ]:

                probability = (
                    run_target_method(
                        method=
                            method,

                        source_model=
                            source_model,

                        X_target=
                            X_target,

                        X_target_raw=
                            X_target_raw,

                        device=
                            device,

                        seed=
                            seed,
                    )
                )

                metrics = (
                    compute_binary_metrics(
                        y_target,
                        probability,
                    )
                )


                rows.append(
                    {
                        "seed":
                            seed,

                        "target":
                            target,

                        "method":
                            method,

                        **metrics,
                    }
                )


                print(
                    f"  {method}"
                    f" AUROC={metrics['roc_auc']:.4f}"
                    f" Brier={metrics['brier']:.4f}"
                )


    # ========================================================
    # Save raw seed-level results
    # ========================================================

    result_df = pd.DataFrame(
        rows
    )

    raw_path = (
        RESULT_DIR
        / "multiseed_sfda_core8.csv"
    )

    result_df.to_csv(
        raw_path,
        index=False,
    )


    # ========================================================
    # Build paired deltas against source-only
    # ========================================================

    source_rows = (
        result_df[
            result_df["method"]
            == "source_only"
        ][
            [
                "seed",
                "target",
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "brier",
            ]
        ]
        .rename(
            columns={
                "roc_auc":
                    "source_roc_auc",

                "pr_auc":
                    "source_pr_auc",

                "balanced_accuracy":
                    "source_balanced_accuracy",

                "brier":
                    "source_brier",
            }
        )
    )


    adapted_rows = (
        result_df[
            result_df["method"]
            != "source_only"
        ]
        .copy()
    )


    comparison = (
        adapted_rows.merge(
            source_rows,
            on=[
                "seed",
                "target",
            ],
            how="left",
        )
    )


    comparison[
        "delta_roc_auc"
    ] = (
        comparison[
            "roc_auc"
        ]
        -
        comparison[
            "source_roc_auc"
        ]
    )


    comparison[
        "delta_pr_auc"
    ] = (
        comparison[
            "pr_auc"
        ]
        -
        comparison[
            "source_pr_auc"
        ]
    )


    comparison[
        "delta_balanced_accuracy"
    ] = (
        comparison[
            "balanced_accuracy"
        ]
        -
        comparison[
            "source_balanced_accuracy"
        ]
    )


    # Positive means improvement.
    comparison[
        "brier_improvement"
    ] = (
        comparison[
            "source_brier"
        ]
        -
        comparison[
            "brier"
        ]
    )


    comparison[
        "negative_adaptation_auc"
    ] = (
        comparison[
            "delta_roc_auc"
        ]
        < -1e-8
    )


    comparison_path = (
        RESULT_DIR
        / "multiseed_sfda_deltas.csv"
    )


    comparison.to_csv(
        comparison_path,
        index=False,
    )


    # ========================================================
    # Summary
    # ========================================================

    summary = (
        comparison
        .groupby(
            [
                "target",
                "method",
            ],
            as_index=False,
        )
        .agg(
            mean_delta_auc=(
                "delta_roc_auc",
                "mean",
            ),

            sd_delta_auc=(
                "delta_roc_auc",
                "std",
            ),

            mean_brier_improvement=(
                "brier_improvement",
                "mean",
            ),

            sd_brier_improvement=(
                "brier_improvement",
                "std",
            ),

            mean_delta_balanced_accuracy=(
                "delta_balanced_accuracy",
                "mean",
            ),

            negative_adaptation_rate=(
                "negative_adaptation_auc",
                "mean",
            ),
        )
    )


    summary[
        "negative_adaptation_rate_pct"
    ] = (
        100.0
        * summary[
            "negative_adaptation_rate"
        ]
    )


    summary_path = (
        RESULT_DIR
        / "multiseed_sfda_summary.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    print(
        "\n"
        + "=" * 110
    )

    print(
        "MULTI-SEED SUMMARY"
    )

    print(
        "=" * 110
    )


    print(
        summary[
            [
                "target",
                "method",
                "mean_delta_auc",
                "sd_delta_auc",
                "mean_brier_improvement",
                "mean_delta_balanced_accuracy",
                "negative_adaptation_rate_pct",
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nSaved raw results:",
        raw_path
    )

    print(
        "Saved deltas:",
        comparison_path
    )

    print(
        "Saved summary:",
        summary_path
    )


if __name__ == "__main__":
    main()