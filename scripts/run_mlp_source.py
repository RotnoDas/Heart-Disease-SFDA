from pathlib import Path
import copy
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import (
    StratifiedKFold,
)

from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from heart_sfda.data.preprocessing import (
    CORE8,
    build_linear_preprocessor,
)

from heart_sfda.evaluation.metrics import (
    compute_binary_metrics,
)

from heart_sfda.models.mlp import (
    HeartMLP,
)

from heart_sfda.utils.device import (
    get_device,
)

from heart_sfda.utils.seed import (
    set_seed,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42

MAX_EPOCHS = 300

PATIENCE = 30

MIN_DELTA = 1e-4

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4


DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "cross_domain"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "source"
    / "mlp"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGETS = {
    "hungary":
        "hungary.csv",

    "switzerland":
        "switzerland.csv",

    "va_long_beach":
        "va_long_beach.csv",
}


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_dataset(filename):

    return pd.read_csv(
        DATA_DIR / filename
    )


def to_tensor_dataset(
    X,
    y,
):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        np.asarray(y),
        dtype=torch.float32,
    )

    return TensorDataset(
        X_tensor,
        y_tensor,
    )


def make_loader(
    X,
    y,
    batch_size,
    shuffle,
):

    dataset = to_tensor_dataset(
        X,
        y,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_probability(
    model,
    X,
    device,
):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():

        logits = model(
            X_tensor
        )

        probability = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

    return probability


# --------------------------------------------------
# Fold training
# --------------------------------------------------

def train_with_validation(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    seed,
):

    set_seed(seed)

    input_dim = X_train.shape[1]

    model = HeartMLP(
        input_dim=input_dim,
        hidden_dims=(32, 16),
        dropout=0.20,
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    criterion = nn.BCEWithLogitsLoss()


    train_loader = make_loader(
        X_train,
        y_train,
        BATCH_SIZE,
        True,
    )


    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
    ).to(device)

    y_val_tensor = torch.tensor(
        np.asarray(y_val),
        dtype=torch.float32,
    ).to(device)


    best_loss = np.inf

    best_epoch = 0

    best_state = None

    patience_counter = 0


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()


        for X_batch, y_batch in train_loader:

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


        # ------------------------------------------
        # Validation loss
        # ------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_val_tensor
            )

            val_loss = criterion(
                val_logits,
                y_val_tensor,
            ).item()


        if (
            val_loss
            < best_loss - MIN_DELTA
        ):

            best_loss = val_loss

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1


        if patience_counter >= PATIENCE:
            break


    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_epoch,
        best_loss,
    )


# --------------------------------------------------
# Final training on full Cleveland
# --------------------------------------------------

def train_fixed_epochs(
    X,
    y,
    epochs,
    device,
):

    set_seed(SEED)

    input_dim = X.shape[1]

    model = HeartMLP(
        input_dim=input_dim,
        hidden_dims=(32, 16),
        dropout=0.20,
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    criterion = nn.BCEWithLogitsLoss()


    loader = make_loader(
        X,
        y,
        BATCH_SIZE,
        True,
    )


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


    return model


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    set_seed(SEED)

    device = get_device()


    source_df = load_dataset(
        "cleveland.csv"
    )


    X_source = source_df[
        CORE8
    ]

    y_source = source_df[
        "target"
    ].to_numpy()


    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )


    oof_probability = np.zeros(
        len(source_df),
        dtype=float,
    )


    best_epochs = []


    print("\n" + "=" * 100)

    print(
        "PYTORCH MLP SOURCE-ONLY BASELINE"
    )

    print("=" * 100)


    # --------------------------------------------------
    # Source 5-fold CV
    # --------------------------------------------------

    for fold, (
        train_index,
        val_index,
    ) in enumerate(
        cv.split(
            X_source,
            y_source,
        ),
        start=1,
    ):

        print(
            f"\nFold {fold}/5"
        )


        X_train_df = (
            X_source.iloc[
                train_index
            ]
        )

        X_val_df = (
            X_source.iloc[
                val_index
            ]
        )


        y_train = (
            y_source[
                train_index
            ]
        )

        y_val = (
            y_source[
                val_index
            ]
        )


        # ------------------------------------------
        # Fit preprocessing ONLY on training fold
        # ------------------------------------------

        preprocessor = (
            build_linear_preprocessor()
        )


        X_train = (
            preprocessor
            .fit_transform(
                X_train_df
            )
            .astype(
                np.float32
            )
        )


        X_val = (
            preprocessor
            .transform(
                X_val_df
            )
            .astype(
                np.float32
            )
        )


        model, best_epoch, best_loss = (
            train_with_validation(
                X_train,
                y_train,
                X_val,
                y_val,
                device,
                SEED + fold,
            )
        )


        probability = (
            predict_probability(
                model,
                X_val,
                device,
            )
        )


        oof_probability[
            val_index
        ] = probability


        best_epochs.append(
            best_epoch
        )


        print(
            f"Best epoch: "
            f"{best_epoch}"
        )

        print(
            f"Validation BCE: "
            f"{best_loss:.4f}"
        )


    # --------------------------------------------------
    # Source OOF metrics
    # --------------------------------------------------

    source_metrics = (
        compute_binary_metrics(
            y_source,
            oof_probability,
        )
    )


    print("\n" + "-" * 100)

    print(
        "CLEVELAND OOF RESULTS"
    )

    print("-" * 100)


    print(
        f"ROC-AUC: "
        f"{source_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{source_metrics['pr_auc']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{source_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Brier: "
        f"{source_metrics['brier']:.4f}"
    )


    # --------------------------------------------------
    # Determine final training length
    # --------------------------------------------------

    final_epochs = int(
        round(
            np.median(
                best_epochs
            )
        )
    )


    final_epochs = max(
        final_epochs,
        1,
    )


    print(
        "\nCV best epochs:",
        best_epochs,
    )

    print(
        "Final training epochs:",
        final_epochs,
    )


    # --------------------------------------------------
    # Fit final source preprocessing
    # --------------------------------------------------

    final_preprocessor = (
        build_linear_preprocessor()
    )


    X_source_final = (
        final_preprocessor
        .fit_transform(
            X_source
        )
        .astype(
            np.float32
        )
    )


    # --------------------------------------------------
    # Train final Cleveland source MLP
    # --------------------------------------------------

    final_model = (
        train_fixed_epochs(
            X_source_final,
            y_source,
            final_epochs,
            device,
        )
    )


    # --------------------------------------------------
    # Save source preprocessor
    # --------------------------------------------------

    preprocessing_path = (
        MODEL_DIR
        / "cleveland_core8_preprocessor.joblib"
    )


    joblib.dump(
        final_preprocessor,
        preprocessing_path,
    )


    # --------------------------------------------------
    # Save neural model
    # --------------------------------------------------

    model_path = (
        MODEL_DIR
        / "cleveland_core8_mlp.pt"
    )


    torch.save(
        {
            "model_state_dict":
                final_model.state_dict(),

            "input_dim":
                X_source_final.shape[1],

            "hidden_dims":
                [32, 16],

            "dropout":
                0.20,

            "feature_set":
                "CORE8",

            "source_domain":
                "cleveland",

            "training_epochs":
                final_epochs,

            "seed":
                SEED,
        },
        model_path,
    )


    # --------------------------------------------------
    # Prepare result table
    # --------------------------------------------------

    results = []


    source_metrics.update(
        {
            "model": "mlp",
            "method": "source_only",
            "feature_set": "CORE8",
            "source": "cleveland",
            "evaluation_domain":
                "cleveland_oof",
        }
    )


    results.append(
        source_metrics.copy()
    )


    # --------------------------------------------------
    # Evaluate target domains
    # --------------------------------------------------

    for (
        target_name,
        filename,
    ) in TARGETS.items():

        target_df = load_dataset(
            filename
        )


        X_target_df = target_df[
            CORE8
        ]


        y_target = (
            target_df[
                "target"
            ]
            .to_numpy()
        )


        # Source-fitted preprocessing ONLY
        X_target = (
            final_preprocessor
            .transform(
                X_target_df
            )
            .astype(
                np.float32
            )
        )


        probability = (
            predict_probability(
                final_model,
                X_target,
                device,
            )
        )


        metrics = (
            compute_binary_metrics(
                y_target,
                probability,
            )
        )


        metrics.update(
            {
                "model":
                    "mlp",

                "method":
                    "source_only",

                "feature_set":
                    "CORE8",

                "source":
                    "cleveland",

                "evaluation_domain":
                    target_name,
            }
        )


        results.append(
            metrics
        )


        print(
            "\nCleveland ->",
            target_name
        )

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC: "
            f"{metrics['pr_auc']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Sensitivity: "
            f"{metrics['sensitivity']:.4f}"
        )

        print(
            f"Specificity: "
            f"{metrics['specificity']:.4f}"
        )

        print(
            f"Brier: "
            f"{metrics['brier']:.4f}"
        )


    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    result_df = pd.DataFrame(
        results
    )


    output_path = (
        RESULT_DIR
        / "source_only_mlp_core8.csv"
    )


    result_df.to_csv(
        output_path,
        index=False,
    )


    epoch_path = (
        RESULT_DIR
        / "mlp_cv_best_epochs.csv"
    )


    pd.DataFrame(
        {
            "fold":
                range(
                    1,
                    len(best_epochs) + 1,
                ),

            "best_epoch":
                best_epochs,
        }
    ).to_csv(
        epoch_path,
        index=False,
    )


    print("\n" + "=" * 100)

    print(
        "MLP RESULTS"
    )

    print("=" * 100)


    print(
        result_df[
            [
                "evaluation_domain",
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "brier",
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nSaved model:",
        model_path
    )

    print(
        "Saved preprocessor:",
        preprocessing_path
    )

    print(
        "Saved results:",
        output_path
    )


if __name__ == "__main__":
    main()