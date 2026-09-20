from pathlib import Path
import copy
import sys

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Project
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from heart_sfda.data.preprocessing import (
    CORE8,
    build_linear_preprocessor,
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


# ============================================================
# Configuration
# ============================================================

# Source-only model-selection seeds.
# These do NOT use target data.
CV_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

N_SPLITS = 5

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
    / "statistics"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DOMAINS = {
    "cleveland":
        "cleveland.csv",

    "hungary":
        "hungary.csv",

    "switzerland":
        "switzerland.csv",

    "va_long_beach":
        "va_long_beach.csv",
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
        batch_size=BATCH_SIZE,
        shuffle=True,
    )


# ============================================================
# One source-CV fold
# ============================================================

def train_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    seed,
):

    set_seed(seed)


    model = HeartMLP(
        input_dim=
            X_train.shape[1],

        hidden_dims=
            (32, 16),

        dropout=
            0.20,
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    criterion = (
        nn.BCEWithLogitsLoss()
    )


    train_loader = make_loader(
        X_train,
        y_train,
    )


    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
        device=device,
    )


    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32,
        device=device,
    )


    best_loss = np.inf

    best_epoch = 0

    best_state = copy.deepcopy(
        model.state_dict()
    )

    patience_counter = 0


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()


        for (
            X_batch,
            y_batch,
        ) in train_loader:

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


        # ----------------------------------------------------
        # Validation BCE
        # ----------------------------------------------------

        model.eval()


        with torch.no_grad():

            val_logits = model(
                X_val_tensor
            )


            val_loss = criterion(
                val_logits,
                y_val_tensor,
            ).item()


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            val_loss
            <
            best_loss - MIN_DELTA
        ):

            best_loss = val_loss

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1


        if (
            patience_counter
            >= PATIENCE
        ):
            break


    model.load_state_dict(
        best_state
    )


    return {
        "best_epoch":
            int(best_epoch),

        "best_validation_bce":
            float(best_loss),

        "stopped_epoch":
            int(epoch),
    }


# ============================================================
# Main
# ============================================================

def main():

    device = get_device()


    fold_rows = []

    plan_rows = []


    print(
        "\n"
        + "=" * 110
    )

    print(
        "SOURCE-SPECIFIC TRAINING "
        "EPOCH ESTIMATION"
    )

    print(
        "=" * 110
    )


    print(
        "\nNo target-domain information "
        "is used in this experiment."
    )


    # ========================================================
    # Each hospital acts only as a source here
    # ========================================================

    for (
        source_name,
        filename,
    ) in DOMAINS.items():

        print(
            "\n"
            + "=" * 110
        )

        print(
            f"SOURCE: "
            f"{source_name.upper()}"
        )

        print(
            "=" * 110
        )


        df = load_dataset(
            filename
        )


        X = (
            df[
                CORE8
            ]
            .copy()
        )


        y = (
            df[
                "target"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )


        positive_n = int(
            y.sum()
        )


        negative_n = int(
            len(y)
            -
            positive_n
        )


        print(
            f"N={len(y)} | "
            f"positive={positive_n} | "
            f"negative={negative_n}"
        )


        if (
            min(
                positive_n,
                negative_n,
            )
            < 10
        ):

            print(
                "WARNING: "
                "minority-class sample count "
                "is very small; epoch estimates "
                "may be unstable."
            )


        domain_best_epochs = []


        # ====================================================
        # Multiple source-only CV seeds
        # ====================================================

        for cv_seed in CV_SEEDS:

            cv = StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=cv_seed,
            )


            print(
                f"\nCV seed {cv_seed}"
            )


            for fold, (
                train_index,
                val_index,
            ) in enumerate(
                cv.split(
                    X,
                    y,
                ),
                start=1,
            ):

                X_train_df = (
                    X.iloc[
                        train_index
                    ]
                )


                X_val_df = (
                    X.iloc[
                        val_index
                    ]
                )


                y_train = (
                    y[
                        train_index
                    ]
                    .astype(
                        np.float32
                    )
                )


                y_val = (
                    y[
                        val_index
                    ]
                    .astype(
                        np.float32
                    )
                )


                # --------------------------------------------
                # CRITICAL:
                # preprocessing fitted ONLY on source
                # training fold.
                # --------------------------------------------

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


                training_seed = (
                    cv_seed * 1000
                    + fold
                )


                result = train_fold(
                    X_train=
                        X_train,

                    y_train=
                        y_train,

                    X_val=
                        X_val,

                    y_val=
                        y_val,

                    device=
                        device,

                    seed=
                        training_seed,
                )


                domain_best_epochs.append(
                    result[
                        "best_epoch"
                    ]
                )


                fold_rows.append(
                    {
                        "source":
                            source_name,

                        "cv_seed":
                            cv_seed,

                        "fold":
                            fold,

                        "train_n":
                            len(
                                train_index
                            ),

                        "val_n":
                            len(
                                val_index
                            ),

                        "train_positive_n":
                            int(
                                y_train.sum()
                            ),

                        "train_negative_n":
                            int(
                                len(y_train)
                                - y_train.sum()
                            ),

                        "val_positive_n":
                            int(
                                y_val.sum()
                            ),

                        "val_negative_n":
                            int(
                                len(y_val)
                                - y_val.sum()
                            ),

                        **result,
                    }
                )


                print(
                    f"  Fold {fold}: "
                    f"best_epoch="
                    f"{result['best_epoch']:3d} | "
                    f"BCE="
                    f"{result['best_validation_bce']:.4f}"
                )


                # Keep GPU memory clean between fits.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


        # ====================================================
        # Robust epoch summary
        # ====================================================

        epochs = np.asarray(
            domain_best_epochs,
            dtype=float,
        )


        median_epoch = int(
            round(
                np.median(
                    epochs
                )
            )
        )


        median_epoch = max(
            median_epoch,
            1,
        )


        q25 = float(
            np.percentile(
                epochs,
                25,
            )
        )


        q75 = float(
            np.percentile(
                epochs,
                75,
            )
        )


        plan_rows.append(
            {
                "source":
                    source_name,

                "n":
                    len(y),

                "positive_n":
                    positive_n,

                "negative_n":
                    negative_n,

                "positive_pct":
                    100.0
                    * positive_n
                    / len(y),

                "cv_runs":
                    len(
                        domain_best_epochs
                    ),

                "median_best_epoch":
                    median_epoch,

                "mean_best_epoch":
                    float(
                        epochs.mean()
                    ),

                "sd_best_epoch":
                    float(
                        epochs.std(
                            ddof=1
                        )
                    ),

                "q25_best_epoch":
                    q25,

                "q75_best_epoch":
                    q75,

                "min_best_epoch":
                    int(
                        epochs.min()
                    ),

                "max_best_epoch":
                    int(
                        epochs.max()
                    ),
            }
        )


        print(
            "\nSource epoch summary:"
        )

        print(
            f"Median = "
            f"{median_epoch}"
        )

        print(
            f"Mean = "
            f"{epochs.mean():.2f}"
        )

        print(
            f"IQR = "
            f"[{q25:.1f}, {q75:.1f}]"
        )

        print(
            f"Range = "
            f"[{int(epochs.min())}, "
            f"{int(epochs.max())}]"
        )


    # ========================================================
    # Save
    # ========================================================

    fold_df = pd.DataFrame(
        fold_rows
    )


    plan_df = pd.DataFrame(
        plan_rows
    )


    fold_path = (
        RESULT_DIR
        / "source_epoch_cv_runs.csv"
    )


    plan_path = (
        RESULT_DIR
        / "source_epoch_plan.csv"
    )


    fold_df.to_csv(
        fold_path,
        index=False,
    )


    plan_df.to_csv(
        plan_path,
        index=False,
    )


    print(
        "\n"
        + "=" * 120
    )

    print(
        "SOURCE EPOCH PLAN"
    )

    print(
        "=" * 120
    )


    print(
        plan_df.to_string(
            index=False
        )
    )


    print(
        "\nSaved fold runs:",
        fold_path
    )


    print(
        "Saved source plan:",
        plan_path
    )


if __name__ == "__main__":
    main()