from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import KFold

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

from heart_sfda.models.dual_head_mlp import (
    DualHeadHeartMLP,
)

from heart_sfda.adaptation.dlar_lcl import (
    adapt_dlar_lcl,
    predict_dlar_lcl_probability,
)

from heart_sfda.evaluation.metrics import (
    compute_binary_metrics,
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

SOURCE = "cleveland"
TARGET = "hungary"

SEED = 0


# Released repository source-training defaults.
SOURCE_EPOCHS = 20

SOURCE_BATCH_SIZE = 64

SOURCE_LR = 1e-3

SOURCE_WEIGHT_DECAY = 5e-4

SOURCE_ALPHA_DISCREPANCY = 0.5


# Target adaptation defaults from released implementation.
TARGET_LR = 1e-4

TARGET_WEIGHT_DECAY = 5e-4

DLAR_EPOCHS = 5

LCL_EPOCHS = 10

BETA = 1.0

GAMMA = 1.0

K_NEIGHBORS = 5


N_TARGET_FOLDS = 5

TARGET_FOLD_SEED = 42


DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "dlar_lcl"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FILES = {
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

def load_domain(
    domain,
):
    return pd.read_csv(
        DATA_DIR
        / FILES[
            domain
        ]
    )


def make_source_loader(
    X,
    y,
):

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32,
        ),
        torch.tensor(
            y,
            dtype=torch.long,
        ),
    )


    return DataLoader(
        dataset,
        batch_size=
            SOURCE_BATCH_SIZE,
        shuffle=True,
    )


# ============================================================
# Dual-head source pretraining
# ============================================================

def train_source_model(
    X_source,
    y_source,
    device,
):

    set_seed(
        SEED
    )


    model = DualHeadHeartMLP(
        input_dim=
            X_source.shape[1],

        hidden_dims=
            (32, 16),

        dropout=
            0.20,
    ).to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=
            SOURCE_LR,
        weight_decay=
            SOURCE_WEIGHT_DECAY,
    )


    loader = make_source_loader(
        X_source,
        y_source,
    )


    epoch_rows = []


    for epoch in range(
        1,
        SOURCE_EPOCHS + 1,
    ):

        model.train()


        epoch_loss = []

        epoch_ce = []

        epoch_discrepancy = []


        for (
            X_batch,
            y_batch,
        ) in loader:

            X_batch = (
                X_batch.to(
                    device
                )
            )

            y_batch = (
                y_batch.to(
                    device
                )
            )


            optimizer.zero_grad()


            (
                _,
                logits_1,
                logits_2,
            ) = model(
                X_batch
            )


            loss_1 = (
                F.cross_entropy(
                    logits_1,
                    y_batch,
                )
            )


            loss_2 = (
                F.cross_entropy(
                    logits_2,
                    y_batch,
                )
            )


            supervised_loss = (
                loss_1
                + loss_2
            )


            probability_1 = (
                F.softmax(
                    logits_1,
                    dim=1,
                )
            )


            probability_2 = (
                F.softmax(
                    logits_2,
                    dim=1,
                )
            )


            # Released code discrepancy:
            # mean absolute difference between
            # the two classifier distributions.
            discrepancy_loss = (
                torch.mean(
                    torch.abs(
                        probability_1
                        - probability_2
                    )
                )
            )


            # ------------------------------------------------
            # Released source implementation applies
            # exp(CE1 + CE2), described there as group-DRO.
            #
            # We preserve it for this baseline smoke test.
            # ------------------------------------------------

            classification_loss = (
                torch.exp(
                    supervised_loss
                )
            )


            total_loss = (
                classification_loss
                +
                SOURCE_ALPHA_DISCREPANCY
                * discrepancy_loss
            )


            if not torch.isfinite(
                total_loss
            ):

                raise RuntimeError(
                    "Non-finite source training loss."
                )


            total_loss.backward()

            optimizer.step()


            epoch_loss.append(
                float(
                    total_loss.item()
                )
            )

            epoch_ce.append(
                float(
                    supervised_loss.item()
                )
            )

            epoch_discrepancy.append(
                float(
                    discrepancy_loss.item()
                )
            )


        row = {
            "epoch":
                epoch,

            "loss":
                float(
                    np.mean(
                        epoch_loss
                    )
                ),

            "ce_sum":
                float(
                    np.mean(
                        epoch_ce
                    )
                ),

            "classifier_discrepancy":
                float(
                    np.mean(
                        epoch_discrepancy
                    )
                ),
        }


        epoch_rows.append(
            row
        )


        print(
            f"Source epoch {epoch:2d}/"
            f"{SOURCE_EPOCHS}"
            f" | loss={row['loss']:.4f}"
            f" | CE={row['ce_sum']:.4f}"
            f" | discrepancy="
            f"{row['classifier_discrepancy']:.4f}"
        )


    model.eval()


    return (
        model,
        pd.DataFrame(
            epoch_rows
        ),
    )


# ============================================================
# Main
# ============================================================

def main():

    device = get_device()


    print(
        "\n"
        + "=" * 110
    )

    print(
        "DLAR-LCL TABULAR PORT — SMOKE TEST"
    )

    print(
        "=" * 110
    )


    print(
        f"\nSource: {SOURCE}"
    )

    print(
        f"Target: {TARGET}"
    )

    print(
        f"Seed:   {SEED}"
    )


    # ========================================================
    # Data
    # ========================================================

    source_df = load_domain(
        SOURCE
    )


    target_df = load_domain(
        TARGET
    )


    X_source_raw = (
        source_df[
            CORE8
        ]
        .copy()
    )


    X_target_raw = (
        target_df[
            CORE8
        ]
        .copy()
    )


    y_source = (
        source_df[
            "target"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )


    y_target = (
        target_df[
            "target"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )


    # --------------------------------------------------------
    # Preprocessor fitted on SOURCE only.
    # --------------------------------------------------------

    preprocessor = (
        build_linear_preprocessor()
    )


    X_source = (
        preprocessor
        .fit_transform(
            X_source_raw
        )
        .astype(
            np.float32
        )
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


    print(
        "\nTransformed input dimension:",
        X_source.shape[1]
    )


    # ========================================================
    # Source training
    # ========================================================

    (
        source_model,
        training_history,
    ) = train_source_model(
        X_source,
        y_source,
        device,
    )


    history_path = (
        RESULT_DIR
        / "smoke_source_training_history.csv"
    )


    training_history.to_csv(
        history_path,
        index=False,
    )


    # ========================================================
    # Source-only target evaluation
    # ========================================================

    source_probability = (
        predict_dlar_lcl_probability(
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


    print(
        "\n"
        + "-" * 110
    )

    print(
        "DUAL-HEAD SOURCE ONLY"
    )

    print(
        "-" * 110
    )


    print(
        f"AUROC = "
        f"{source_metrics['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC = "
        f"{source_metrics['pr_auc']:.6f}"
    )

    print(
        f"Balanced Accuracy = "
        f"{source_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Brier = "
        f"{source_metrics['brier']:.6f}"
    )


    # ========================================================
    # Target-label-free cross-fitted adaptation
    # ========================================================

    kfold = KFold(
        n_splits=
            N_TARGET_FOLDS,
        shuffle=True,
        random_state=
            TARGET_FOLD_SEED,
    )


    adapted_probability = np.zeros(
        len(
            X_target
        ),
        dtype=float,
    )


    diagnostic_rows = []


    for fold, (
        adapt_indices,
        test_indices,
    ) in enumerate(
        kfold.split(
            X_target
        ),
        start=1,
    ):

        print(
            f"\nTarget fold {fold}/"
            f"{N_TARGET_FOLDS}"
        )


        # ----------------------------------------------------
        # Matched deterministic seed.
        # No target labels participate.
        # ----------------------------------------------------

        set_seed(
            1000
            + SEED * 100
            + fold
        )


        (
            adapted_model,
            diagnostics,
        ) = adapt_dlar_lcl(
            source_model=
                source_model,

            X_target_unlabeled=
                X_target[
                    adapt_indices
                ],

            device=
                device,

            learning_rate=
                TARGET_LR,

            weight_decay=
                TARGET_WEIGHT_DECAY,

            dlar_epochs=
                DLAR_EPOCHS,

            lcl_epochs=
                LCL_EPOCHS,

            beta=
                BETA,

            gamma=
                GAMMA,

            k_neighbors=
                K_NEIGHBORS,
        )


        fold_probability = (
            predict_dlar_lcl_probability(
                adapted_model,

                X_target[
                    test_indices
                ],

                device,
            )
        )


        adapted_probability[
            test_indices
        ] = (
            fold_probability
        )


        diagnostic_rows.append(
            {
                "fold":
                    fold,

                "adapt_n":
                    int(
                        len(
                            adapt_indices
                        )
                    ),

                "test_n":
                    int(
                        len(
                            test_indices
                        )
                    ),

                **diagnostics,
            }
        )


        print(
            "  confident_fraction="
            f"{diagnostics['mean_dlar_confident_fraction']:.4f}"
        )

        print(
            "  pseudo_positive_fraction="
            f"{diagnostics['mean_pseudo_positive_fraction']:.4f}"
        )

        print(
            "  mean_classifier_discrepancy="
            f"{diagnostics['mean_classifier_discrepancy']:.6f}"
        )

        print(
            "  common_neighbors="
            f"{diagnostics['mean_common_neighbors']:.4f}"
        )

        print(
            "  probability_drift="
            f"{diagnostics['probability_drift']:.6f}"
        )

        print(
            "  positive_rate="
            f"{diagnostics['positive_rate_before']:.4f}"
            " -> "
            f"{diagnostics['positive_rate_after']:.4f}"
        )


        del adapted_model


        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # ========================================================
    # Adapted evaluation
    # ========================================================

    adapted_metrics = (
        compute_binary_metrics(
            y_target,
            adapted_probability,
        )
    )


    print(
        "\n"
        + "=" * 110
    )

    print(
        "DLAR-LCL SMOKE TEST RESULT"
    )

    print(
        "=" * 110
    )


    print(
        f"Source-only AUROC: "
        f"{source_metrics['roc_auc']:.6f}"
    )

    print(
        f"Adapted AUROC:     "
        f"{adapted_metrics['roc_auc']:.6f}"
    )

    print(
        f"Delta AUROC:       "
        f"{adapted_metrics['roc_auc'] - source_metrics['roc_auc']:+.6f}"
    )


    print()


    print(
        f"Source-only Brier: "
        f"{source_metrics['brier']:.6f}"
    )

    print(
        f"Adapted Brier:     "
        f"{adapted_metrics['brier']:.6f}"
    )

    print(
        f"Brier improvement: "
        f"{source_metrics['brier'] - adapted_metrics['brier']:+.6f}"
    )


    print()


    print(
        f"Source BalAcc:     "
        f"{source_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Adapted BalAcc:    "
        f"{adapted_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Delta BalAcc:      "
        f"{adapted_metrics['balanced_accuracy'] - source_metrics['balanced_accuracy']:+.6f}"
    )


    # ========================================================
    # Save
    # ========================================================

    diagnostics_df = pd.DataFrame(
        diagnostic_rows
    )


    diagnostic_path = (
        RESULT_DIR
        / "smoke_cleveland_to_hungary_diagnostics.csv"
    )


    diagnostics_df.to_csv(
        diagnostic_path,
        index=False,
    )


    prediction_df = pd.DataFrame(
        {
            "y_true":
                y_target,

            "source_only":
                source_probability,

            "dlar_lcl":
                adapted_probability,
        }
    )


    prediction_path = (
        RESULT_DIR
        / "smoke_cleveland_to_hungary_predictions.csv"
    )


    prediction_df.to_csv(
        prediction_path,
        index=False,
    )


    summary_df = pd.DataFrame(
        [
            {
                "source":
                    SOURCE,

                "target":
                    TARGET,

                "seed":
                    SEED,

                "method":
                    "dual_head_source_only",

                **source_metrics,
            },

            {
                "source":
                    SOURCE,

                "target":
                    TARGET,

                "seed":
                    SEED,

                "method":
                    "dlar_lcl",

                **adapted_metrics,
            },
        ]
    )


    summary_path = (
        RESULT_DIR
        / "smoke_cleveland_to_hungary_summary.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False,
    )


    print(
        "\nSaved training history:",
        history_path
    )

    print(
        "Saved diagnostics:",
        diagnostic_path
    )

    print(
        "Saved predictions:",
        prediction_path
    )

    print(
        "Saved summary:",
        summary_path
    )


if __name__ == "__main__":
    main()