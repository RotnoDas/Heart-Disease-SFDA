from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Project path
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from heart_sfda.data.preprocessing import CORE8

from heart_sfda.models.mlp import HeartMLP

from heart_sfda.adaptation.reliability_gate import (
    adapt_reliability_gated,
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
# Fixed experiment configuration
# ============================================================

SEEDS = list(range(10))

SOURCE_EPOCHS = 22

SOURCE_LR = 1e-3
SOURCE_WEIGHT_DECAY = 1e-4
SOURCE_BATCH_SIZE = 32

N_FOLDS = 5
FOLD_RANDOM_STATE = 42


# ============================================================
# Reliability-gated v1 configuration
# ============================================================

SELECTION_FRACTION = 0.40

MINIMUM_PER_CLASS = 8

ADAPT_LR = 5e-5

ADAPT_WEIGHT_DECAY = 1e-4

ADAPT_EPOCHS = 30


# ============================================================
# Target-specific deterministic offsets
#
# IMPORTANT:
# These do NOT depend on the ablation variant.
#
# Therefore:
#
# same source seed
# + same target
# + same fold
#
# => same adaptation random seed
#
# across all ablation variants.
# ============================================================

TARGET_SEED_OFFSET = {
    "hungary": 100,
    "switzerland": 200,
    "va_long_beach": 300,
}


# ============================================================
# Loss ablations
#
# Full v1:
#
# lambda_pseudo  = 1.0
# lambda_entropy = 0.05
# lambda_anchor  = 1.0
# lambda_prior   = 0.50
#
# Here one component is removed at a time.
# ============================================================

ABLATIONS = {

    "no_pseudo": {
        "lambda_pseudo": 0.0,
        "lambda_entropy": 0.05,
        "lambda_anchor": 1.0,
        "lambda_prior": 0.50,
    },

    "no_entropy": {
        "lambda_pseudo": 1.0,
        "lambda_entropy": 0.0,
        "lambda_anchor": 1.0,
        "lambda_prior": 0.50,
    },

    "no_anchor": {
        "lambda_pseudo": 1.0,
        "lambda_entropy": 0.05,
        "lambda_anchor": 0.0,
        "lambda_prior": 0.50,
    },

    "no_prior": {
        "lambda_pseudo": 1.0,
        "lambda_entropy": 0.05,
        "lambda_anchor": 1.0,
        "lambda_prior": 0.0,
    },
}


# ============================================================
# Paths
# ============================================================

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
    "hungary":
        "hungary.csv",

    "switzerland":
        "switzerland.csv",

    "va_long_beach":
        "va_long_beach.csv",
}


# ============================================================
# Data utilities
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


# ============================================================
# Source model training
# ============================================================

def train_source_model(
    X,
    y,
    seed,
    device,
):

    # --------------------------------------------------------
    # Source initialization/training controlled by source seed
    # --------------------------------------------------------

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


# ============================================================
# Prediction
# ============================================================

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

        logits = model(
            X_tensor
        )


        probability = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )


    return probability


# ============================================================
# Cross-fitted ablation
# ============================================================

def crossfit_ablation(
    source_model,
    X_target,
    X_target_raw,
    device,
    config,
    adapt_seed_base,
):

    """
    Cross-fitted target adaptation.

    Important fairness rule:

    Every ablation variant receives the same
    adaptation seed for the same:

        source seed
        + target
        + fold

    This controls Dropout randomness.
    """

    kfold = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=FOLD_RANDOM_STATE,
    )


    probability = np.zeros(
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

        # ----------------------------------------------------
        # Adaptation data
        #
        # Features only.
        # No target labels used here.
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Missingness measured from raw CORE8 target features
        #
        # Still label-free.
        # ----------------------------------------------------

        missingness_rate = float(
            X_target_raw
            .iloc[
                adapt_indices
            ]
            .isna()
            .mean()
            .mean()
        )


        # ----------------------------------------------------
        # CRITICAL RNG CONTROL
        #
        # Same source seed + target + fold
        # gets the exact same adaptation seed
        # for every ablation variant.
        #
        # Variant name is intentionally NOT used.
        # ----------------------------------------------------

        adaptation_seed = (
            adapt_seed_base
            + fold
        )


        set_seed(
            adaptation_seed
        )


        # ----------------------------------------------------
        # Adapt
        #
        # adapt_reliability_gated internally deep-copies
        # the source model, so source_model itself remains
        # unchanged.
        # ----------------------------------------------------

        adapted_model, _ = (
            adapt_reliability_gated(
                source_model=
                    source_model,

                X_target_unlabeled=
                    X_adapt,

                device=device,

                missingness_rate=
                    missingness_rate,

                selection_fraction=
                    SELECTION_FRACTION,

                minimum_per_class=
                    MINIMUM_PER_CLASS,

                learning_rate=
                    ADAPT_LR,

                weight_decay=
                    ADAPT_WEIGHT_DECAY,

                epochs=
                    ADAPT_EPOCHS,

                lambda_pseudo=
                    config[
                        "lambda_pseudo"
                    ],

                lambda_entropy=
                    config[
                        "lambda_entropy"
                    ],

                lambda_anchor=
                    config[
                        "lambda_anchor"
                    ],

                lambda_prior=
                    config[
                        "lambda_prior"
                    ],
            )
        )


        # ----------------------------------------------------
        # Evaluate only held-out target patients
        # ----------------------------------------------------

        probability[
            test_indices
        ] = predict_probability(
            adapted_model,
            X_test,
            device,
        )


    return probability


# ============================================================
# Main experiment
# ============================================================

def main():

    device = get_device()


    # --------------------------------------------------------
    # Source-fitted preprocessor
    # --------------------------------------------------------

    preprocessor = joblib.load(
        SOURCE_MODEL_DIR
        / "cleveland_core8_preprocessor.joblib"
    )


    # ========================================================
    # Cleveland source data
    # ========================================================

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
        + "=" * 120
    )

    print(
        "RELIABILITY-GATED V1 "
        "LOSS ABLATION — RNG CONTROLLED"
    )

    print(
        "=" * 120
    )


    print(
        "\nSeeds:",
        SEEDS
    )


    print(
        "Ablations:",
        list(
            ABLATIONS.keys()
        )
    )


    # ========================================================
    # Source seeds
    # ========================================================

    for seed in SEEDS:

        print(
            "\n"
            + "#" * 120
        )

        print(
            f"SOURCE SEED {seed}"
        )

        print(
            "#" * 120
        )


        # ----------------------------------------------------
        # Train ONE source model for this seed
        # ----------------------------------------------------

        source_model = (
            train_source_model(
                X_source,
                y_source,
                seed,
                device,
            )
        )


        # ====================================================
        # Target domains
        # ====================================================

        for (
            target,
            filename,
        ) in TARGETS.items():

            print(
                f"\nTarget: {target}"
            )


            target_df = load_dataset(
                filename
            )


            X_target_raw = (
                target_df[
                    CORE8
                ]
                .copy()
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


            # ------------------------------------------------
            # Labels are used ONLY for final evaluation.
            # ------------------------------------------------

            y_target = (
                target_df[
                    "target"
                ]
                .to_numpy()
            )


            # =================================================
            # Source-only reference
            # =================================================

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


            print(
                "Source-only:"
                f" AUROC="
                f"{source_metrics['roc_auc']:.4f}"
                f" Brier="
                f"{source_metrics['brier']:.4f}"
            )


            # ------------------------------------------------
            # Same seed base across ALL ablations
            #
            # Example:
            #
            # seed 3 + Switzerland:
            # 3 * 10000 + 200
            #
            # folds:
            # +1, +2, +3, +4, +5
            #
            # No variant-specific offset.
            # ------------------------------------------------

            adapt_seed_base = (
                seed * 10000
                +
                TARGET_SEED_OFFSET[
                    target
                ]
            )


            # =================================================
            # Ablations
            # =================================================

            for (
                variant,
                config,
            ) in ABLATIONS.items():

                adapted_probability = (
                    crossfit_ablation(
                        source_model=
                            source_model,

                        X_target=
                            X_target,

                        X_target_raw=
                            X_target_raw,

                        device=device,

                        config=
                            config,

                        adapt_seed_base=
                            adapt_seed_base,
                    )
                )


                adapted_metrics = (
                    compute_binary_metrics(
                        y_target,
                        adapted_probability,
                    )
                )


                # ------------------------------------------------
                # Paired differences against same source model
                # ------------------------------------------------

                delta_auc = (
                    adapted_metrics[
                        "roc_auc"
                    ]
                    -
                    source_metrics[
                        "roc_auc"
                    ]
                )


                delta_pr_auc = (
                    adapted_metrics[
                        "pr_auc"
                    ]
                    -
                    source_metrics[
                        "pr_auc"
                    ]
                )


                delta_balanced_accuracy = (
                    adapted_metrics[
                        "balanced_accuracy"
                    ]
                    -
                    source_metrics[
                        "balanced_accuracy"
                    ]
                )


                # Positive = adapted model improved Brier.
                brier_improvement = (
                    source_metrics[
                        "brier"
                    ]
                    -
                    adapted_metrics[
                        "brier"
                    ]
                )


                negative_adaptation_auc = (
                    delta_auc
                    < -1e-8
                )


                brier_worsened = (
                    brier_improvement
                    < -1e-8
                )


                rows.append(
                    {
                        "seed":
                            seed,

                        "target":
                            target,

                        "variant":
                            variant,

                        # Source metrics
                        "source_roc_auc":
                            source_metrics[
                                "roc_auc"
                            ],

                        "source_pr_auc":
                            source_metrics[
                                "pr_auc"
                            ],

                        "source_balanced_accuracy":
                            source_metrics[
                                "balanced_accuracy"
                            ],

                        "source_brier":
                            source_metrics[
                                "brier"
                            ],

                        # Adapted metrics
                        "roc_auc":
                            adapted_metrics[
                                "roc_auc"
                            ],

                        "pr_auc":
                            adapted_metrics[
                                "pr_auc"
                            ],

                        "balanced_accuracy":
                            adapted_metrics[
                                "balanced_accuracy"
                            ],

                        "brier":
                            adapted_metrics[
                                "brier"
                            ],

                        # Paired deltas
                        "delta_roc_auc":
                            delta_auc,

                        "delta_pr_auc":
                            delta_pr_auc,

                        "delta_balanced_accuracy":
                            delta_balanced_accuracy,

                        "brier_improvement":
                            brier_improvement,

                        # Event indicators
                        "negative_adaptation_auc":
                            negative_adaptation_auc,

                        "brier_worsened":
                            brier_worsened,
                    }
                )


                print(
                    f"  {variant:10s}"
                    f" ΔAUROC="
                    f"{delta_auc:+.6f}"
                    f" BrierImp="
                    f"{brier_improvement:+.6f}"
                    f" ΔBalAcc="
                    f"{delta_balanced_accuracy:+.6f}"
                )


    # ========================================================
    # Raw results
    # ========================================================

    result_df = pd.DataFrame(
        rows
    )


    raw_path = (
        RESULT_DIR
        / "reliability_v1_loss_ablation_rng_controlled.csv"
    )


    result_df.to_csv(
        raw_path,
        index=False,
    )


    # ========================================================
    # Summary
    # ========================================================

    summary = (
        result_df
        .groupby(
            [
                "target",
                "variant",
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

            mean_abs_delta_auc=(
                "delta_roc_auc",
                lambda x:
                    float(
                        np.mean(
                            np.abs(x)
                        )
                    ),
            ),

            mean_delta_pr_auc=(
                "delta_pr_auc",
                "mean",
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

            brier_worsening_rate=(
                "brier_worsened",
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


    summary[
        "brier_worsening_rate_pct"
    ] = (
        100.0
        * summary[
            "brier_worsening_rate"
        ]
    )


    # ========================================================
    # Save summary
    # ========================================================

    summary_path = (
        RESULT_DIR
        / "reliability_v1_loss_ablation_rng_controlled_summary.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    # ========================================================
    # Print final summary
    # ========================================================

    print(
        "\n"
        + "=" * 140
    )

    print(
        "LOSS ABLATION SUMMARY — RNG CONTROLLED"
    )

    print(
        "=" * 140
    )


    display_columns = [

        "target",

        "variant",

        "mean_delta_auc",

        "sd_delta_auc",

        "mean_abs_delta_auc",

        "mean_delta_pr_auc",

        "mean_brier_improvement",

        "sd_brier_improvement",

        "mean_delta_balanced_accuracy",

        "negative_adaptation_rate_pct",

        "brier_worsening_rate_pct",
    ]


    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )


    print(
        "\nSaved raw results:",
        raw_path
    )


    print(
        "Saved summary:",
        summary_path
    )


if __name__ == "__main__":
    main()