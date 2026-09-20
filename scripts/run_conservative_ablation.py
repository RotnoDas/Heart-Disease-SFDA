from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold

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


from heart_sfda.data.preprocessing import CORE8

from heart_sfda.models.mlp import HeartMLP

from heart_sfda.adaptation.conservative import (
    adapt_conservative,
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

SEEDS = list(range(10))

SOURCE_EPOCHS = 22

SOURCE_LR = 1e-3
SOURCE_WEIGHT_DECAY = 1e-4
SOURCE_BATCH_SIZE = 32

N_FOLDS = 5
FOLD_RANDOM_STATE = 42

SELECTION_FRACTION = 0.40
MINIMUM_PER_CLASS = 8

ADAPT_LR = 5e-5
ADAPT_WEIGHT_DECAY = 1e-4
ADAPT_EPOCHS = 30


TARGET_SEED_OFFSET = {
    "hungary": 100,
    "switzerland": 200,
    "va_long_beach": 300,
}


# ============================================================
# Candidate variants
#
# First four:
# loss-component analysis after removing pseudo loss.
#
# Last four:
# reliability-structure analysis.
# ============================================================

VARIANTS = {

    "candidate_full": {
        "reliability_mode": "full",
        "lambda_entropy": 0.05,
        "lambda_anchor": 1.0,
        "lambda_prior": 0.50,
    },

    "candidate_no_entropy": {
        "reliability_mode": "full",
        "lambda_entropy": 0.0,
        "lambda_anchor": 1.0,
        "lambda_prior": 0.50,
    },

    "candidate_no_anchor": {
        "reliability_mode": "full",
        "lambda_entropy": 0.05,
        "lambda_anchor": 0.0,
        "lambda_prior": 0.50,
    },

    "candidate_no_prior": {
        "reliability_mode": "full",
        "lambda_entropy": 0.05,
        "lambda_anchor": 1.0,
        "lambda_prior": 0.0,
    },

    "unit_reliability": {
        "reliability_mode": "unit",
        "lambda_entropy": 0.05,
        "lambda_anchor": 1.0,
        "lambda_prior": 0.50,
    },

    "no_missingness_factor": {
        "reliability_mode": "no_missingness",
        "lambda_entropy": 0.05,
        "lambda_anchor": 1.0,
        "lambda_prior": 0.50,
    },

    "no_class_support_factor": {
        "reliability_mode": "no_class_support",
        "lambda_entropy": 0.05,
        "lambda_anchor": 1.0,
        "lambda_prior": 0.50,
    },

    "no_confidence_factor": {
        "reliability_mode": "no_confidence",
        "lambda_entropy": 0.05,
        "lambda_anchor": 1.0,
        "lambda_prior": 0.50,
    },
}


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
):

    dataset = TensorDataset(

        torch.tensor(
            X,
            dtype=torch.float32,
        ),

        torch.tensor(
            y,
            dtype=torch.float32,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=
            SOURCE_BATCH_SIZE,
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
        input_dim=
            X.shape[1],
        hidden_dims=
            (32, 16),
        dropout=
            0.20,
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SOURCE_LR,
        weight_decay=
            SOURCE_WEIGHT_DECAY,
    )


    criterion = (
        nn.BCEWithLogitsLoss()
    )


    loader = make_loader(
        X,
        y,
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
                model(
                    X_tensor
                )
            )
            .cpu()
            .numpy()
        )

    return probability


# ============================================================
# Cross-fitting
# ============================================================

def run_crossfit(
    source_model,
    X_target,
    X_target_raw,
    config,
    device,
    adapt_seed_base,
):

    kfold = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=
            FOLD_RANDOM_STATE,
    )

    probabilities = np.zeros(
        len(X_target),
        dtype=float,
    )

    fold_stats = []


    for fold, (
        adapt_indices,
        test_indices,
    ) in enumerate(
        kfold.split(
            X_target
        ),
        start=1,
    ):

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
        # Matched stochastic seed across ALL variants
        # ----------------------------------------------------

        set_seed(
            adapt_seed_base
            + fold
        )


        adapted_model, stats = (
            adapt_conservative(
                source_model=
                    source_model,

                X_target_unlabeled=
                    X_adapt,

                device=
                    device,

                missingness_rate=
                    missingness_rate,

                reliability_mode=
                    config[
                        "reliability_mode"
                    ],

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


        probabilities[
            test_indices
        ] = predict_probability(
            adapted_model,
            X_test,
            device,
        )


        fold_stats.append(
            stats
        )


    return (
        probabilities,
        fold_stats,
    )


# ============================================================
# Main
# ============================================================

def main():

    device = get_device()


    preprocessor = joblib.load(
        SOURCE_MODEL_DIR
        / "cleveland_core8_preprocessor.joblib"
    )


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


    result_rows = []
    diagnostic_rows = []


    print(
        "\n"
        + "=" * 130
    )

    print(
        "PSEUDO-FREE CONSERVATIVE SFDA "
        "COMPONENT ABLATION"
    )

    print(
        "=" * 130
    )


    for seed in SEEDS:

        print(
            f"\nSOURCE SEED {seed}"
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

            target_df = (
                load_dataset(
                    filename
                )
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


            y_target = (
                target_df[
                    "target"
                ]
                .to_numpy()
            )


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


            adapt_seed_base = (
                seed * 10000
                +
                TARGET_SEED_OFFSET[
                    target
                ]
            )


            for (
                variant,
                config,
            ) in VARIANTS.items():

                (
                    probability,
                    fold_stats,
                ) = run_crossfit(
                    source_model=
                        source_model,

                    X_target=
                        X_target,

                    X_target_raw=
                        X_target_raw,

                    config=
                        config,

                    device=
                        device,

                    adapt_seed_base=
                        adapt_seed_base,
                )


                metrics = (
                    compute_binary_metrics(
                        y_target,
                        probability,
                    )
                )


                delta_auc = (
                    metrics[
                        "roc_auc"
                    ]
                    -
                    source_metrics[
                        "roc_auc"
                    ]
                )


                brier_improvement = (
                    source_metrics[
                        "brier"
                    ]
                    -
                    metrics[
                        "brier"
                    ]
                )


                delta_balanced_accuracy = (
                    metrics[
                        "balanced_accuracy"
                    ]
                    -
                    source_metrics[
                        "balanced_accuracy"
                    ]
                )


                result_rows.append(
                    {
                        "seed":
                            seed,

                        "target":
                            target,

                        "variant":
                            variant,

                        "source_roc_auc":
                            source_metrics[
                                "roc_auc"
                            ],

                        "roc_auc":
                            metrics[
                                "roc_auc"
                            ],

                        "delta_roc_auc":
                            delta_auc,

                        "source_brier":
                            source_metrics[
                                "brier"
                            ],

                        "brier":
                            metrics[
                                "brier"
                            ],

                        "brier_improvement":
                            brier_improvement,

                        "delta_balanced_accuracy":
                            delta_balanced_accuracy,

                        "negative_adaptation_auc":
                            (
                                delta_auc
                                < -1e-8
                            ),

                        "brier_worsened":
                            (
                                brier_improvement
                                < -1e-8
                            ),
                    }
                )


                for (
                    fold_number,
                    stats,
                ) in enumerate(
                    fold_stats,
                    start=1,
                ):

                    diagnostic_rows.append(
                        {
                            "seed":
                                seed,

                            "target":
                                target,

                            "variant":
                                variant,

                            "fold":
                                fold_number,

                            **stats,
                        }
                    )


                print(
                    f"Seed={seed:2d}"
                    f" Target={target:14s}"
                    f" Variant={variant:25s}"
                    f" ΔAUC={delta_auc:+.6f}"
                    f" BrierImp={brier_improvement:+.6f}"
                )


    # ========================================================
    # Save raw
    # ========================================================

    result_df = pd.DataFrame(
        result_rows
    )

    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )


    raw_path = (
        RESULT_DIR
        / "conservative_candidate_ablation.csv"
    )


    diagnostic_path = (
        RESULT_DIR
        / "conservative_candidate_diagnostics.csv"
    )


    result_df.to_csv(
        raw_path,
        index=False,
    )


    diagnostic_df.to_csv(
        diagnostic_path,
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
        100
        * summary[
            "negative_adaptation_rate"
        ]
    )


    summary[
        "brier_worsening_rate_pct"
    ] = (
        100
        * summary[
            "brier_worsening_rate"
        ]
    )


    summary_path = (
        RESULT_DIR
        / "conservative_candidate_ablation_summary.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    print(
        "\n"
        + "=" * 145
    )

    print(
        "CONSERVATIVE CANDIDATE ABLATION SUMMARY"
    )

    print(
        "=" * 145
    )


    print(
        summary[
            [
                "target",
                "variant",
                "mean_delta_auc",
                "sd_delta_auc",
                "mean_abs_delta_auc",
                "mean_brier_improvement",
                "mean_delta_balanced_accuracy",
                "negative_adaptation_rate_pct",
                "brier_worsening_rate_pct",
            ]
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\nSaved raw:",
        raw_path
    )

    print(
        "Saved diagnostics:",
        diagnostic_path
    )

    print(
        "Saved summary:",
        summary_path
    )


if __name__ == "__main__":
    main()