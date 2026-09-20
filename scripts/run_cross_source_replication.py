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

from heart_sfda.adaptation.pseudo_label import (
    adapt_with_pseudo_labels,
)

from heart_sfda.adaptation.entropy import (
    adapt_with_entropy,
)

from heart_sfda.adaptation.reliability_gate import (
    adapt_reliability_gated,
)

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
# Frozen experiment configuration
# ============================================================

SEEDS = list(range(10))

N_FOLDS = 5

FOLD_RANDOM_STATE = 42


# ------------------------------------------------------------
# Source model
# ------------------------------------------------------------

SOURCE_LR = 1e-3
SOURCE_WEIGHT_DECAY = 1e-4
SOURCE_BATCH_SIZE = 32


# ------------------------------------------------------------
# Pseudo-label baseline
# ------------------------------------------------------------

PSEUDO_POSITIVE_THRESHOLD = 0.90
PSEUDO_NEGATIVE_THRESHOLD = 0.10

PSEUDO_LR = 1e-4
PSEUDO_WEIGHT_DECAY = 1e-4
PSEUDO_EPOCHS = 20
PSEUDO_BATCH_SIZE = 16
PSEUDO_MIN_SELECTED = 10


# ------------------------------------------------------------
# Entropy baseline
# ------------------------------------------------------------

ENTROPY_LR = 1e-4
ENTROPY_WEIGHT_DECAY = 1e-4
ENTROPY_EPOCHS = 20
ENTROPY_BATCH_SIZE = 32


# ------------------------------------------------------------
# Reliability-gated v1
# ------------------------------------------------------------

RG_SELECTION_FRACTION = 0.40
RG_MINIMUM_PER_CLASS = 8

RG_LR = 5e-5
RG_WEIGHT_DECAY = 1e-4
RG_EPOCHS = 30

RG_LAMBDA_PSEUDO = 1.0
RG_LAMBDA_ENTROPY = 0.05
RG_LAMBDA_ANCHOR = 1.0
RG_LAMBDA_PRIOR = 0.50


# ------------------------------------------------------------
# Conservative candidate
# ------------------------------------------------------------

CONS_SELECTION_FRACTION = 0.40
CONS_MINIMUM_PER_CLASS = 8

CONS_LR = 5e-5
CONS_WEIGHT_DECAY = 1e-4
CONS_EPOCHS = 30

CONS_LAMBDA_ENTROPY = 0.05
CONS_LAMBDA_ANCHOR = 1.0
CONS_LAMBDA_PRIOR = 0.50


# ============================================================
# Paths
# ============================================================

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)

STATISTICS_DIR = (
    ROOT
    / "results"
    / "statistics"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "cross_source"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EPOCH_PLAN_FILE = (
    STATISTICS_DIR
    / "source_epoch_plan.csv"
)


# ============================================================
# Domains
# ============================================================

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


DOMAIN_OFFSET = {
    "cleveland": 1,
    "hungary": 2,
    "switzerland": 3,
    "va_long_beach": 4,
}


METHODS = [
    "pseudo_label_sfda",
    "entropy_sfda",
    "reliability_gated_sfda",
    "conservative_candidate",
]


# ============================================================
# Basic utilities
# ============================================================

def load_dataset(
    domain,
):

    return pd.read_csv(
        DATA_DIR
        / DOMAINS[
            domain
        ]
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


# ============================================================
# Source training
# ============================================================

def train_source_model(
    X,
    y,
    epochs,
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
        epochs
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

        probabilities = (
            torch.sigmoid(
                model(
                    X_tensor
                )
            )
            .cpu()
            .numpy()
        )


    return probabilities


# ============================================================
# Cross-fitted adaptation
# ============================================================

def run_crossfit_method(
    method,
    source_model,
    X_target,
    X_target_raw,
    source_name,
    target_name,
    source_seed,
    device,
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


    fold_diagnostics = []


    # --------------------------------------------------------
    # IMPORTANT:
    # method is NOT part of this seed.
    #
    # Same source seed + source + target + fold
    # gets same stochastic seed for all methods.
    # --------------------------------------------------------

    adaptation_seed_base = (

        source_seed
        * 100000

        +

        DOMAIN_OFFSET[
            source_name
        ]
        * 1000

        +

        DOMAIN_OFFSET[
            target_name
        ]
        * 100
    )


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


        X_adapt_raw = (
            X_target_raw
            .iloc[
                adapt_indices
            ]
        )


        missingness_rate = float(
            X_adapt_raw
            .isna()
            .mean()
            .mean()
        )


        # ----------------------------------------------------
        # Matched adaptation RNG
        # ----------------------------------------------------

        adaptation_seed = (
            adaptation_seed_base
            + fold
        )


        set_seed(
            adaptation_seed
        )


        # ====================================================
        # Method
        # ====================================================

        if (
            method
            == "pseudo_label_sfda"
        ):

            adapted_model, stats = (
                adapt_with_pseudo_labels(
                    source_model=
                        source_model,

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

            adapted_model, stats = (
                adapt_with_entropy(
                    source_model=
                        source_model,

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

            adapted_model, stats = (
                adapt_reliability_gated(
                    source_model=
                        source_model,

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


        elif (
            method
            == "conservative_candidate"
        ):

            adapted_model, stats = (
                adapt_conservative(
                    source_model=
                        source_model,

                    X_target_unlabeled=
                        X_adapt,

                    device=device,

                    missingness_rate=
                        missingness_rate,

                    reliability_mode=
                        "full",

                    selection_fraction=
                        CONS_SELECTION_FRACTION,

                    minimum_per_class=
                        CONS_MINIMUM_PER_CLASS,

                    learning_rate=
                        CONS_LR,

                    weight_decay=
                        CONS_WEIGHT_DECAY,

                    epochs=
                        CONS_EPOCHS,

                    lambda_entropy=
                        CONS_LAMBDA_ENTROPY,

                    lambda_anchor=
                        CONS_LAMBDA_ANCHOR,

                    lambda_prior=
                        CONS_LAMBDA_PRIOR,
                )
            )


        else:

            raise ValueError(
                f"Unknown method: "
                f"{method}"
            )


        # ----------------------------------------------------
        # Held-out target prediction
        # ----------------------------------------------------

        probabilities[
            test_indices
        ] = predict_probability(
            adapted_model,
            X_test,
            device,
        )


        fold_diagnostics.append(
            {
                "fold":
                    fold,

                "adapt_n":
                    len(
                        adapt_indices
                    ),

                "test_n":
                    len(
                        test_indices
                    ),

                "missingness_rate":
                    missingness_rate,

                **stats,
            }
        )


        del adapted_model


        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    return (
        probabilities,
        fold_diagnostics,
    )


# ============================================================
# Main
# ============================================================

def main():

    device = get_device()


    # --------------------------------------------------------
    # Frozen source-specific epochs
    # --------------------------------------------------------

    epoch_plan_df = pd.read_csv(
        EPOCH_PLAN_FILE
    )


    epoch_plan = dict(
        zip(
            epoch_plan_df[
                "source"
            ],

            epoch_plan_df[
                "median_best_epoch"
            ].astype(int),
        )
    )


    print(
        "\n"
        + "=" * 130
    )

    print(
        "12-DIRECTION CROSS-SOURCE "
        "SFDA REPLICATION"
    )

    print(
        "=" * 130
    )


    print(
        "\nFrozen source epoch plan:"
    )


    for domain in DOMAINS:

        print(
            f"  {domain:15s}: "
            f"{epoch_plan[domain]} epochs"
        )


    result_rows = []

    prediction_rows = []

    diagnostic_rows = []


    # ========================================================
    # Each hospital becomes source
    # ========================================================

    for source_name in DOMAINS:

        print(
            "\n"
            + "#" * 130
        )

        print(
            f"SOURCE DOMAIN: "
            f"{source_name.upper()}"
        )

        print(
            "#" * 130
        )


        source_df = load_dataset(
            source_name
        )


        X_source_raw = (
            source_df[
                CORE8
            ]
            .copy()
        )


        y_source = (
            source_df[
                "target"
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


        # ----------------------------------------------------
        # Source-fitted preprocessing.
        #
        # No target data enters fitting.
        # ----------------------------------------------------

        source_preprocessor = (
            build_linear_preprocessor()
        )


        X_source = (
            source_preprocessor
            .fit_transform(
                X_source_raw
            )
            .astype(
                np.float32
            )
        )


        # ====================================================
        # Prepare every target using this source preprocessor
        # ====================================================

        target_cache = {}


        for target_name in DOMAINS:

            if (
                target_name
                == source_name
            ):
                continue


            target_df = load_dataset(
                target_name
            )


            X_target_raw = (
                target_df[
                    CORE8
                ]
                .copy()
            )


            X_target = (
                source_preprocessor
                .transform(
                    X_target_raw
                )
                .astype(
                    np.float32
                )
            )


            target_cache[
                target_name
            ] = {

                "df":
                    target_df,

                "X_raw":
                    X_target_raw,

                "X":
                    X_target,

                "y":
                    target_df[
                        "target"
                    ]
                    .to_numpy(),

                "row_id":
                    target_df[
                        "row_id"
                    ]
                    .to_numpy(),
            }


        # ====================================================
        # Source seeds
        # ====================================================

        for seed in SEEDS:

            print(
                f"\nSource={source_name} | "
                f"Seed={seed}"
            )


            source_model = (
                train_source_model(
                    X=
                        X_source,

                    y=
                        y_source,

                    epochs=
                        epoch_plan[
                            source_name
                        ],

                    seed=
                        seed,

                    device=
                        device,
                )
            )


            # =================================================
            # Three targets
            # =================================================

            for target_name in DOMAINS:

                if (
                    target_name
                    == source_name
                ):
                    continue


                cached = (
                    target_cache[
                        target_name
                    ]
                )


                X_target = (
                    cached["X"]
                )


                X_target_raw = (
                    cached["X_raw"]
                )


                y_target = (
                    cached["y"]
                )


                row_ids = (
                    cached["row_id"]
                )


                print(
                    f"  "
                    f"{source_name}"
                    f" -> "
                    f"{target_name}"
                )


                # =============================================
                # Source-only
                # =============================================

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


                result_rows.append(
                    {
                        "source":
                            source_name,

                        "target":
                            target_name,

                        "seed":
                            seed,

                        "source_epochs":
                            epoch_plan[
                                source_name
                            ],

                        "method":
                            "source_only",

                        **source_metrics,
                    }
                )


                # ---------------------------------------------
                # Patient prediction dictionary
                # ---------------------------------------------

                patient_probabilities = {
                    "source_only":
                        source_probability
                }


                print(
                    f"    source_only"
                    f" AUROC="
                    f"{source_metrics['roc_auc']:.4f}"
                    f" Brier="
                    f"{source_metrics['brier']:.4f}"
                )


                # =============================================
                # Adaptation methods
                # =============================================

                for method in METHODS:

                    (
                        adapted_probability,
                        fold_stats,
                    ) = run_crossfit_method(
                        method=
                            method,

                        source_model=
                            source_model,

                        X_target=
                            X_target,

                        X_target_raw=
                            X_target_raw,

                        source_name=
                            source_name,

                        target_name=
                            target_name,

                        source_seed=
                            seed,

                        device=
                            device,
                    )


                    metrics = (
                        compute_binary_metrics(
                            y_target,
                            adapted_probability,
                        )
                    )


                    result_rows.append(
                        {
                            "source":
                                source_name,

                            "target":
                                target_name,

                            "seed":
                                seed,

                            "source_epochs":
                                epoch_plan[
                                    source_name
                                ],

                            "method":
                                method,

                            **metrics,
                        }
                    )


                    patient_probabilities[
                        method
                    ] = (
                        adapted_probability
                    )


                    print(
                        f"    "
                        f"{method:25s}"
                        f" AUROC="
                        f"{metrics['roc_auc']:.4f}"
                        f" Brier="
                        f"{metrics['brier']:.4f}"
                    )


                    # -----------------------------------------
                    # Save diagnostics
                    # -----------------------------------------

                    for fold_stats_row in (
                        fold_stats
                    ):

                        diagnostic_rows.append(
                            {
                                "source":
                                    source_name,

                                "target":
                                    target_name,

                                "seed":
                                    seed,

                                "method":
                                    method,

                                **fold_stats_row,
                            }
                        )


                # =============================================
                # Patient-level predictions
                # =============================================

                for i in range(
                    len(
                        y_target
                    )
                ):

                    prediction_rows.append(
                        {
                            "source":
                                source_name,

                            "target":
                                target_name,

                            "seed":
                                seed,

                            "row_id":
                                row_ids[i],

                            "y_true":
                                int(
                                    y_target[i]
                                ),

                            "source_only":
                                float(
                                    patient_probabilities[
                                        "source_only"
                                    ][i]
                                ),

                            "pseudo_label_sfda":
                                float(
                                    patient_probabilities[
                                        "pseudo_label_sfda"
                                    ][i]
                                ),

                            "entropy_sfda":
                                float(
                                    patient_probabilities[
                                        "entropy_sfda"
                                    ][i]
                                ),

                            "reliability_gated_sfda":
                                float(
                                    patient_probabilities[
                                        "reliability_gated_sfda"
                                    ][i]
                                ),

                            "conservative_candidate":
                                float(
                                    patient_probabilities[
                                        "conservative_candidate"
                                    ][i]
                                ),
                        }
                    )


            del source_model


            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    # ========================================================
    # Save raw method-level results
    # ========================================================

    results_df = pd.DataFrame(
        result_rows
    )


    raw_path = (
        RESULT_DIR
        / "cross_source_multiseed_results.csv"
    )


    results_df.to_csv(
        raw_path,
        index=False,
    )


    # ========================================================
    # Save patient-level probabilities
    # ========================================================

    prediction_df = pd.DataFrame(
        prediction_rows
    )


    prediction_path = (
        RESULT_DIR
        / "cross_source_patient_predictions.csv"
    )


    prediction_df.to_csv(
        prediction_path,
        index=False,
    )


    # ========================================================
    # Save fold diagnostics
    # ========================================================

    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )


    diagnostic_path = (
        RESULT_DIR
        / "cross_source_fold_diagnostics.csv"
    )


    diagnostic_df.to_csv(
        diagnostic_path,
        index=False,
    )


    # ========================================================
    # Build paired source references
    # ========================================================

    source_reference = (

        results_df[
            results_df[
                "method"
            ]
            == "source_only"
        ][
            [
                "source",
                "target",
                "seed",
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


    adapted_df = (

        results_df[
            results_df[
                "method"
            ]
            != "source_only"
        ]

        .copy()
    )


    comparison = (
        adapted_df
        .merge(
            source_reference,

            on=[
                "source",
                "target",
                "seed",
            ],

            how="left",
        )
    )


    # ========================================================
    # Paired deltas
    # ========================================================

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


    comparison[
        "brier_worsened"
    ] = (
        comparison[
            "brier_improvement"
        ]
        < -1e-8
    )


    comparison_path = (
        RESULT_DIR
        / "cross_source_multiseed_deltas.csv"
    )


    comparison.to_csv(
        comparison_path,
        index=False,
    )


    # ========================================================
    # Direction-level summary
    # ========================================================

    summary = (

        comparison

        .groupby(
            [
                "source",
                "target",
                "method",
            ],

            as_index=False,
        )

        .agg(

            mean_source_auc=(
                "source_roc_auc",
                "mean",
            ),

            mean_adapted_auc=(
                "roc_auc",
                "mean",
            ),

            mean_delta_auc=(
                "delta_roc_auc",
                "mean",
            ),

            sd_delta_auc=(
                "delta_roc_auc",
                "std",
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


    summary_path = (
        RESULT_DIR
        / "cross_source_direction_summary.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    # ========================================================
    # Pair-balanced global method summary
    #
    # First direction means are computed above.
    # Then 12 directions receive equal weight.
    # ========================================================

    global_summary = (

        summary

        .groupby(
            "method",
            as_index=False,
        )

        .agg(

            mean_direction_delta_auc=(
                "mean_delta_auc",
                "mean",
            ),

            sd_direction_delta_auc=(
                "mean_delta_auc",
                "std",
            ),

            mean_direction_brier_improvement=(
                "mean_brier_improvement",
                "mean",
            ),

            sd_direction_brier_improvement=(
                "mean_brier_improvement",
                "std",
            ),

            mean_direction_delta_balanced_accuracy=(
                "mean_delta_balanced_accuracy",
                "mean",
            ),

            mean_negative_adaptation_rate_pct=(
                "negative_adaptation_rate_pct",
                "mean",
            ),

            mean_brier_worsening_rate_pct=(
                "brier_worsening_rate_pct",
                "mean",
            ),
        )
    )


    global_path = (
        RESULT_DIR
        / "cross_source_global_summary.csv"
    )


    global_summary.to_csv(
        global_path,
        index=False,
    )


    # ========================================================
    # Print direction summary
    # ========================================================

    print(
        "\n"
        + "=" * 170
    )

    print(
        "12-DIRECTION REPLICATION SUMMARY"
    )

    print(
        "=" * 170
    )


    print(
        summary[
            [
                "source",
                "target",
                "method",
                "mean_source_auc",
                "mean_adapted_auc",
                "mean_delta_auc",
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
        "\n"
        + "=" * 150
    )

    print(
        "PAIR-BALANCED GLOBAL SUMMARY"
    )

    print(
        "=" * 150
    )


    print(
        global_summary.to_string(
            index=False
        )
    )


    print(
        "\nSaved raw results:",
        raw_path
    )

    print(
        "Saved patient predictions:",
        prediction_path
    )

    print(
        "Saved fold diagnostics:",
        diagnostic_path
    )

    print(
        "Saved paired deltas:",
        comparison_path
    )

    print(
        "Saved direction summary:",
        summary_path
    )

    print(
        "Saved global summary:",
        global_path
    )


if __name__ == "__main__":
    main()