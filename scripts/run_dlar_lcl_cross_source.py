from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import KFold
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
# Frozen configuration
# ============================================================

SEEDS = list(range(10))

N_TARGET_FOLDS = 5
TARGET_FOLD_SEED = 42


# ------------------------------------------------------------
# Source pretraining — released repository defaults
# ------------------------------------------------------------

SOURCE_EPOCHS = 20

SOURCE_BATCH_SIZE = 64

SOURCE_LR = 1e-3

SOURCE_WEIGHT_DECAY = 5e-4

SOURCE_ALPHA_DISCREPANCY = 0.5


# ------------------------------------------------------------
# DLAR + LCL — frozen repository-derived settings
# ------------------------------------------------------------

TARGET_LR = 1e-4

TARGET_WEIGHT_DECAY = 5e-4

DLAR_EPOCHS = 5

LCL_EPOCHS = 10

BETA = 1.0

GAMMA = 1.0

K_NEIGHBORS = 5


# ============================================================
# Paths
# ============================================================

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


# ============================================================
# Domains
# ============================================================

DOMAINS = {
    "cleveland": "cleveland.csv",
    "hungary": "hungary.csv",
    "switzerland": "switzerland.csv",
    "va_long_beach": "va_long_beach.csv",
}


DOMAIN_OFFSET = {
    "cleveland": 1,
    "hungary": 2,
    "switzerland": 3,
    "va_long_beach": 4,
}


# ============================================================
# Utilities
# ============================================================

def load_domain(domain):

    return pd.read_csv(
        DATA_DIR
        / DOMAINS[
            domain
        ]
    )


def make_source_loader(
    X,
    y,
    seed,
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

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=SOURCE_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )


# ============================================================
# Source pretraining
# ============================================================

def train_source_model(
    X_source,
    y_source,
    seed,
    device,
):

    set_seed(seed)

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
        seed,
    )


    final_loss = np.nan

    final_ce = np.nan

    final_discrepancy = np.nan


    for epoch in range(
        1,
        SOURCE_EPOCHS + 1,
    ):

        model.train()

        epoch_losses = []

        epoch_ce = []

        epoch_discrepancies = []


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


            # ------------------------------------------------
            # Two supervised classifier losses
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Classifier discrepancy
            #
            # Equivalent in spirit to released discrepancy():
            # mean |softmax(C1)-softmax(C2)|
            # ------------------------------------------------

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


            discrepancy_loss = (
                torch.mean(
                    torch.abs(
                        probability_1
                        -
                        probability_2
                    )
                )
            )


            # ------------------------------------------------
            # Repository source objective
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
                    "Non-finite source "
                    f"training loss: seed={seed}"
                )


            total_loss.backward()

            optimizer.step()


            epoch_losses.append(
                float(
                    total_loss.item()
                )
            )

            epoch_ce.append(
                float(
                    supervised_loss.item()
                )
            )

            epoch_discrepancies.append(
                float(
                    discrepancy_loss.item()
                )
            )


        final_loss = float(
            np.mean(
                epoch_losses
            )
        )

        final_ce = float(
            np.mean(
                epoch_ce
            )
        )

        final_discrepancy = float(
            np.mean(
                epoch_discrepancies
            )
        )


    model.eval()


    training_stats = {
        "final_source_loss":
            final_loss,

        "final_source_ce":
            final_ce,

        "final_source_discrepancy":
            final_discrepancy,
    }


    return (
        model,
        training_stats,
    )


# ============================================================
# Cross-fitted target adaptation
# ============================================================

def run_target_crossfit(
    source_model,
    X_target,
    source_name,
    target_name,
    seed,
    device,
):

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


    fold_diagnostics = []


    adaptation_seed_base = (
        seed * 100000
        +
        DOMAIN_OFFSET[
            source_name
        ] * 1000
        +
        DOMAIN_OFFSET[
            target_name
        ] * 100
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

        adaptation_seed = (
            adaptation_seed_base
            + fold
        )


        set_seed(
            adaptation_seed
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

            device=device,

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


        fold_diagnostics.append(
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


        del adapted_model


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    return (
        adapted_probability,
        fold_diagnostics,
    )


# ============================================================
# Main experiment
# ============================================================

def main():

    device = get_device()


    print(
        "\n"
        + "=" * 130
    )

    print(
        "DLAR-LCL TABULAR PORT — "
        "12-DIRECTION MULTI-SEED REPLICATION"
    )

    print(
        "=" * 130
    )


    print(
        "\nFrozen settings:"
    )

    print(
        f"  source epochs = {SOURCE_EPOCHS}"
    )

    print(
        f"  source lr = {SOURCE_LR}"
    )

    print(
        f"  target lr = {TARGET_LR}"
    )

    print(
        f"  DLAR epochs = {DLAR_EPOCHS}"
    )

    print(
        f"  LCL epochs = {LCL_EPOCHS}"
    )

    print(
        f"  k = {K_NEIGHBORS}"
    )


    result_rows = []

    prediction_rows = []

    diagnostic_rows = []

    source_training_rows = []


    # ========================================================
    # Every hospital becomes source
    # ========================================================

    for source_name in DOMAINS:

        print(
            "\n"
            + "#" * 130
        )

        print(
            "SOURCE:",
            source_name.upper()
        )

        print(
            "#" * 130
        )


        source_df = load_domain(
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
                dtype=np.int64
            )
        )


        # ----------------------------------------------------
        # Source-only preprocessing
        # ----------------------------------------------------

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


        # ====================================================
        # Prepare targets using source-fitted preprocessing
        # ====================================================

        target_cache = {}


        for target_name in DOMAINS:

            if (
                target_name
                == source_name
            ):
                continue


            target_df = load_domain(
                target_name
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


            if (
                "row_id"
                in target_df.columns
            ):

                row_ids = (
                    target_df[
                        "row_id"
                    ]
                    .to_numpy()
                )

            else:

                row_ids = np.arange(
                    len(
                        target_df
                    )
                )


            target_cache[
                target_name
            ] = {

                "X":
                    X_target,

                "y":
                    target_df[
                        "target"
                    ]
                    .to_numpy(
                        dtype=np.int64
                    ),

                "row_id":
                    row_ids,
            }


        # ====================================================
        # Ten source initializations
        # ====================================================

        for seed in SEEDS:

            print(
                f"\nSource={source_name}"
                f" | Seed={seed}"
            )


            (
                source_model,
                source_training_stats,
            ) = train_source_model(
                X_source=
                    X_source,

                y_source=
                    y_source,

                seed=
                    seed,

                device=
                    device,
            )


            source_training_rows.append(
                {
                    "source":
                        source_name,

                    "seed":
                        seed,

                    **source_training_stats,
                }
            )


            print(
                "  source training:"
                f" loss="
                f"{source_training_stats['final_source_loss']:.4f}"
                f" CE="
                f"{source_training_stats['final_source_ce']:.4f}"
                f" discrepancy="
                f"{source_training_stats['final_source_discrepancy']:.4f}"
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
                    cached[
                        "X"
                    ]
                )


                y_target = (
                    cached[
                        "y"
                    ]
                )


                row_ids = (
                    cached[
                        "row_id"
                    ]
                )


                # =============================================
                # Architecture-matched source-only reference
                # =============================================

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


                # =============================================
                # Cross-fitted DLAR-LCL
                # =============================================

                (
                    adapted_probability,
                    fold_diagnostics,
                ) = run_target_crossfit(
                    source_model=
                        source_model,

                    X_target=
                        X_target,

                    source_name=
                        source_name,

                    target_name=
                        target_name,

                    seed=
                        seed,

                    device=
                        device,
                )


                adapted_metrics = (
                    compute_binary_metrics(
                        y_target,
                        adapted_probability,
                    )
                )


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


                delta_bal_acc = (
                    adapted_metrics[
                        "balanced_accuracy"
                    ]
                    -
                    source_metrics[
                        "balanced_accuracy"
                    ]
                )


                brier_improvement = (
                    source_metrics[
                        "brier"
                    ]
                    -
                    adapted_metrics[
                        "brier"
                    ]
                )


                # =============================================
                # Method-level rows
                # =============================================

                result_rows.append(
                    {
                        "source":
                            source_name,

                        "target":
                            target_name,

                        "seed":
                            seed,

                        "method":
                            "dual_head_source_only",

                        **source_metrics,
                    }
                )


                result_rows.append(
                    {
                        "source":
                            source_name,

                        "target":
                            target_name,

                        "seed":
                            seed,

                        "method":
                            "dlar_lcl",

                        **adapted_metrics,
                    }
                )


                print(
                    f"  {source_name:14s}"
                    f" -> "
                    f"{target_name:14s}"
                    f" | source AUC="
                    f"{source_metrics['roc_auc']:.4f}"
                    f" adapted AUC="
                    f"{adapted_metrics['roc_auc']:.4f}"
                    f" Δ="
                    f"{delta_auc:+.4f}"
                    f" | BrierImp="
                    f"{brier_improvement:+.4f}"
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

                            "dual_head_source_only":
                                float(
                                    source_probability[i]
                                ),

                            "dlar_lcl":
                                float(
                                    adapted_probability[i]
                                ),
                        }
                    )


                # =============================================
                # Diagnostics
                # =============================================

                for fold_stats in (
                    fold_diagnostics
                ):

                    diagnostic_rows.append(
                        {
                            "source":
                                source_name,

                            "target":
                                target_name,

                            "seed":
                                seed,

                            **fold_stats,
                        }
                    )


            del source_model


            if torch.cuda.is_available():

                torch.cuda.empty_cache()


    # ========================================================
    # Save raw method results
    # ========================================================

    result_df = pd.DataFrame(
        result_rows
    )


    raw_path = (
        RESULT_DIR
        / "dlar_lcl_cross_source_results.csv"
    )


    result_df.to_csv(
        raw_path,
        index=False,
    )


    # ========================================================
    # Source training diagnostics
    # ========================================================

    source_training_df = (
        pd.DataFrame(
            source_training_rows
        )
    )


    source_training_path = (
        RESULT_DIR
        / "dlar_lcl_source_training_stats.csv"
    )


    source_training_df.to_csv(
        source_training_path,
        index=False,
    )


    # ========================================================
    # Patient predictions
    # ========================================================

    prediction_df = (
        pd.DataFrame(
            prediction_rows
        )
    )


    prediction_path = (
        RESULT_DIR
        / "dlar_lcl_patient_predictions.csv"
    )


    prediction_df.to_csv(
        prediction_path,
        index=False,
    )


    # ========================================================
    # Diagnostics
    # ========================================================

    diagnostic_df = (
        pd.DataFrame(
            diagnostic_rows
        )
    )


    diagnostic_path = (
        RESULT_DIR
        / "dlar_lcl_cross_source_diagnostics.csv"
    )


    diagnostic_df.to_csv(
        diagnostic_path,
        index=False,
    )


    # ========================================================
    # Build source/adapted comparison
    # ========================================================

    source_reference = (

        result_df[
            result_df[
                "method"
            ]
            == "dual_head_source_only"
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


    adapted = (

        result_df[
            result_df[
                "method"
            ]
            == "dlar_lcl"
        ]

        .copy()
    )


    comparison = (
        adapted
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
        / "dlar_lcl_cross_source_deltas.csv"
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
            ],

            as_index=False,
        )

        .agg(

            mean_source_auc=(
                "source_roc_auc",
                "mean",
            ),

            sd_source_auc=(
                "source_roc_auc",
                "std",
            ),

            mean_adapted_auc=(
                "roc_auc",
                "mean",
            ),

            sd_adapted_auc=(
                "roc_auc",
                "std",
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
        / "dlar_lcl_direction_summary.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    # ========================================================
    # Diagnostic summary
    # ========================================================

    diagnostic_summary = (

        diagnostic_df

        .groupby(
            [
                "source",
                "target",
            ],

            as_index=False,
        )

        .agg(

            mean_confident_fraction=(
                "mean_dlar_confident_fraction",
                "mean",
            ),

            mean_pseudo_positive_fraction=(
                "mean_pseudo_positive_fraction",
                "mean",
            ),

            mean_classifier_discrepancy=(
                "mean_classifier_discrepancy",
                "mean",
            ),

            mean_common_neighbors=(
                "mean_common_neighbors",
                "mean",
            ),

            mean_probability_drift=(
                "probability_drift",
                "mean",
            ),

            mean_positive_rate_before=(
                "positive_rate_before",
                "mean",
            ),

            mean_positive_rate_after=(
                "positive_rate_after",
                "mean",
            ),

            mean_entropy_before=(
                "entropy_before",
                "mean",
            ),

            mean_entropy_after=(
                "entropy_after",
                "mean",
            ),
        )
    )


    diagnostic_summary[
        "mean_positive_rate_change"
    ] = (
        diagnostic_summary[
            "mean_positive_rate_after"
        ]
        -
        diagnostic_summary[
            "mean_positive_rate_before"
        ]
    )


    diagnostic_summary[
        "mean_entropy_change"
    ] = (
        diagnostic_summary[
            "mean_entropy_after"
        ]
        -
        diagnostic_summary[
            "mean_entropy_before"
        ]
    )


    diagnostic_summary_path = (
        RESULT_DIR
        / "dlar_lcl_diagnostic_summary.csv"
    )


    diagnostic_summary.to_csv(
        diagnostic_summary_path,
        index=False,
    )


    # ========================================================
    # Pair-balanced global summary
    # ========================================================

    global_summary = pd.DataFrame(
        [
            {
                "method":
                    "dlar_lcl",

                "mean_direction_delta_auc":
                    float(
                        summary[
                            "mean_delta_auc"
                        ].mean()
                    ),

                "sd_direction_delta_auc":
                    float(
                        summary[
                            "mean_delta_auc"
                        ].std()
                    ),

                "mean_direction_brier_improvement":
                    float(
                        summary[
                            "mean_brier_improvement"
                        ].mean()
                    ),

                "sd_direction_brier_improvement":
                    float(
                        summary[
                            "mean_brier_improvement"
                        ].std()
                    ),

                "mean_direction_delta_balanced_accuracy":
                    float(
                        summary[
                            "mean_delta_balanced_accuracy"
                        ].mean()
                    ),

                "mean_negative_adaptation_rate_pct":
                    float(
                        summary[
                            "negative_adaptation_rate_pct"
                        ].mean()
                    ),

                "mean_brier_worsening_rate_pct":
                    float(
                        summary[
                            "brier_worsening_rate_pct"
                        ].mean()
                    ),
            }
        ]
    )


    global_path = (
        RESULT_DIR
        / "dlar_lcl_global_summary.csv"
    )


    global_summary.to_csv(
        global_path,
        index=False,
    )


    # ========================================================
    # Print
    # ========================================================

    print(
        "\n"
        + "=" * 160
    )

    print(
        "DLAR-LCL 12-DIRECTION SUMMARY"
    )

    print(
        "=" * 160
    )


    print(
        summary[
            [
                "source",
                "target",
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
        + "=" * 130
    )

    print(
        "DLAR-LCL DIAGNOSTIC SUMMARY"
    )

    print(
        "=" * 130
    )


    print(
        diagnostic_summary[
            [
                "source",
                "target",
                "mean_confident_fraction",
                "mean_pseudo_positive_fraction",
                "mean_classifier_discrepancy",
                "mean_common_neighbors",
                "mean_probability_drift",
                "mean_positive_rate_before",
                "mean_positive_rate_after",
                "mean_positive_rate_change",
            ]
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 130
    )

    print(
        "DLAR-LCL PAIR-BALANCED GLOBAL SUMMARY"
    )

    print(
        "=" * 130
    )


    print(
        global_summary.to_string(
            index=False
        )
    )


    print(
        "\nSaved raw:",
        raw_path
    )

    print(
        "Saved source training:",
        source_training_path
    )

    print(
        "Saved patient predictions:",
        prediction_path
    )

    print(
        "Saved diagnostics:",
        diagnostic_path
    )

    print(
        "Saved deltas:",
        comparison_path
    )

    print(
        "Saved direction summary:",
        summary_path
    )

    print(
        "Saved diagnostic summary:",
        diagnostic_summary_path
    )

    print(
        "Saved global summary:",
        global_path
    )


if __name__ == "__main__":
    main()