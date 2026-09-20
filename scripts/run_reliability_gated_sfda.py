from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold


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


# ==================================================
# Configuration
# ==================================================

SEED = 42

N_FOLDS = 5

SELECTION_FRACTION = 0.40

MINIMUM_PER_CLASS = 8

ADAPT_LR = 5e-5

ADAPT_WEIGHT_DECAY = 1e-4

ADAPT_EPOCHS = 30

LAMBDA_PSEUDO = 1.0

LAMBDA_ENTROPY = 0.05

LAMBDA_ANCHOR = 1.0

LAMBDA_PRIOR = 0.50


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
    / "sfda"
)

ADAPTED_MODEL_DIR = (
    ROOT
    / "models"
    / "adapted"
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


def load_dataset(filename):

    return pd.read_csv(
        DATA_DIR / filename
    )


def load_source_model(
    device,
):

    checkpoint = torch.load(
        SOURCE_MODEL_DIR
        / "cleveland_core8_mlp.pt",
        map_location=device,
        weights_only=False,
    )


    model = HeartMLP(
        input_dim=
            checkpoint["input_dim"],

        hidden_dims=tuple(
            checkpoint[
                "hidden_dims"
            ]
        ),

        dropout=
            checkpoint["dropout"],
    ).to(device)


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


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


def main():

    set_seed(SEED)

    device = get_device()


    preprocessor = joblib.load(
        SOURCE_MODEL_DIR
        / "cleveland_core8_preprocessor.joblib"
    )


    result_rows = []
    fold_rows = []


    print(
        "\n"
        + "=" * 100
    )

    print(
        "RELIABILITY-GATED "
        "CLASS-BALANCED SFDA"
    )

    print(
        "=" * 100
    )

    print(
        "\nProtocol: "
        "5-fold target cross-fitting"
    )

    print(
        "Target labels are NOT used "
        "during adaptation."
    )


    for (
        target_name,
        filename,
    ) in TARGETS.items():

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"TARGET: "
            f"{target_name.upper()}"
        )

        print(
            "=" * 100
        )


        target_df = load_dataset(
            filename
        )


        X_target_df = (
            target_df[
                CORE8
            ]
        )


        y_target = (
            target_df[
                "target"
            ]
            .to_numpy()
        )


        X_target = (
            preprocessor
            .transform(
                X_target_df
            )
            .astype(
                np.float32
            )
        )


        # --------------------------------------------------
        # Source-only reference
        # --------------------------------------------------

        source_model = (
            load_source_model(
                device
            )
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


        # --------------------------------------------------
        # Cross-fitting
        # --------------------------------------------------

        kfold = KFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=SEED,
        )


        adapted_probability = (
            np.zeros(
                len(target_df),
                dtype=float,
            )
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

            print(
                f"\nFold {fold}/{N_FOLDS}"
            )


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


            # ----------------------------------------------
            # Raw target missingness.
            #
            # No labels used.
            # ----------------------------------------------

            X_adapt_raw = (
                X_target_df
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


            fold_source_model = (
                load_source_model(
                    device
                )
            )


            adapted_model, stats = (
                adapt_reliability_gated(
                    source_model=
                        fold_source_model,

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
                        LAMBDA_PSEUDO,

                    lambda_entropy=
                        LAMBDA_ENTROPY,

                    lambda_anchor=
                        LAMBDA_ANCHOR,

                    lambda_prior=
                        LAMBDA_PRIOR,
                )
            )


            probability = (
                predict_probability(
                    adapted_model,
                    X_test,
                    device,
                )
            )


            adapted_probability[
                test_indices
            ] = probability


            # ----------------------------------------------
            # Save fold model
            # ----------------------------------------------

            target_model_dir = (
                ADAPTED_MODEL_DIR
                / target_name
            )


            target_model_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            torch.save(
                {
                    "model_state_dict":
                        adapted_model
                        .state_dict(),

                    "target_domain":
                        target_name,

                    "fold":
                        fold,

                    "method":
                        "reliability_gated",

                    "selection_fraction":
                        SELECTION_FRACTION,

                    "epochs":
                        ADAPT_EPOCHS,

                    "learning_rate":
                        ADAPT_LR,
                },
                target_model_dir
                / (
                    "reliability_gated_"
                    f"fold_{fold}.pt"
                ),
            )


            fold_rows.append(
                {
                    "target":
                        target_name,

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

                    **stats,
                }
            )


            print(
                "Available predicted +:",
                stats[
                    "available_positive"
                ]
            )

            print(
                "Available predicted -:",
                stats[
                    "available_negative"
                ]
            )

            print(
                "Selected per class:",
                stats[
                    "selected_per_class"
                ]
            )

            print(
                "Coverage:",
                f"{100 * stats['coverage']:.2f}%"
            )

            print(
                "Mean selected confidence:",
                f"{stats['mean_selected_confidence']:.4f}"
            )

            print(
                "Missingness:",
                f"{100 * stats['missingness_rate']:.2f}%"
            )

            print(
                "Reliability score:",
                f"{stats['reliability_score']:.4f}"
            )

            print(
                "Positive rate:",
                f"{stats['positive_rate_before']:.4f}",
                "->",
                f"{stats['positive_rate_after']:.4f}",
            )

            print(
                "Entropy:",
                f"{stats['entropy_before']:.4f}",
                "->",
                f"{stats['entropy_after']:.4f}",
            )

            print(
                "Teacher drift:",
                f"{stats['mean_abs_teacher_drift']:.4f}"
            )

            print(
                "Adaptation skipped:",
                stats[
                    "adaptation_skipped"
                ]
            )


        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

        adapted_metrics = (
            compute_binary_metrics(
                y_target,
                adapted_probability,
            )
        )


        result_rows.append(
            {
                "target":
                    target_name,

                "method":
                    "source_only",

                **source_metrics,
            }
        )


        result_rows.append(
            {
                "target":
                    target_name,

                "method":
                    "reliability_gated_sfda",

                **adapted_metrics,
            }
        )


        print(
            "\n"
            + "-" * 100
        )

        print(
            f"FINAL RESULTS: "
            f"{target_name.upper()}"
        )

        print(
            "-" * 100
        )


        print(
            "Source AUROC:",
            f"{source_metrics['roc_auc']:.4f}"
        )

        print(
            "Adapted AUROC:",
            f"{adapted_metrics['roc_auc']:.4f}"
        )

        print(
            "Delta AUROC:",
            f"{adapted_metrics['roc_auc'] - source_metrics['roc_auc']:+.4f}"
        )


        print(
            "\nSource Brier:",
            f"{source_metrics['brier']:.4f}"
        )

        print(
            "Adapted Brier:",
            f"{adapted_metrics['brier']:.4f}"
        )

        print(
            "Delta Brier:",
            f"{adapted_metrics['brier'] - source_metrics['brier']:+.4f}"
        )


        print(
            "\nSource Balanced Accuracy:",
            f"{source_metrics['balanced_accuracy']:.4f}"
        )

        print(
            "Adapted Balanced Accuracy:",
            f"{adapted_metrics['balanced_accuracy']:.4f}"
        )


    # ==================================================
    # Save
    # ==================================================

    results_df = pd.DataFrame(
        result_rows
    )


    folds_df = pd.DataFrame(
        fold_rows
    )


    result_path = (
        RESULT_DIR
        / "reliability_gated_crossfit_core8.csv"
    )


    fold_path = (
        RESULT_DIR
        / "reliability_gated_fold_stats.csv"
    )


    results_df.to_csv(
        result_path,
        index=False,
    )


    folds_df.to_csv(
        fold_path,
        index=False,
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "RELIABILITY-GATED SFDA SUMMARY"
    )

    print(
        "=" * 100
    )


    columns = [
        "target",
        "method",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "brier",
    ]


    print(
        results_df[
            columns
        ].to_string(
            index=False
        )
    )


    print(
        "\nSaved results:",
        result_path
    )

    print(
        "Saved fold stats:",
        fold_path
    )


if __name__ == "__main__":
    main()