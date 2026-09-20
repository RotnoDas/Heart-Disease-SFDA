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


from heart_sfda.data.preprocessing import (
    CORE8,
)

from heart_sfda.models.mlp import (
    HeartMLP,
)

from heart_sfda.adaptation.pseudo_label import (
    adapt_with_pseudo_labels,
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

POSITIVE_THRESHOLD = 0.90

NEGATIVE_THRESHOLD = 0.10

ADAPT_LR = 1e-4

ADAPT_WEIGHT_DECAY = 1e-4

ADAPT_EPOCHS = 20

BATCH_SIZE = 16

MIN_SELECTED = 10


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


# ==================================================
# Utilities
# ==================================================

def load_dataset(filename):

    return pd.read_csv(
        DATA_DIR / filename
    )


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

        probabilities = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

    return probabilities


def load_source_model(
    device,
):

    checkpoint_path = (
        SOURCE_MODEL_DIR
        / "cleveland_core8_mlp.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
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


# ==================================================
# Main
# ==================================================

def main():

    set_seed(SEED)

    device = get_device()


    # --------------------------------------------------
    # Source-trained preprocessor
    # --------------------------------------------------

    preprocessor_path = (
        SOURCE_MODEL_DIR
        / "cleveland_core8_preprocessor.joblib"
    )


    source_preprocessor = (
        joblib.load(
            preprocessor_path
        )
    )


    all_result_rows = []

    all_fold_rows = []


    print("\n" + "=" * 100)

    print(
        "CONFIDENCE-BASED PSEUDO-LABEL SFDA"
    )

    print("=" * 100)

    print(
        "\nProtocol: "
        "5-fold target cross-fitting"
    )

    print(
        "Target labels are NOT used "
        "during adaptation."
    )

    print(
        f"\nConfidence thresholds: "
        f"p <= {NEGATIVE_THRESHOLD} "
        f"or p >= {POSITIVE_THRESHOLD}"
    )


    # ==================================================
    # Target domains
    # ==================================================

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


        # IMPORTANT:
        # labels are stored only for final evaluation
        y_target = (
            target_df[
                "target"
            ]
            .to_numpy()
        )


        # --------------------------------------------------
        # Transform using SOURCE-fitted preprocessor
        # --------------------------------------------------

        X_target = (
            source_preprocessor
            .transform(
                X_target_df
            )
            .astype(
                np.float32
            )
        )


        # --------------------------------------------------
        # Source-only probabilities
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
        # Target cross-fitting
        #
        # IMPORTANT:
        # KFold, NOT StratifiedKFold.
        #
        # Target labels are not used to decide folds.
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


            # ----------------------------------------------
            # Adaptation subset:
            # FEATURES ONLY
            # ----------------------------------------------

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


            # Fresh source model every fold
            fold_source_model = (
                load_source_model(
                    device
                )
            )


            adapted_model, stats = (
                adapt_with_pseudo_labels(
                    source_model=
                        fold_source_model,

                    X_target_unlabeled=
                        X_adapt,

                    device=device,

                    positive_threshold=
                        POSITIVE_THRESHOLD,

                    negative_threshold=
                        NEGATIVE_THRESHOLD,

                    learning_rate=
                        ADAPT_LR,

                    weight_decay=
                        ADAPT_WEIGHT_DECAY,

                    epochs=
                        ADAPT_EPOCHS,

                    batch_size=
                        BATCH_SIZE,

                    min_selected=
                        MIN_SELECTED,
                )
            )


            # ----------------------------------------------
            # Predict ONLY held-out patients
            # ----------------------------------------------

            fold_probability = (
                predict_probability(
                    adapted_model,
                    X_test,
                    device,
                )
            )


            adapted_probability[
                test_indices
            ] = fold_probability


            # ----------------------------------------------
            # Save adapted checkpoint
            # ----------------------------------------------

            target_model_dir = (
                ADAPTED_MODEL_DIR
                / target_name
            )

            target_model_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            model_path = (
                target_model_dir
                / (
                    "pseudo_label_"
                    f"fold_{fold}.pt"
                )
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
                        "pseudo_label",

                    "positive_threshold":
                        POSITIVE_THRESHOLD,

                    "negative_threshold":
                        NEGATIVE_THRESHOLD,

                    "adaptation_epochs":
                        ADAPT_EPOCHS,

                    "adaptation_lr":
                        ADAPT_LR,
                },
                model_path,
            )


            fold_row = {
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


            all_fold_rows.append(
                fold_row
            )


            print(
                "Adapt patients:",
                len(adapt_indices)
            )

            print(
                "Selected:",
                stats[
                    "selected_n"
                ]
            )

            print(
                "Coverage:",
                f"{100 * stats['coverage']:.2f}%"
            )

            print(
                "Pseudo positive:",
                stats[
                    "pseudo_positive_n"
                ]
            )

            print(
                "Pseudo negative:",
                stats[
                    "pseudo_negative_n"
                ]
            )

            print(
                "Mean confidence:",
                stats[
                    "mean_selected_confidence"
                ]
            )

            print(
                "Adaptation skipped:",
                stats[
                    "adaptation_skipped"
                ]
            )


        # ==================================================
        # Final target evaluation
        # ==================================================

        adapted_metrics = (
            compute_binary_metrics(
                y_target,
                adapted_probability,
            )
        )


        # --------------------------------------------------
        # Save source-only row
        # --------------------------------------------------

        source_row = {
            "target":
                target_name,

            "method":
                "source_only",

            **source_metrics,
        }


        all_result_rows.append(
            source_row
        )


        # --------------------------------------------------
        # Save SFDA row
        # --------------------------------------------------

        adapted_row = {
            "target":
                target_name,

            "method":
                "pseudo_label_sfda",

            **adapted_metrics,
        }


        all_result_rows.append(
            adapted_row
        )


        # --------------------------------------------------
        # Print comparison
        # --------------------------------------------------

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
            f"Source-only AUROC: "
            f"{source_metrics['roc_auc']:.4f}"
        )

        print(
            f"Pseudo-label AUROC: "
            f"{adapted_metrics['roc_auc']:.4f}"
        )

        print(
            f"Delta AUROC: "
            f"{adapted_metrics['roc_auc'] - source_metrics['roc_auc']:+.4f}"
        )


        print(
            f"\nSource-only Brier: "
            f"{source_metrics['brier']:.4f}"
        )

        print(
            f"Pseudo-label Brier: "
            f"{adapted_metrics['brier']:.4f}"
        )

        print(
            f"Delta Brier: "
            f"{adapted_metrics['brier'] - source_metrics['brier']:+.4f}"
        )


        print(
            f"\nSource Balanced Acc: "
            f"{source_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Adapted Balanced Acc: "
            f"{adapted_metrics['balanced_accuracy']:.4f}"
        )


    # ==================================================
    # Save results
    # ==================================================

    results_df = pd.DataFrame(
        all_result_rows
    )


    fold_df = pd.DataFrame(
        all_fold_rows
    )


    result_path = (
        RESULT_DIR
        / "pseudo_label_crossfit_core8.csv"
    )


    fold_path = (
        RESULT_DIR
        / "pseudo_label_fold_stats.csv"
    )


    results_df.to_csv(
        result_path,
        index=False,
    )


    fold_df.to_csv(
        fold_path,
        index=False,
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "PSEUDO-LABEL SFDA SUMMARY"
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