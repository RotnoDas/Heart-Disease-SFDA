from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from heart_sfda.data.preprocessing import CORE8

from heart_sfda.models.baselines import (
    get_baseline_models,
)

from heart_sfda.models.factory import (
    build_model_pipeline,
)

from heart_sfda.evaluation.metrics import (
    compute_binary_metrics,
)


DATA_DIR = ROOT / "data" / "processed"

RESULT_DIR = (
    ROOT
    / "results"
    / "cross_domain"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "source"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET_FILES = {

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


def main():

    # --------------------------------------------------
    # SOURCE DOMAIN
    # --------------------------------------------------

    source_df = load_dataset(
        "cleveland.csv"
    )

    X_source = source_df[CORE8]
    y_source = source_df["target"]


    # --------------------------------------------------
    # Five-fold stratified source validation
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )


    models = get_baseline_models()

    all_results = []


    print("\n" + "=" * 100)
    print(
        "SOURCE-ONLY CROSS-DOMAIN BASELINE"
    )
    print("=" * 100)

    print(
        "\nPrimary feature set: CORE8"
    )

    print(
        "Source domain: Cleveland"
    )


    for model_name, estimator in models.items():

        print("\n" + "-" * 100)
        print(
            f"MODEL: {model_name.upper()}"
        )
        print("-" * 100)


        # --------------------------------------------------
        # SOURCE INTERNAL VALIDATION
        # --------------------------------------------------

        cv_pipeline = build_model_pipeline(
            model_name,
            estimator,
        )

        source_oof_probability = (
            cross_val_predict(
                cv_pipeline,
                X_source,
                y_source,
                cv=cv,
                method="predict_proba",
                n_jobs=None,
            )[:, 1]
        )

        source_metrics = (
            compute_binary_metrics(
                y_source,
                source_oof_probability,
            )
        )

        source_metrics.update(
            {
                "model": model_name,
                "source": "cleveland",
                "evaluation_domain":
                    "cleveland_oof",
                "method": "source_only",
                "feature_set": "CORE8",
            }
        )

        all_results.append(
            source_metrics
        )

        print(
            f"Cleveland OOF ROC-AUC: "
            f"{source_metrics['roc_auc']:.4f}"
        )

        print(
            f"Cleveland OOF PR-AUC: "
            f"{source_metrics['pr_auc']:.4f}"
        )


        # --------------------------------------------------
        # Fit final source model using ALL Cleveland data
        # --------------------------------------------------

        final_pipeline = (
            build_model_pipeline(
                model_name,
                estimator,
            )
        )

        final_pipeline.fit(
            X_source,
            y_source,
        )


        # --------------------------------------------------
        # Save model
        # --------------------------------------------------

        model_dir = (
            MODEL_DIR
            / model_name
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            model_dir
            / "cleveland_core8.joblib"
        )

        joblib.dump(
            final_pipeline,
            model_path,
        )


        # --------------------------------------------------
        # Cross-domain testing
        # --------------------------------------------------

        for target_domain, filename in (
            TARGET_FILES.items()
        ):

            target_df = load_dataset(
                filename
            )

            X_target = target_df[CORE8]
            y_target = target_df["target"]

            target_probability = (
                final_pipeline.predict_proba(
                    X_target
                )[:, 1]
            )

            target_metrics = (
                compute_binary_metrics(
                    y_target,
                    target_probability,
                )
            )

            target_metrics.update(
                {
                    "model": model_name,
                    "source": "cleveland",
                    "evaluation_domain":
                        target_domain,
                    "method":
                        "source_only",
                    "feature_set":
                        "CORE8",
                }
            )

            all_results.append(
                target_metrics
            )


            print(
                f"\nCleveland -> "
                f"{target_domain}"
            )

            print(
                f"ROC-AUC: "
                f"{target_metrics['roc_auc']:.4f}"
            )

            print(
                f"PR-AUC: "
                f"{target_metrics['pr_auc']:.4f}"
            )

            print(
                f"Balanced Accuracy: "
                f"{target_metrics['balanced_accuracy']:.4f}"
            )

            print(
                f"Sensitivity: "
                f"{target_metrics['sensitivity']:.4f}"
            )

            print(
                f"Specificity: "
                f"{target_metrics['specificity']:.4f}"
            )

            print(
                f"Brier: "
                f"{target_metrics['brier']:.4f}"
            )


    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    result_df = pd.DataFrame(
        all_results
    )

    preferred_columns = [
        "model",
        "method",
        "feature_set",
        "source",
        "evaluation_domain",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "mcc",
        "brier",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    result_df = result_df[
        preferred_columns
    ]

    output_path = (
        RESULT_DIR
        / "source_only_core8.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 100)

    print(
        "RESULTS SAVED TO:"
    )

    print(output_path)

    print("\n")

    print(
        result_df[
            [
                "model",
                "evaluation_domain",
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "brier",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()