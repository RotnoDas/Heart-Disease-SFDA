from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from heart_sfda.models.mlp import HeartMLP


MODEL_ROOT = (
    ROOT
    / "models"
    / "source"
)

PREDICTION_FILE = (
    ROOT
    / "results"
    / "cross_source"
    / "cross_source_patient_predictions.csv"
)

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)


DOMAIN_FILES = {
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
# Verification tolerance
# ============================================================

ABS_TOLERANCE = 1e-6


# ============================================================
# Load exported source package
# ============================================================

def load_package(
    source,
    seed,
    device,
):

    package_dir = (
        MODEL_ROOT
        / source
        / f"seed_{seed}"
    )


    metadata_path = (
        package_dir
        / "metadata.json"
    )


    preprocessor_path = (
        package_dir
        / "preprocessor.joblib"
    )


    model_path = (
        package_dir
        / "model.pt"
    )


    if not metadata_path.exists():
        raise FileNotFoundError(
            metadata_path
        )

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            preprocessor_path
        )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )


    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)


    preprocessor = joblib.load(
        preprocessor_path
    )


    model = HeartMLP(
        input_dim=
            int(
                metadata[
                    "input_dim"
                ]
            ),

        hidden_dims=
            tuple(
                metadata[
                    "hidden_dims"
                ]
            ),

        dropout=
            float(
                metadata[
                    "dropout"
                ]
            ),
    ).to(device)


    state_dict = torch.load(
        model_path,
        map_location=device,
    )


    model.load_state_dict(
        state_dict
    )

    model.eval()


    return (
        metadata,
        preprocessor,
        model,
    )


# ============================================================
# Prediction
# ============================================================

def predict_probability(
    model,
    X,
    device,
):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )


    model.eval()


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
# Target loading
#
# No target label is required here.
# ============================================================

def load_target_features(
    target,
    raw_features,
):

    target_path = (
        DATA_DIR
        / DOMAIN_FILES[
            target
        ]
    )


    usecols = (
        ["row_id"]
        +
        list(
            raw_features
        )
    )


    df = pd.read_csv(
        target_path,
        usecols=usecols,
    )


    if not df["row_id"].is_unique:

        raise RuntimeError(
            f"row_id is not unique: {target}"
        )


    return df


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "\n"
        + "=" * 120
    )

    print(
        "VERIFY EXPORTED SOURCE MODELS "
        "AGAINST FINAL CROSS-SOURCE PREDICTIONS"
    )

    print(
        "=" * 120
    )


    print(
        "\nDevice:",
        device
    )

    print(
        "Tolerance:",
        ABS_TOLERANCE
    )


    original = pd.read_csv(
        PREDICTION_FILE
    )


    required_columns = {
        "source",
        "target",
        "seed",
        "row_id",
        "source_only",
    }


    missing = (
        required_columns
        -
        set(
            original.columns
        )
    )


    if missing:

        raise ValueError(
            "Missing columns: "
            f"{sorted(missing)}"
        )


    direction_seed_pairs = (
        original[
            [
                "source",
                "target",
                "seed",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "source",
                "target",
                "seed",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    verification_rows = []

    total_comparisons = 0

    passed_comparisons = 0

    global_max_difference = 0.0

    global_mean_difference = 0.0


    # Cache package predictions so the same source/seed
    # model does not need to be reloaded unnecessarily.
    package_cache = {}

    target_cache = {}


    for row in (
        direction_seed_pairs
        .itertuples(
            index=False
        )
    ):

        source = row.source

        target = row.target

        seed = int(
            row.seed
        )


        total_comparisons += 1


        package_key = (
            source,
            seed,
        )


        # ====================================================
        # Load source model package
        # ====================================================

        if (
            package_key
            not in package_cache
        ):

            (
                metadata,
                preprocessor,
                model,
            ) = load_package(
                source=
                    source,

                seed=
                    seed,

                device=
                    device,
            )


            package_cache[
                package_key
            ] = {
                "metadata":
                    metadata,

                "preprocessor":
                    preprocessor,

                "model":
                    model,
            }


        package = (
            package_cache[
                package_key
            ]
        )


        metadata = (
            package[
                "metadata"
            ]
        )

        preprocessor = (
            package[
                "preprocessor"
            ]
        )

        model = (
            package[
                "model"
            ]
        )


        raw_features = (
            metadata[
                "raw_features"
            ]
        )


        # ====================================================
        # Load target predictors only
        # ====================================================

        target_key = (
            source,
            target,
        )


        # Different source domains have different fitted
        # preprocessors, so transformed target matrices
        # are cached by source-target pair.
        if (
            target_key
            not in target_cache
        ):

            target_df = (
                load_target_features(
                    target=
                        target,

                    raw_features=
                        raw_features,
                )
            )


            X_raw = (
                target_df[
                    raw_features
                ]
                .copy()
            )


            X_target = (
                preprocessor
                .transform(
                    X_raw
                )
                .astype(
                    np.float32
                )
            )


            target_cache[
                target_key
            ] = {
                "row_id":
                    target_df[
                        "row_id"
                    ]
                    .astype(str)
                    .to_numpy(),

                "X":
                    X_target,
            }


        cached_target = (
            target_cache[
                target_key
            ]
        )


        # ====================================================
        # Exported model prediction
        # ====================================================

        exported_probability = (
            predict_probability(
                model=
                    model,

                X=
                    cached_target[
                        "X"
                    ],

                device=
                    device,
            )
        )


        exported_df = pd.DataFrame(
            {
                "row_id":
                    cached_target[
                        "row_id"
                    ],

                "exported_probability":
                    exported_probability,
            }
        )


        # ====================================================
        # Original final experiment prediction
        # ====================================================

        original_subset = (
            original[
                (
                    original[
                        "source"
                    ]
                    == source
                )
                &
                (
                    original[
                        "target"
                    ]
                    == target
                )
                &
                (
                    original[
                        "seed"
                    ]
                    == seed
                )
            ][
                [
                    "row_id",
                    "source_only",
                ]
            ]
            .copy()
        )


        original_subset[
            "row_id"
        ] = (
            original_subset[
                "row_id"
            ]
            .astype(str)
        )


        if not (
            original_subset[
                "row_id"
            ]
            .is_unique
        ):

            raise RuntimeError(
                "Duplicate original row_id: "
                f"{source}->{target} "
                f"seed={seed}"
            )


        # ====================================================
        # Align by patient ID
        # ====================================================

        comparison = (
            original_subset
            .merge(
                exported_df,
                on="row_id",
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
        )


        if not (
            comparison[
                "_merge"
            ]
            == "both"
        ).all():

            bad = (
                comparison[
                    comparison[
                        "_merge"
                    ]
                    != "both"
                ]
            )

            raise RuntimeError(
                "Patient alignment failure:\n"
                f"{source}->{target} "
                f"seed={seed}\n"
                f"{bad.head()}"
            )


        difference = np.abs(
            comparison[
                "source_only"
            ]
            .to_numpy(
                dtype=float
            )
            -
            comparison[
                "exported_probability"
            ]
            .to_numpy(
                dtype=float
            )
        )


        max_difference = float(
            difference.max()
        )


        mean_difference = float(
            difference.mean()
        )


        passed = bool(
            max_difference
            <= ABS_TOLERANCE
        )


        if passed:
            passed_comparisons += 1


        global_max_difference = max(
            global_max_difference,
            max_difference,
        )


        global_mean_difference += (
            mean_difference
        )


        verification_rows.append(
            {
                "source":
                    source,

                "target":
                    target,

                "seed":
                    seed,

                "n_patients":
                    len(
                        comparison
                    ),

                "max_absolute_difference":
                    max_difference,

                "mean_absolute_difference":
                    mean_difference,

                "passed":
                    passed,
            }
        )


        status = (
            "PASS"
            if passed
            else "FAIL"
        )


        print(
            f"{source:15s}"
            f" -> "
            f"{target:15s}"
            f" seed={seed}"
            f"  max_diff="
            f"{max_difference:.10f}"
            f"  {status}"
        )


    # ========================================================
    # Summary
    # ========================================================

    global_mean_difference /= (
        total_comparisons
    )


    verification_df = (
        pd.DataFrame(
            verification_rows
        )
    )


    output_path = (
        ROOT
        / "results"
        / "final"
        / "source_model_export_verification.csv"
    )


    verification_df.to_csv(
        output_path,
        index=False,
    )


    print(
        "\n"
        + "=" * 120
    )

    print(
        "VERIFICATION SUMMARY"
    )

    print(
        "=" * 120
    )


    print(
        "Comparisons checked:",
        total_comparisons
    )

    print(
        "Comparisons passed:",
        passed_comparisons
    )

    print(
        "Comparisons failed:",
        (
            total_comparisons
            -
            passed_comparisons
        )
    )

    print(
        "Worst max absolute difference:",
        f"{global_max_difference:.12f}"
    )

    print(
        "Mean comparison-level "
        "absolute difference:",
        f"{global_mean_difference:.12f}"
    )


    print(
        "\nSaved:",
        output_path
    )


    if (
        passed_comparisons
        != total_comparisons
    ):

        failed = (
            verification_df[
                ~verification_df[
                    "passed"
                ]
            ]
        )

        print(
            "\nFAILED COMPARISONS:"
        )

        print(
            failed.to_string(
                index=False
            )
        )

        raise SystemExit(
            1
        )


    print(
        "\n"
        + "=" * 120
    )

    print(
        "EXPORTED SOURCE MODELS "
        "REPRODUCE FINAL SOURCE-ONLY PREDICTIONS"
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":
    main()