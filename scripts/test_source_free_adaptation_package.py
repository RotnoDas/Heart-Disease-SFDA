from pathlib import Path
import argparse
import importlib.util
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


CANONICAL_SCRIPT = (
    ROOT
    / "scripts"
    / "run_cross_source_replication.py"
)

MODEL_ROOT = (
    ROOT
    / "models"
    / "source"
)

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)

REFERENCE_FILE = (
    ROOT
    / "results"
    / "cross_source"
    / "cross_source_patient_predictions.csv"
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


ABS_TOLERANCE = 1e-6


# ============================================================
# Load canonical final experiment module
# ============================================================

def load_script_module(
    module_name,
    path,
):

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"Cannot import {path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


canonical = load_script_module(
    "canonical_cross_source",
    CANONICAL_SCRIPT,
)


# ============================================================
# Load saved source package
# ============================================================

def load_source_package(
    source,
    seed,
    device,
):

    package_dir = (
        MODEL_ROOT
        / source
        / f"seed_{seed}"
    )


    with open(
        package_dir / "metadata.json",
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)


    preprocessor = joblib.load(
        package_dir
        / "preprocessor.joblib"
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
        package_dir / "model.pt",
        map_location=device,
    )


    model.load_state_dict(
        state_dict
    )

    model.eval()


    return (
        package_dir,
        metadata,
        preprocessor,
        model,
    )


# ============================================================
# Load target predictors only
#
# NO target labels.
# ============================================================

def load_unlabeled_target(
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


    forbidden = {
        "target",
        "num",
        "y_true",
    }


    if (
        forbidden
        &
        set(
            df.columns
        )
    ):
        raise RuntimeError(
            "Target label leakage detected."
        )


    return (
        target_path,
        df,
    )


# ============================================================
# Verification reference
#
# IMPORTANT:
# Only prediction columns are read.
# y_true is deliberately NOT loaded.
# ============================================================

def load_reference_predictions(
    source,
    target,
    seed,
    methods,
):

    usecols = [
        "source",
        "target",
        "seed",
        "row_id",
        *methods,
    ]


    reference = pd.read_csv(
        REFERENCE_FILE,
        usecols=usecols,
    )


    reference = (
        reference[
            (
                reference["source"]
                == source
            )
            &
            (
                reference["target"]
                == target
            )
            &
            (
                reference["seed"]
                == seed
            )
        ]
        .copy()
    )


    reference[
        "row_id"
    ] = (
        reference[
            "row_id"
        ]
        .astype(str)
    )


    return reference


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--source",
        default="cleveland",
        choices=DOMAIN_FILES.keys(),
    )


    parser.add_argument(
        "--target",
        default="hungary",
        choices=DOMAIN_FILES.keys(),
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )


    parser.add_argument(
        "--method",
        default="all",
        choices=[
            "all",
            *canonical.METHODS,
        ],
    )


    args = parser.parse_args()


    if (
        args.source
        == args.target
    ):
        raise ValueError(
            "Source and target must differ."
        )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    methods = (
        list(
            canonical.METHODS
        )
        if args.method == "all"
        else [
            args.method
        ]
    )


    print(
        "\n"
        + "=" * 110
    )

    print(
        "TRUE SOURCE-FREE SFDA ADAPTATION TEST"
    )

    print(
        "=" * 110
    )


    print(
        "\nDevice:",
        device
    )

    print(
        "Source:",
        args.source
    )

    print(
        "Target:",
        args.target
    )

    print(
        "Seed:",
        args.seed
    )

    print(
        "Methods:",
        ", ".join(methods)
    )


    (
        package_dir,
        metadata,
        preprocessor,
        source_model,
    ) = load_source_package(
        source=
            args.source,

        seed=
            args.seed,

        device=
            device,
    )


    raw_features = (
        metadata[
            "raw_features"
        ]
    )


    (
        target_path,
        target_df,
    ) = load_unlabeled_target(
        target=
            args.target,

        raw_features=
            raw_features,
    )


    row_ids = (
        target_df[
            "row_id"
        ]
        .astype(str)
        .to_numpy()
    )


    X_target_raw = (
        target_df[
            raw_features
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


    # ========================================================
    # Adaptation
    #
    # Canonical final experiment function is reused exactly.
    # ========================================================

    generated = {}


    for method in methods:

        print(
            "\nRunning:",
            method
        )


        (
            probability,
            fold_diagnostics,
        ) = canonical.run_crossfit_method(
            method=
                method,

            source_model=
                source_model,

            X_target=
                X_target,

            X_target_raw=
                X_target_raw,

            source_name=
                args.source,

            target_name=
                args.target,

            source_seed=
                args.seed,

            device=
                device,
        )


        probability = np.asarray(
            probability,
            dtype=float,
        )


        if not np.all(
            np.isfinite(
                probability
            )
        ):
            raise RuntimeError(
                f"{method}: non-finite probability."
            )


        if (
            np.any(
                probability < 0
            )
            or
            np.any(
                probability > 1
            )
        ):
            raise RuntimeError(
                f"{method}: probability outside [0, 1]."
            )


        generated[
            method
        ] = probability


        print(
            "  N:",
            len(probability)
        )

        print(
            "  mean probability:",
            f"{probability.mean():.6f}"
        )

        print(
            "  predicted positive rate:",
            f"{(probability >= 0.5).mean():.6f}"
        )

        print(
            "  folds:",
            len(
                fold_diagnostics
            )
        )


    # ========================================================
    # Reference verification
    #
    # This occurs AFTER adaptation.
    # No target labels are read.
    # ========================================================

    reference = (
        load_reference_predictions(
            source=
                args.source,

            target=
                args.target,

            seed=
                args.seed,

            methods=
                methods,
        )
    )


    generated_df = pd.DataFrame(
        {
            "row_id":
                row_ids,
        }
    )


    for method in methods:

        generated_df[
            method
        ] = (
            generated[
                method
            ]
        )


    comparison = (
        reference
        .merge(
            generated_df,
            on="row_id",
            how="outer",
            suffixes=(
                "_reference",
                "_generated",
            ),
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

        raise RuntimeError(
            "Patient alignment failure."
        )


    print(
        "\n"
        + "-" * 110
    )

    print(
        "SOURCE-FREE INPUT AUDIT"
    )

    print(
        "-" * 110
    )


    print(
        "Source training dataset loaded:"
        " NO"
    )

    print(
        "Target labels loaded:"
        " NO"
    )

    print(
        "Saved source model loaded:"
        " YES"
    )

    print(
        "Saved source preprocessor loaded:"
        " YES"
    )

    print(
        "Unlabeled target predictors loaded:"
        " YES"
    )

    print(
        "Canonical SFDA function used:"
        " YES"
    )


    print(
        "\nPackage:",
        package_dir
    )

    print(
        "Target predictor file:",
        target_path
    )


    print(
        "\n"
        + "-" * 110
    )

    print(
        "FINAL-PREDICTION REPRODUCTION"
    )

    print(
        "-" * 110
    )


    all_passed = True


    for method in methods:

        reference_probability = (
            comparison[
                f"{method}_reference"
            ]
            .to_numpy(
                dtype=float
            )
        )


        generated_probability = (
            comparison[
                f"{method}_generated"
            ]
            .to_numpy(
                dtype=float
            )
        )


        difference = np.abs(
            reference_probability
            -
            generated_probability
        )


        max_difference = float(
            difference.max()
        )


        mean_difference = float(
            difference.mean()
        )


        passed = (
            max_difference
            <= ABS_TOLERANCE
        )


        all_passed = (
            all_passed
            and passed
        )


        print(
            f"{method:28s}"
            f" max_diff="
            f"{max_difference:.12f}"
            f" mean_diff="
            f"{mean_difference:.12f}"
            f" "
            f"{'PASS' if passed else 'FAIL'}"
        )


    if not all_passed:

        raise SystemExit(
            "Adaptation reproduction failed."
        )


    print(
        "\n"
        + "=" * 110
    )

    print(
        "SOURCE-FREE SFDA ADAPTATION PACKAGE TEST: PASSED"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()