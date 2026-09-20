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


from heart_sfda.models.dual_head_mlp import (
    DualHeadHeartMLP,
)


CANONICAL_SCRIPT = (
    ROOT
    / "scripts"
    / "run_dlar_lcl_cross_source.py"
)

MODEL_ROOT = (
    ROOT
    / "models"
    / "source_dual_head"
)

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)

REFERENCE_FILE = (
    ROOT
    / "results"
    / "dlar_lcl"
    / "dlar_lcl_patient_predictions.csv"
)


DOMAIN_FILES = {
    "cleveland": "cleveland.csv",
    "hungary": "hungary.csv",
    "switzerland": "switzerland.csv",
    "va_long_beach": "va_long_beach.csv",
}


ABS_TOLERANCE = 1e-6


# ============================================================
# Load canonical experiment module
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
    "canonical_dlar_lcl_cross_source",
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

    model = DualHeadHeartMLP(
        input_dim=int(
            metadata["input_dim"]
        ),

        hidden_dims=tuple(
            metadata["hidden_dims"]
        ),

        dropout=float(
            metadata["dropout"]
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
        package_dir,
        metadata,
        preprocessor,
        model,
    )


# ============================================================
# Target predictors only
#
# NO target labels.
# ============================================================

def load_unlabeled_target(
    target,
    raw_features,
):

    target_path = (
        DATA_DIR
        / DOMAIN_FILES[target]
    )

    usecols = (
        ["row_id"]
        +
        list(raw_features)
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
        set(df.columns)
    ):

        raise RuntimeError(
            "Target-label leakage detected."
        )

    if not df["row_id"].is_unique:

        raise RuntimeError(
            f"Duplicate row_id in {target}"
        )

    return (
        target_path,
        df,
    )


# ============================================================
# Frozen reference predictions
#
# y_true is deliberately NOT loaded.
# ============================================================

def load_reference_predictions(
    source,
    target,
    seed,
):

    usecols = [
        "source",
        "target",
        "seed",
        "row_id",
        "dual_head_source_only",
        "dlar_lcl",
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
        reference["row_id"]
        .astype(str)
    )

    if len(reference) == 0:

        raise RuntimeError(
            "Reference prediction rows not found."
        )

    if not reference["row_id"].is_unique:

        raise RuntimeError(
            "Duplicate reference row_id."
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

    args = parser.parse_args()

    if args.source == args.target:

        raise ValueError(
            "Source and target must differ."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "TRUE SOURCE-FREE DLAR-LCL PACKAGE TEST"
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


    # ========================================================
    # Saved source artifacts
    # ========================================================

    (
        package_dir,
        metadata,
        preprocessor,
        source_model,
    ) = load_source_package(
        source=args.source,
        seed=args.seed,
        device=device,
    )


    raw_features = (
        metadata[
            "raw_features"
        ]
    )


    # ========================================================
    # Target features only
    # ========================================================

    (
        target_path,
        target_df,
    ) = load_unlabeled_target(
        target=args.target,
        raw_features=raw_features,
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


    if (
        X_target.shape[1]
        != int(
            metadata[
                "input_dim"
            ]
        )
    ):

        raise RuntimeError(
            "Transformed dimension "
            "does not match metadata."
        )


    # ========================================================
    # Architecture-matched source-only prediction
    # ========================================================

    source_probability = (
        canonical
        .predict_dlar_lcl_probability(
            source_model,
            X_target,
            device,
        )
    )


    source_probability = np.asarray(
        source_probability,
        dtype=float,
    )


    # ========================================================
    # Source-free DLAR-LCL adaptation
    #
    # Exact canonical cross-fit wrapper.
    # ========================================================

    (
        adapted_probability,
        fold_diagnostics,
    ) = canonical.run_target_crossfit(
        source_model=
            source_model,

        X_target=
            X_target,

        source_name=
            args.source,

        target_name=
            args.target,

        seed=
            args.seed,

        device=
            device,
    )


    adapted_probability = np.asarray(
        adapted_probability,
        dtype=float,
    )


    # ========================================================
    # Probability sanity
    # ========================================================

    for name, probability in [
        (
            "dual_head_source_only",
            source_probability,
        ),
        (
            "dlar_lcl",
            adapted_probability,
        ),
    ]:

        if not np.all(
            np.isfinite(
                probability
            )
        ):

            raise RuntimeError(
                f"{name}: non-finite probability."
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
                f"{name}: probability outside [0, 1]."
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
        "Saved dual-head model loaded:"
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
        "Canonical DLAR-LCL cross-fit used:"
        " YES"
    )

    print(
        "DLAR stage executed:"
        " YES"
    )

    print(
        "LCL stage executed:"
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
        "PREDICTION SUMMARY"
    )

    print(
        "-" * 110
    )

    print(
        "Target N:",
        len(
            adapted_probability
        )
    )

    print(
        "Transformed dimension:",
        X_target.shape[1]
    )

    print(
        "Source-only mean probability:",
        f"{source_probability.mean():.6f}"
    )

    print(
        "DLAR-LCL mean probability:",
        f"{adapted_probability.mean():.6f}"
    )

    print(
        "Source-only positive rate:",
        f"{(source_probability >= 0.5).mean():.6f}"
    )

    print(
        "DLAR-LCL positive rate:",
        f"{(adapted_probability >= 0.5).mean():.6f}"
    )

    print(
        "Cross-fit folds:",
        len(
            fold_diagnostics
        )
    )


    # ========================================================
    # Reference verification
    # ========================================================

    reference = (
        load_reference_predictions(
            source=args.source,
            target=args.target,
            seed=args.seed,
        )
    )


    generated = pd.DataFrame(
        {
            "row_id":
                row_ids,

            "dual_head_source_only_generated":
                source_probability,

            "dlar_lcl_generated":
                adapted_probability,
        }
    )


    comparison = (
        reference
        .merge(
            generated,
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

        raise RuntimeError(
            "Patient alignment failure."
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


    checks = [
        (
            "dual_head_source_only",
            comparison[
                "dual_head_source_only"
            ].to_numpy(
                dtype=float
            ),
            comparison[
                "dual_head_source_only_generated"
            ].to_numpy(
                dtype=float
            ),
        ),

        (
            "dlar_lcl",
            comparison[
                "dlar_lcl"
            ].to_numpy(
                dtype=float
            ),
            comparison[
                "dlar_lcl_generated"
            ].to_numpy(
                dtype=float
            ),
        ),
    ]


    all_passed = True


    for (
        name,
        reference_probability,
        generated_probability,
    ) in checks:

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


        passed = bool(
            max_difference
            <= ABS_TOLERANCE
        )


        all_passed = (
            all_passed
            and passed
        )


        print(
            f"{name:28s}"
            f" max_diff="
            f"{max_difference:.12f}"
            f" mean_diff="
            f"{mean_difference:.12f}"
            f" "
            f"{'PASS' if passed else 'FAIL'}"
        )


    if not all_passed:

        raise SystemExit(
            "DLAR-LCL reproduction failed."
        )


    print(
        "\n"
        + "=" * 110
    )

    print(
        "SOURCE-FREE DLAR-LCL PACKAGE TEST: PASSED"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()