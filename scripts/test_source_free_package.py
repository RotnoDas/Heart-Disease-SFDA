from pathlib import Path
import argparse
import hashlib
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


DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)

MODEL_ROOT = (
    ROOT
    / "models"
    / "source"
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
# Utilities
# ============================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def verify_hashes(
    package_dir,
):

    hash_file = (
        package_dir
        / "hashes.json"
    )

    if not hash_file.exists():

        raise FileNotFoundError(
            f"Missing hash file: {hash_file}"
        )


    with open(
        hash_file,
        "r",
        encoding="utf-8",
    ) as f:

        expected = json.load(f)


    checks = {
        "model_sha256":
            package_dir
            / "model.pt",

        "preprocessor_sha256":
            package_dir
            / "preprocessor.joblib",

        "metadata_sha256":
            package_dir
            / "metadata.json",
    }


    for key, path in checks.items():

        actual = (
            sha256_file(
                path
            )
        )

        if (
            actual
            != expected[key]
        ):

            raise RuntimeError(
                "Artifact hash mismatch:\n"
                f"  file={path}\n"
                f"  expected={expected[key]}\n"
                f"  actual={actual}"
            )


# ============================================================
# Load source-free package
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


    if not package_dir.exists():

        raise FileNotFoundError(
            f"Package not found: {package_dir}"
        )


    verify_hashes(
        package_dir
    )


    metadata_path = (
        package_dir
        / "metadata.json"
    )


    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)


    if (
        metadata[
            "source_domain"
        ]
        != source
    ):

        raise RuntimeError(
            "Source metadata mismatch."
        )


    if (
        int(
            metadata["seed"]
        )
        != seed
    ):

        raise RuntimeError(
            "Seed metadata mismatch."
        )


    preprocessor = (
        joblib.load(
            package_dir
            / "preprocessor.joblib"
        )
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
        package_dir
        / "model.pt",
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
# Load target FEATURES ONLY
#
# IMPORTANT:
# - target labels are not loaded
# - source dataset is not loaded
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


    # --------------------------------------------------------
    # Read only predictor columns.
    #
    # "target" and "num" are deliberately not requested.
    # --------------------------------------------------------

    target_df = pd.read_csv(
        target_path,
        usecols=raw_features,
    )


    forbidden = {
        "target",
        "num",
    }


    if (
        forbidden
        &
        set(
            target_df.columns
        )
    ):

        raise RuntimeError(
            "Target label leaked into "
            "source-free input."
        )


    return (
        target_path,
        target_df
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

        logits = model(
            X_tensor
        )

        probability = (
            torch.sigmoid(
                logits
            )
            .cpu()
            .numpy()
        )


    return probability


# ============================================================
# Main smoke test
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


    print(
        "\n"
        + "=" * 100
    )

    print(
        "TRUE SOURCE-FREE DEPLOYMENT SMOKE TEST"
    )

    print(
        "=" * 100
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


    (
        package_dir,
        metadata,
        preprocessor,
        model,
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
        X_target_raw,
    ) = load_unlabeled_target(
        target=
            args.target,

        raw_features=
            raw_features,
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
            "Transformed input dimension "
            "does not match model metadata."
        )


    probability = (
        predict_probability(
            model=
                model,

            X=
                X_target,

            device=
                device,
        )
    )


    if not np.all(
        np.isfinite(
            probability
        )
    ):

        raise RuntimeError(
            "Non-finite probabilities produced."
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
            "Probability outside [0, 1]."
        )


    predicted_class = (
        probability
        >= float(
            metadata[
                "classification_threshold"
            ]
        )
    ).astype(int)


    print(
        "\n"
        + "-" * 100
    )

    print(
        "SOURCE-FREE INPUT AUDIT"
    )

    print(
        "-" * 100
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
        "Saved metadata loaded:"
        " YES"
    )


    print(
        "\nPackage:",
        package_dir
    )

    print(
        "Target feature file:",
        target_path
    )


    print(
        "\n"
        + "-" * 100
    )

    print(
        "PREDICTION SUMMARY"
    )

    print(
        "-" * 100
    )


    print(
        "Target N:",
        len(
            probability
        )
    )

    print(
        "Raw feature count:",
        len(
            raw_features
        )
    )

    print(
        "Transformed dimension:",
        X_target.shape[1]
    )

    print(
        "Mean predicted probability:",
        f"{probability.mean():.6f}"
    )

    print(
        "Min predicted probability:",
        f"{probability.min():.6f}"
    )

    print(
        "Max predicted probability:",
        f"{probability.max():.6f}"
    )

    print(
        "Predicted positive rate:",
        f"{predicted_class.mean():.6f}"
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "SOURCE-FREE PACKAGE TEST: PASSED"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()