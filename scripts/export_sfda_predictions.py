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
from heart_sfda.utils.device import get_device


DATA_DIR = ROOT / "data" / "processed"

SOURCE_MODEL_DIR = (
    ROOT / "models" / "source" / "mlp"
)

ADAPTED_MODEL_DIR = (
    ROOT / "models" / "adapted"
)

RESULT_DIR = (
    ROOT / "results" / "statistics"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 42
N_FOLDS = 5


TARGETS = {
    "hungary": "hungary.csv",
    "switzerland": "switzerland.csv",
    "va_long_beach": "va_long_beach.csv",
}


METHOD_FILES = {
    "pseudo_label_sfda":
        "pseudo_label_fold_{fold}.pt",

    "entropy_sfda":
        "entropy_fold_{fold}.pt",

    "reliability_gated_sfda":
        "reliability_gated_fold_{fold}.pt",
}


def build_model(device):
    source_checkpoint = torch.load(
        SOURCE_MODEL_DIR
        / "cleveland_core8_mlp.pt",
        map_location=device,
        weights_only=False,
    )

    model = HeartMLP(
        input_dim=
            source_checkpoint["input_dim"],

        hidden_dims=tuple(
            source_checkpoint["hidden_dims"]
        ),

        dropout=
            source_checkpoint["dropout"],
    ).to(device)

    return model


def load_source_model(device):
    checkpoint = torch.load(
        SOURCE_MODEL_DIR
        / "cleveland_core8_mlp.pt",
        map_location=device,
        weights_only=False,
    )

    model = build_model(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def load_adapted_model(
    target,
    filename,
    device,
):
    checkpoint = torch.load(
        ADAPTED_MODEL_DIR
        / target
        / filename,
        map_location=device,
        weights_only=False,
    )

    model = build_model(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def predict(model, X, device):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    with torch.no_grad():

        probabilities = (
            torch.sigmoid(
                model(X_tensor)
            )
            .cpu()
            .numpy()
        )

    return probabilities


def main():

    device = get_device()

    preprocessor = joblib.load(
        SOURCE_MODEL_DIR
        / "cleveland_core8_preprocessor.joblib"
    )

    all_rows = []

    for target, filename in TARGETS.items():

        print(
            f"\nProcessing {target}..."
        )

        df = pd.read_csv(
            DATA_DIR / filename
        )

        X = (
            preprocessor
            .transform(
                df[CORE8]
            )
            .astype(np.float32)
        )

        y = (
            df["target"]
            .to_numpy()
        )

        row_ids = (
            df["row_id"]
            .to_numpy()
        )

        # ------------------------------------------
        # Source-only probability
        # ------------------------------------------

        source_model = (
            load_source_model(device)
        )

        source_prob = predict(
            source_model,
            X,
            device,
        )

        predictions = {
            "source_only":
                source_prob
        }

        # ------------------------------------------
        # Reconstruct exact cross-fit folds
        # ------------------------------------------

        kfold = KFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=SEED,
        )

        for (
            method,
            filename_pattern,
        ) in METHOD_FILES.items():

            method_prob = np.zeros(
                len(df),
                dtype=float,
            )

            for fold, (
                _,
                test_indices,
            ) in enumerate(
                kfold.split(X),
                start=1,
            ):

                adapted_model = (
                    load_adapted_model(
                        target,
                        filename_pattern.format(
                            fold=fold
                        ),
                        device,
                    )
                )

                method_prob[
                    test_indices
                ] = predict(
                    adapted_model,
                    X[test_indices],
                    device,
                )

            predictions[
                method
            ] = method_prob

        # ------------------------------------------
        # Patient-level table
        # ------------------------------------------

        for i in range(len(df)):

            all_rows.append(
                {
                    "target": target,
                    "row_id": row_ids[i],
                    "y_true": int(y[i]),

                    "source_only":
                        float(
                            predictions[
                                "source_only"
                            ][i]
                        ),

                    "pseudo_label_sfda":
                        float(
                            predictions[
                                "pseudo_label_sfda"
                            ][i]
                        ),

                    "entropy_sfda":
                        float(
                            predictions[
                                "entropy_sfda"
                            ][i]
                        ),

                    "reliability_gated_sfda":
                        float(
                            predictions[
                                "reliability_gated_sfda"
                            ][i]
                        ),
                }
            )

    result = pd.DataFrame(
        all_rows
    )

    output_path = (
        RESULT_DIR
        / "patient_level_sfda_predictions.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved:",
        output_path
    )

    print(
        "\nRows:",
        len(result)
    )


if __name__ == "__main__":
    main()