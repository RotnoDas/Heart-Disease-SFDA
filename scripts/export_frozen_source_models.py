from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import joblib
import numpy as np
import torch


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


SINGLE_SCRIPT = (
    ROOT
    / "scripts"
    / "run_cross_source_replication.py"
)

DUAL_SCRIPT = (
    ROOT
    / "scripts"
    / "run_dlar_lcl_cross_source.py"
)


SINGLE_OUTPUT = (
    ROOT
    / "models"
    / "source"
)

DUAL_OUTPUT = (
    ROOT
    / "models"
    / "source_dual_head"
)


# ============================================================
# Dynamic import of canonical experiment scripts
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

    if spec is None or spec.loader is None:
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


single = load_script_module(
    "canonical_cross_source",
    SINGLE_SCRIPT,
)

dual = load_script_module(
    "canonical_dlar_lcl",
    DUAL_SCRIPT,
)


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


def write_json(
    path,
    payload,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
        )


def get_feature_names(
    preprocessor,
):

    try:

        names = (
            preprocessor
            .get_feature_names_out()
        )

        return [
            str(x)
            for x in names
        ]

    except Exception:

        return []


# ============================================================
# Reload verification:
# saved state_dict must reproduce original model outputs
# ============================================================

def verify_single_reload(
    package_dir,
    metadata,
    X,
    original_probability,
    device,
):

    from heart_sfda.models.mlp import (
        HeartMLP,
    )

    reloaded = HeartMLP(
        input_dim=
            metadata[
                "input_dim"
            ],

        hidden_dims=
            tuple(
                metadata[
                    "hidden_dims"
                ]
            ),

        dropout=
            metadata[
                "dropout"
            ],
    ).to(device)


    state_dict = torch.load(
        package_dir
        / "model.pt",
        map_location=device,
    )


    reloaded.load_state_dict(
        state_dict
    )

    reloaded.eval()


    reloaded_probability = (
        single.predict_probability(
            reloaded,
            X,
            device,
        )
    )


    max_difference = float(
        np.max(
            np.abs(
                original_probability
                -
                reloaded_probability
            )
        )
    )


    if max_difference > 1e-7:

        raise RuntimeError(
            "Reload verification failed: "
            f"{package_dir} "
            f"max diff={max_difference}"
        )


    return max_difference


def verify_dual_reload(
    package_dir,
    metadata,
    X,
    original_model,
    device,
):

    from heart_sfda.models.dual_head_mlp import (
        DualHeadHeartMLP,
    )


    reloaded = DualHeadHeartMLP(
        input_dim=
            metadata[
                "input_dim"
            ],

        hidden_dims=
            tuple(
                metadata[
                    "hidden_dims"
                ]
            ),

        dropout=
            metadata[
                "dropout"
            ],
    ).to(device)


    state_dict = torch.load(
        package_dir
        / "model.pt",
        map_location=device,
    )


    reloaded.load_state_dict(
        state_dict
    )

    reloaded.eval()

    original_model.eval()


    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )


    with torch.no_grad():

        (
            _,
            original_logits_1,
            original_logits_2,
        ) = original_model(
            X_tensor
        )

        (
            _,
            reload_logits_1,
            reload_logits_2,
        ) = reloaded(
            X_tensor
        )


    diff_1 = float(
        torch.max(
            torch.abs(
                original_logits_1
                -
                reload_logits_1
            )
        )
        .cpu()
        .item()
    )


    diff_2 = float(
        torch.max(
            torch.abs(
                original_logits_2
                -
                reload_logits_2
            )
        )
        .cpu()
        .item()
    )


    max_difference = max(
        diff_1,
        diff_2,
    )


    if max_difference > 1e-7:

        raise RuntimeError(
            "Dual-head reload verification failed: "
            f"{package_dir} "
            f"max diff={max_difference}"
        )


    return max_difference


# ============================================================
# Single-head export
# ============================================================

def export_single_head(
    device,
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "EXPORTING SINGLE-HEAD "
        "SOURCE MODELS"
    )

    print(
        "=" * 100
    )


    SINGLE_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Canonical source epoch plan
    # --------------------------------------------------------

    epoch_plan_df = (
        single.pd.read_csv(
            single.EPOCH_PLAN_FILE
        )
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


    manifest_rows = []


    for source_name in (
        single.DOMAINS
    ):

        print(
            f"\nSource: {source_name}"
        )


        # ----------------------------------------------------
        # Canonical source data loading
        # ----------------------------------------------------

        source_df = (
            single.load_dataset(
                source_name
            )
        )


        X_source_raw = (
            source_df[
                single.CORE8
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
        # Canonical source-only preprocessor
        # ----------------------------------------------------

        preprocessor = (
            single
            .build_linear_preprocessor()
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


        transformed_features = (
            get_feature_names(
                preprocessor
            )
        )


        for seed in single.SEEDS:

            print(
                f"  seed={seed}"
            )


            model = (
                single.train_source_model(
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


            package_dir = (
                SINGLE_OUTPUT
                / source_name
                / f"seed_{seed}"
            )


            package_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            # ------------------------------------------------
            # Save model weights
            # ------------------------------------------------

            model_path = (
                package_dir
                / "model.pt"
            )


            torch.save(
                model.state_dict(),
                model_path,
            )


            # ------------------------------------------------
            # Save fitted source preprocessor
            # ------------------------------------------------

            preprocessor_path = (
                package_dir
                / "preprocessor.joblib"
            )


            joblib.dump(
                preprocessor,
                preprocessor_path,
            )


            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = {
                "artifact_version":
                    "1.0",

                "model_family":
                    "HeartMLP",

                "architecture":
                    "single_head",

                "source_domain":
                    source_name,

                "seed":
                    int(
                        seed
                    ),

                "source_epochs":
                    int(
                        epoch_plan[
                            source_name
                        ]
                    ),

                "input_dim":
                    int(
                        X_source.shape[1]
                    ),

                "hidden_dims":
                    [32, 16],

                "dropout":
                    0.20,

                "raw_features":
                    list(
                        single.CORE8
                    ),

                "transformed_features":
                    transformed_features,

                "target_definition":
                    "num > 0",

                "classification_threshold":
                    0.5,

                "source_n":
                    int(
                        len(
                            source_df
                        )
                    ),

                "source_positive_n":
                    int(
                        y_source.sum()
                    ),

                "source_prevalence":
                    float(
                        y_source.mean()
                    ),

                "training": {
                    "optimizer":
                        "AdamW",

                    "learning_rate":
                        float(
                            single.SOURCE_LR
                        ),

                    "weight_decay":
                        float(
                            single
                            .SOURCE_WEIGHT_DECAY
                        ),

                    "batch_size":
                        int(
                            single
                            .SOURCE_BATCH_SIZE
                        ),
                },

                "source_data_required_at_adaptation":
                    False,

                "canonical_training_script":
                    (
                        "scripts/"
                        "run_cross_source_replication.py"
                    ),
            }


            metadata_path = (
                package_dir
                / "metadata.json"
            )


            write_json(
                metadata_path,
                metadata,
            )


            # ------------------------------------------------
            # Serialization integrity check
            # ------------------------------------------------

            original_probability = (
                single.predict_probability(
                    model,
                    X_source,
                    device,
                )
            )


            max_reload_diff = (
                verify_single_reload(
                    package_dir=
                        package_dir,

                    metadata=
                        metadata,

                    X=
                        X_source,

                    original_probability=
                        original_probability,

                    device=
                        device,
                )
            )


            # ------------------------------------------------
            # Hashes
            # ------------------------------------------------

            hashes = {
                "model_sha256":
                    sha256_file(
                        model_path
                    ),

                "preprocessor_sha256":
                    sha256_file(
                        preprocessor_path
                    ),

                "metadata_sha256":
                    sha256_file(
                        metadata_path
                    ),
            }


            hash_path = (
                package_dir
                / "hashes.json"
            )


            write_json(
                hash_path,
                hashes,
            )


            manifest_rows.append(
                {
                    "architecture":
                        "single_head",

                    "source":
                        source_name,

                    "seed":
                        int(seed),

                    "epochs":
                        int(
                            epoch_plan[
                                source_name
                            ]
                        ),

                    "input_dim":
                        int(
                            X_source.shape[1]
                        ),

                    "max_reload_difference":
                        max_reload_diff,

                    "package":
                        str(
                            package_dir
                            .relative_to(
                                ROOT
                            )
                        ),
                }
            )


    manifest_path = (
        SINGLE_OUTPUT
        / "manifest.json"
    )


    write_json(
        manifest_path,
        manifest_rows,
    )


    print(
        "\nSingle-head packages:",
        len(
            manifest_rows
        )
    )

    print(
        "Manifest:",
        manifest_path
    )


# ============================================================
# Dual-head DLAR-LCL export
# ============================================================

def export_dual_head(
    device,
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "EXPORTING DUAL-HEAD "
        "DLAR-LCL SOURCE MODELS"
    )

    print(
        "=" * 100
    )


    DUAL_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )


    manifest_rows = []


    for source_name in dual.DOMAINS:

        print(
            f"\nSource: {source_name}"
        )


        source_df = (
            dual.load_domain(
                source_name
            )
        )


        X_source_raw = (
            source_df[
                dual.CORE8
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


        preprocessor = (
            dual
            .build_linear_preprocessor()
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


        transformed_features = (
            get_feature_names(
                preprocessor
            )
        )


        for seed in dual.SEEDS:

            print(
                f"  seed={seed}"
            )


            (
                model,
                training_stats,
            ) = dual.train_source_model(
                X_source=
                    X_source,

                y_source=
                    y_source,

                seed=
                    seed,

                device=
                    device,
            )


            package_dir = (
                DUAL_OUTPUT
                / source_name
                / f"seed_{seed}"
            )


            package_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            model_path = (
                package_dir
                / "model.pt"
            )


            torch.save(
                model.state_dict(),
                model_path,
            )


            preprocessor_path = (
                package_dir
                / "preprocessor.joblib"
            )


            joblib.dump(
                preprocessor,
                preprocessor_path,
            )


            metadata = {
                "artifact_version":
                    "1.0",

                "model_family":
                    "DualHeadHeartMLP",

                "architecture":
                    "dual_head",

                "source_domain":
                    source_name,

                "seed":
                    int(seed),

                "source_epochs":
                    int(
                        dual.SOURCE_EPOCHS
                    ),

                "input_dim":
                    int(
                        X_source.shape[1]
                    ),

                "hidden_dims":
                    [32, 16],

                "dropout":
                    0.20,

                "raw_features":
                    list(
                        dual.CORE8
                    ),

                "transformed_features":
                    transformed_features,

                "target_definition":
                    "num > 0",

                "classification_threshold":
                    0.5,

                "source_n":
                    int(
                        len(
                            source_df
                        )
                    ),

                "source_positive_n":
                    int(
                        y_source.sum()
                    ),

                "source_prevalence":
                    float(
                        y_source.mean()
                    ),

                "training": {
                    "optimizer":
                        "Adam",

                    "learning_rate":
                        float(
                            dual.SOURCE_LR
                        ),

                    "weight_decay":
                        float(
                            dual
                            .SOURCE_WEIGHT_DECAY
                        ),

                    "batch_size":
                        int(
                            dual
                            .SOURCE_BATCH_SIZE
                        ),
                },

                "source_training_stats":
                    {
                        key:
                            float(value)

                        for key, value
                        in training_stats.items()
                    },

                "source_data_required_at_adaptation":
                    False,

                "canonical_training_script":
                    (
                        "scripts/"
                        "run_dlar_lcl_cross_source.py"
                    ),
            }


            metadata_path = (
                package_dir
                / "metadata.json"
            )


            write_json(
                metadata_path,
                metadata,
            )


            max_reload_diff = (
                verify_dual_reload(
                    package_dir=
                        package_dir,

                    metadata=
                        metadata,

                    X=
                        X_source,

                    original_model=
                        model,

                    device=
                        device,
                )
            )


            hashes = {
                "model_sha256":
                    sha256_file(
                        model_path
                    ),

                "preprocessor_sha256":
                    sha256_file(
                        preprocessor_path
                    ),

                "metadata_sha256":
                    sha256_file(
                        metadata_path
                    ),
            }


            write_json(
                package_dir
                / "hashes.json",
                hashes,
            )


            manifest_rows.append(
                {
                    "architecture":
                        "dual_head",

                    "source":
                        source_name,

                    "seed":
                        int(seed),

                    "epochs":
                        int(
                            dual.SOURCE_EPOCHS
                        ),

                    "input_dim":
                        int(
                            X_source.shape[1]
                        ),

                    "max_reload_difference":
                        max_reload_diff,

                    "package":
                        str(
                            package_dir
                            .relative_to(
                                ROOT
                            )
                        ),
                }
            )


    manifest_path = (
        DUAL_OUTPUT
        / "manifest.json"
    )


    write_json(
        manifest_path,
        manifest_rows,
    )


    print(
        "\nDual-head packages:",
        len(
            manifest_rows
        )
    )

    print(
        "Manifest:",
        manifest_path
    )


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
        "Device:",
        device
    )


    export_single_head(
        device
    )


    export_dual_head(
        device
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "SOURCE MODEL EXPORT COMPLETE"
    )

    print(
        "=" * 100
    )


    print(
        "\nSingle-head root:",
        SINGLE_OUTPUT
    )

    print(
        "Dual-head root:",
        DUAL_OUTPUT
    )


if __name__ == "__main__":
    main()