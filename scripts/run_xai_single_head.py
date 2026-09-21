from pathlib import Path
import argparse
import importlib.util
import json
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn


# ============================================================
# Project
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

REFERENCE_FILE = (
    ROOT
    / "results"
    / "cross_source"
    / "cross_source_patient_predictions.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "explainability"
    / "xai_single_head"
)

CACHE_DIR = (
    OUTPUT_ROOT
    / "cache_v1"
)


# ============================================================
# Frozen XAI settings
# ============================================================

XAI_VERSION = (
    "v1_deepshap_logit_targetbg32_raw8"
)

REQUIRED_SHAP_VERSION = "0.52.0"

BACKGROUND_SIZE = 32

TOP_K = 3

REFERENCE_TOLERANCE = 1e-6

ADDITIVITY_TOLERANCE = 1e-3


# ============================================================
# Canonical experiment module
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


METHODS = list(
    canonical.METHODS
)

MODEL_STATES = [
    "source_only",
    *METHODS,
]


# ============================================================
# Model wrapper
#
# DeepExplainer is most convenient with output shape [N, 1].
# Original HeartMLP returns [N].
# ============================================================

class LogitWrapper(nn.Module):

    def __init__(
        self,
        model,
    ):

        super().__init__()

        self.model = model


    def forward(
        self,
        x,
    ):

        logits = self.model(x)

        return logits.unsqueeze(1)


# ============================================================
# Source package
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
        input_dim=int(
            metadata[
                "input_dim"
            ]
        ),

        hidden_dims=tuple(
            metadata[
                "hidden_dims"
            ]
        ),

        dropout=float(
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
        metadata,
        preprocessor,
        model,
    )


# ============================================================
# Target predictors only
#
# IMPORTANT:
# target and num are never read here.
# ============================================================

def load_unlabeled_target(
    target,
    raw_features,
):

    target_path = (
        canonical.DATA_DIR
        / canonical.DOMAINS[
            target
        ]
    )


    usecols = [
        "row_id",
        *raw_features,
    ]


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
            "Target-label leakage detected."
        )


    if not df["row_id"].is_unique:

        raise RuntimeError(
            f"Duplicate row_id: {target}"
        )


    return df


# ============================================================
# Exact transformed -> raw feature map
# ============================================================

def build_raw_feature_map(
    preprocessor,
    raw_features,
):

    mapping = []


    numeric_columns = list(
        preprocessor
        .transformers_[0][2]
    )


    mapping.extend(
        numeric_columns
    )


    categorical_columns = list(
        preprocessor
        .transformers_[1][2]
    )


    categorical_pipeline = (
        preprocessor
        .named_transformers_[
            "categorical"
        ]
    )


    onehot = (
        categorical_pipeline
        .named_steps[
            "onehot"
        ]
    )


    categories = (
        onehot.categories_
    )


    for (
        feature,
        feature_categories,
    ) in zip(
        categorical_columns,
        categories,
    ):

        mapping.extend(
            [
                feature
            ]
            *
            len(
                feature_categories
            )
        )


    transformed_names = list(
        preprocessor
        .get_feature_names_out()
    )


    if (
        len(mapping)
        != len(
            transformed_names
        )
    ):

        raise RuntimeError(
            "Transformed feature mapping "
            "length mismatch."
        )


    unknown = (
        set(mapping)
        -
        set(raw_features)
    )


    if unknown:

        raise RuntimeError(
            f"Unknown raw features: {unknown}"
        )


    return (
        mapping,
        transformed_names,
    )


# ============================================================
# Aggregate transformed SHAP -> CORE8
#
# Signed SHAP values are summed.
# This preserves local additive contribution.
# ============================================================

def aggregate_shap_to_raw(
    transformed_shap,
    transformed_to_raw,
    raw_features,
):

    output = np.zeros(
        (
            transformed_shap.shape[0],
            len(raw_features),
        ),
        dtype=float,
    )


    raw_index = {
        feature: i
        for i, feature
        in enumerate(
            raw_features
        )
    }


    for transformed_index, raw_feature in enumerate(
        transformed_to_raw
    ):

        output[
            :,
            raw_index[
                raw_feature
            ],
        ] += (
            transformed_shap[
                :,
                transformed_index
            ]
        )


    return output


# ============================================================
# Deterministic unlabeled target background
# ============================================================

def choose_background(
    X_adapt,
    adapt_indices,
    seed,
):

    n_background = min(
        BACKGROUND_SIZE,
        len(
            X_adapt
        ),
    )


    rng = np.random.default_rng(
        seed
    )


    local_indices = np.sort(
        rng.choice(
            len(
                X_adapt
            ),
            size=n_background,
            replace=False,
        )
    )


    X_background = (
        X_adapt[
            local_indices
        ]
    )


    absolute_indices = (
        adapt_indices[
            local_indices
        ]
    )


    return (
        X_background,
        absolute_indices,
    )


# ============================================================
# DeepSHAP
# ============================================================

def compute_deepshap(
    model,
    X_background,
    X_explain,
    device,
    transformed_to_raw,
    raw_features,
):

    wrapper = LogitWrapper(
        model
    ).to(device)

    wrapper.eval()


    background_tensor = torch.tensor(
        X_background,
        dtype=torch.float32,
        device=device,
    )


    explain_tensor = torch.tensor(
        X_explain,
        dtype=torch.float32,
        device=device,
    )


    explainer = shap.DeepExplainer(
        wrapper,
        background_tensor,
    )


    shap_values = (
        explainer.shap_values(
            explain_tensor,
            check_additivity=False,
        )
    )


    if isinstance(
        shap_values,
        list,
    ):

        if len(
            shap_values
        ) != 1:

            raise RuntimeError(
                "Unexpected multi-output SHAP."
            )

        shap_values = (
            shap_values[0]
        )


    transformed_shap = np.asarray(
        shap_values,
        dtype=float,
    )


    if (
        transformed_shap.ndim
        == 3
        and
        transformed_shap.shape[-1]
        == 1
    ):

        transformed_shap = (
            transformed_shap[
                ...,
                0
            ]
        )


    if (
        transformed_shap.ndim
        != 2
    ):

        raise RuntimeError(
            "Unexpected SHAP shape: "
            f"{transformed_shap.shape}"
        )


    if (
        transformed_shap.shape[1]
        != X_explain.shape[1]
    ):

        raise RuntimeError(
            "SHAP/input dimension mismatch."
        )


    expected_value = float(
        np.asarray(
            explainer.expected_value
        )
        .reshape(-1)[0]
    )


    with torch.no_grad():

        logits = (
            wrapper(
                explain_tensor
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )


    reconstructed_logits = (
        expected_value
        +
        transformed_shap.sum(
            axis=1
        )
    )


    additivity_error = np.abs(
        logits
        -
        reconstructed_logits
    )


    max_additivity_error = float(
        additivity_error.max()
    )


    mean_additivity_error = float(
        additivity_error.mean()
    )


    if (
        max_additivity_error
        >
        ADDITIVITY_TOLERANCE
    ):

        raise RuntimeError(
            "DeepSHAP additivity check failed: "
            f"max_error="
            f"{max_additivity_error:.8f}"
        )


    raw_shap = (
        aggregate_shap_to_raw(
            transformed_shap=
                transformed_shap,

            transformed_to_raw=
                transformed_to_raw,

            raw_features=
                raw_features,
        )
    )


    return {
        "raw_shap":
            raw_shap,

        "base_value":
            expected_value,

        "max_additivity_error":
            max_additivity_error,

        "mean_additivity_error":
            mean_additivity_error,
    }


# ============================================================
# Canonical prediction
# ============================================================

def predict_probability(
    model,
    X,
    device,
):

    return np.asarray(
        canonical.predict_probability(
            model,
            X,
            device,
        ),
        dtype=float,
    )


# ============================================================
# Exact canonical adaptation
# ============================================================

def adapt_model(
    method,
    source_model,
    X_adapt,
    X_adapt_raw,
    device,
):

    missingness_rate = float(
        X_adapt_raw
        .isna()
        .mean()
        .mean()
    )


    if (
        method
        == "pseudo_label_sfda"
    ):

        return (
            canonical
            .adapt_with_pseudo_labels(
                source_model=
                    source_model,

                X_target_unlabeled=
                    X_adapt,

                device=device,

                positive_threshold=
                    canonical
                    .PSEUDO_POSITIVE_THRESHOLD,

                negative_threshold=
                    canonical
                    .PSEUDO_NEGATIVE_THRESHOLD,

                learning_rate=
                    canonical
                    .PSEUDO_LR,

                weight_decay=
                    canonical
                    .PSEUDO_WEIGHT_DECAY,

                epochs=
                    canonical
                    .PSEUDO_EPOCHS,

                batch_size=
                    canonical
                    .PSEUDO_BATCH_SIZE,

                min_selected=
                    canonical
                    .PSEUDO_MIN_SELECTED,
            )
        )


    if (
        method
        == "entropy_sfda"
    ):

        return (
            canonical
            .adapt_with_entropy(
                source_model=
                    source_model,

                X_target_unlabeled=
                    X_adapt,

                device=device,

                learning_rate=
                    canonical
                    .ENTROPY_LR,

                weight_decay=
                    canonical
                    .ENTROPY_WEIGHT_DECAY,

                epochs=
                    canonical
                    .ENTROPY_EPOCHS,

                batch_size=
                    canonical
                    .ENTROPY_BATCH_SIZE,
            )
        )


    if (
        method
        == "reliability_gated_sfda"
    ):

        return (
            canonical
            .adapt_reliability_gated(
                source_model=
                    source_model,

                X_target_unlabeled=
                    X_adapt,

                device=device,

                missingness_rate=
                    missingness_rate,

                selection_fraction=
                    canonical
                    .RG_SELECTION_FRACTION,

                minimum_per_class=
                    canonical
                    .RG_MINIMUM_PER_CLASS,

                learning_rate=
                    canonical
                    .RG_LR,

                weight_decay=
                    canonical
                    .RG_WEIGHT_DECAY,

                epochs=
                    canonical
                    .RG_EPOCHS,

                lambda_pseudo=
                    canonical
                    .RG_LAMBDA_PSEUDO,

                lambda_entropy=
                    canonical
                    .RG_LAMBDA_ENTROPY,

                lambda_anchor=
                    canonical
                    .RG_LAMBDA_ANCHOR,

                lambda_prior=
                    canonical
                    .RG_LAMBDA_PRIOR,
            )
        )


    if (
        method
        == "conservative_candidate"
    ):

        return (
            canonical
            .adapt_conservative(
                source_model=
                    source_model,

                X_target_unlabeled=
                    X_adapt,

                device=device,

                missingness_rate=
                    missingness_rate,

                reliability_mode=
                    "full",

                selection_fraction=
                    canonical
                    .CONS_SELECTION_FRACTION,

                minimum_per_class=
                    canonical
                    .CONS_MINIMUM_PER_CLASS,

                learning_rate=
                    canonical
                    .CONS_LR,

                weight_decay=
                    canonical
                    .CONS_WEIGHT_DECAY,

                epochs=
                    canonical
                    .CONS_EPOCHS,

                lambda_entropy=
                    canonical
                    .CONS_LAMBDA_ENTROPY,

                lambda_anchor=
                    canonical
                    .CONS_LAMBDA_ANCHOR,

                lambda_prior=
                    canonical
                    .CONS_LAMBDA_PRIOR,
            )
        )


    raise ValueError(
        f"Unknown method: {method}"
    )


# ============================================================
# Scalar adaptation diagnostics
# ============================================================

def scalar_stats(
    stats,
):

    output = {}


    if stats is None:

        return output


    for key, value in stats.items():

        if isinstance(
            value,
            (
                bool,
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):

            output[
                f"adapt_{key}"
            ] = (
                float(value)
                if not isinstance(
                    value,
                    bool,
                )
                else bool(value)
            )


    return output


# ============================================================
# Patient SHAP rows
# ============================================================

def make_patient_rows(
    source,
    target,
    seed,
    fold,
    model_state,
    row_ids,
    probabilities,
    raw_shap,
    base_value,
    raw_features,
):

    rows = []


    for i in range(
        len(
            row_ids
        )
    ):

        row = {
            "xai_version":
                XAI_VERSION,

            "source":
                source,

            "target":
                target,

            "seed":
                int(seed),

            "fold":
                int(fold),

            "row_id":
                str(
                    row_ids[i]
                ),

            "model_state":
                model_state,

            "probability":
                float(
                    probabilities[i]
                ),

            "base_value_logit":
                float(
                    base_value
                ),
        }


        for j, feature in enumerate(
            raw_features
        ):

            row[
                f"shap_{feature}"
            ] = float(
                raw_shap[
                    i,
                    j
                ]
            )


        rows.append(
            row
        )


    return rows


# ============================================================
# Reference verification
#
# No y_true column is read.
# ============================================================

def verify_against_frozen_predictions(
    patient_df,
    source,
    target,
    seed,
):

    usecols = [
        "source",
        "target",
        "seed",
        "row_id",
        *MODEL_STATES,
    ]


    reference = pd.read_csv(
        REFERENCE_FILE,
        usecols=usecols,
    )


    reference = (
        reference[
            (
                reference[
                    "source"
                ]
                == source
            )
            &
            (
                reference[
                    "target"
                ]
                == target
            )
            &
            (
                reference[
                    "seed"
                ]
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


    verification = []


    for state in MODEL_STATES:

        generated = (
            patient_df[
                patient_df[
                    "model_state"
                ]
                == state
            ][
                [
                    "row_id",
                    "probability",
                ]
            ]
            .copy()
        )


        generated = (
            generated.rename(
                columns={
                    "probability":
                        "generated",
                }
            )
        )


        expected = (
            reference[
                [
                    "row_id",
                    state,
                ]
            ]
            .rename(
                columns={
                    state:
                        "reference",
                }
            )
        )


        comparison = (
            expected.merge(
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
                "Reference patient alignment "
                f"failed for {state}"
            )


        difference = np.abs(
            comparison[
                "reference"
            ]
            .to_numpy(
                dtype=float
            )
            -
            comparison[
                "generated"
            ]
            .to_numpy(
                dtype=float
            )
        )


        max_difference = float(
            difference.max()
        )


        verification.append(
            {
                "source":
                    source,

                "target":
                    target,

                "seed":
                    int(seed),

                "model_state":
                    state,

                "max_probability_difference":
                    max_difference,

                "mean_probability_difference":
                    float(
                        difference.mean()
                    ),

                "passed":
                    bool(
                        max_difference
                        <=
                        REFERENCE_TOLERANCE
                    ),
            }
        )


        if (
            max_difference
            >
            REFERENCE_TOLERANCE
        ):

            raise RuntimeError(
                "Frozen prediction mismatch: "
                f"{source}->{target}, "
                f"seed={seed}, "
                f"{state}, "
                f"max_diff="
                f"{max_difference:.12f}"
            )


    return pd.DataFrame(
        verification
    )


# ============================================================
# One source-target-seed job
# ============================================================

def process_job(
    source,
    target,
    seed,
    device,
    overwrite,
):

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    stem = (
        f"{source}__"
        f"{target}__"
        f"seed_{seed}"
    )


    patient_path = (
        CACHE_DIR
        / f"{stem}__patient.csv.gz"
    )


    fold_path = (
        CACHE_DIR
        / f"{stem}__fold.csv"
    )


    verification_path = (
        CACHE_DIR
        / f"{stem}__verification.csv"
    )


    if (
        patient_path.exists()
        and
        fold_path.exists()
        and
        verification_path.exists()
        and
        not overwrite
    ):

        print(
            f"SKIP cached: "
            f"{source}->{target} "
            f"seed={seed}"
        )

        return


    print(
        "\n"
        + "=" * 110
    )

    print(
        f"XAI: {source} -> {target}, "
        f"seed={seed}"
    )

    print(
        "=" * 110
    )


    (
        metadata,
        preprocessor,
        source_model,
    ) = load_source_package(
        source=
            source,

        seed=
            seed,

        device=
            device,
    )


    raw_features = list(
        metadata[
            "raw_features"
        ]
    )


    (
        transformed_to_raw,
        transformed_names,
    ) = build_raw_feature_map(
        preprocessor=
            preprocessor,

        raw_features=
            raw_features,
    )


    target_df = (
        load_unlabeled_target(
            target=
                target,

            raw_features=
                raw_features,
        )
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
        != len(
            transformed_names
        )
    ):

        raise RuntimeError(
            "Transformed feature-count mismatch."
        )


    kfold = canonical.KFold(
        n_splits=
            canonical.N_FOLDS,

        shuffle=True,

        random_state=
            canonical
            .FOLD_RANDOM_STATE,
    )


    patient_rows = []

    fold_rows = []


    adaptation_seed_base = (
        seed
        * 100000

        +

        canonical.DOMAIN_OFFSET[
            source
        ]
        * 1000

        +

        canonical.DOMAIN_OFFSET[
            target
        ]
        * 100
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
            f"  fold {fold}/"
            f"{canonical.N_FOLDS}"
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


        X_adapt_raw = (
            X_target_raw
            .iloc[
                adapt_indices
            ]
        )


        test_row_ids = (
            row_ids[
                test_indices
            ]
        )


        adaptation_seed = (
            adaptation_seed_base
            +
            fold
        )


        # ----------------------------------------------------
        # Deterministic unlabeled target background.
        #
        # Local NumPy Generator does NOT modify global RNG
        # used by canonical adaptation.
        # ----------------------------------------------------

        (
            X_background,
            background_absolute_indices,
        ) = choose_background(
            X_adapt=
                X_adapt,

            adapt_indices=
                adapt_indices,

            seed=
                adaptation_seed
                +
                9_000_000,
        )


        # ====================================================
        # Source-only explanation
        # ====================================================

        source_probability = (
            predict_probability(
                source_model,
                X_test,
                device,
            )
        )


        source_shap = (
            compute_deepshap(
                model=
                    source_model,

                X_background=
                    X_background,

                X_explain=
                    X_test,

                device=
                    device,

                transformed_to_raw=
                    transformed_to_raw,

                raw_features=
                    raw_features,
            )
        )


        patient_rows.extend(
            make_patient_rows(
                source=
                    source,

                target=
                    target,

                seed=
                    seed,

                fold=
                    fold,

                model_state=
                    "source_only",

                row_ids=
                    test_row_ids,

                probabilities=
                    source_probability,

                raw_shap=
                    source_shap[
                        "raw_shap"
                    ],

                base_value=
                    source_shap[
                        "base_value"
                    ],

                raw_features=
                    raw_features,
            )
        )


        fold_rows.append(
            {
                "source":
                    source,

                "target":
                    target,

                "seed":
                    int(seed),

                "fold":
                    int(fold),

                "model_state":
                    "source_only",

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

                "background_n":
                    int(
                        len(
                            X_background
                        )
                    ),

                "background_target_indices":
                    ";".join(
                        str(
                            int(x)
                        )
                        for x
                        in background_absolute_indices
                    ),

                "max_additivity_error":
                    source_shap[
                        "max_additivity_error"
                    ],

                "mean_additivity_error":
                    source_shap[
                        "mean_additivity_error"
                    ],
            }
        )


        # ====================================================
        # Adapted explanations
        # ====================================================

        for method in METHODS:

            print(
                f"    {method}"
            )


            # Exact canonical matched RNG.
            canonical.set_seed(
                adaptation_seed
            )


            (
                adapted_model,
                stats,
            ) = adapt_model(
                method=
                    method,

                source_model=
                    source_model,

                X_adapt=
                    X_adapt,

                X_adapt_raw=
                    X_adapt_raw,

                device=
                    device,
            )


            adapted_probability = (
                predict_probability(
                    adapted_model,
                    X_test,
                    device,
                )
            )


            adapted_shap = (
                compute_deepshap(
                    model=
                        adapted_model,

                    X_background=
                        X_background,

                    X_explain=
                        X_test,

                    device=
                        device,

                    transformed_to_raw=
                        transformed_to_raw,

                    raw_features=
                        raw_features,
                )
            )


            patient_rows.extend(
                make_patient_rows(
                    source=
                        source,

                    target=
                        target,

                    seed=
                        seed,

                    fold=
                        fold,

                    model_state=
                        method,

                    row_ids=
                        test_row_ids,

                    probabilities=
                        adapted_probability,

                    raw_shap=
                        adapted_shap[
                            "raw_shap"
                        ],

                    base_value=
                        adapted_shap[
                            "base_value"
                        ],

                    raw_features=
                        raw_features,
                )
            )


            fold_row = {
                "source":
                    source,

                "target":
                    target,

                "seed":
                    int(seed),

                "fold":
                    int(fold),

                "model_state":
                    method,

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

                "background_n":
                    int(
                        len(
                            X_background
                        )
                    ),

                "background_target_indices":
                    ";".join(
                        str(
                            int(x)
                        )
                        for x
                        in background_absolute_indices
                    ),

                "max_additivity_error":
                    adapted_shap[
                        "max_additivity_error"
                    ],

                "mean_additivity_error":
                    adapted_shap[
                        "mean_additivity_error"
                    ],
            }


            fold_row.update(
                scalar_stats(
                    stats
                )
            )


            fold_rows.append(
                fold_row
            )


            del adapted_model


            if torch.cuda.is_available():

                torch.cuda.empty_cache()


    # ========================================================
    # Job-level data
    # ========================================================

    patient_df = pd.DataFrame(
        patient_rows
    )


    fold_df = pd.DataFrame(
        fold_rows
    )


    # Every patient must have exactly one explanation
    # for every model state.
    expected_rows = (
        len(
            target_df
        )
        *
        len(
            MODEL_STATES
        )
    )


    if (
        len(
            patient_df
        )
        != expected_rows
    ):

        raise RuntimeError(
            "Unexpected patient SHAP row count: "
            f"{len(patient_df)} != "
            f"{expected_rows}"
        )


    verification_df = (
        verify_against_frozen_predictions(
            patient_df=
                patient_df,

            source=
                source,

            target=
                target,

            seed=
                seed,
        )
    )


    patient_df.to_csv(
        patient_path,
        index=False,
        compression="gzip",
    )


    fold_df.to_csv(
        fold_path,
        index=False,
    )


    verification_df.to_csv(
        verification_path,
        index=False,
    )


    print(
        "  frozen predictions: PASS"
    )

    print(
        "  patient SHAP cache:",
        patient_path.name
    )


# ============================================================
# Similarity helpers
# ============================================================

def cosine_similarity_rows(
    A,
    B,
):

    numerator = (
        A
        *
        B
    ).sum(
        axis=1
    )


    norm_a = np.linalg.norm(
        A,
        axis=1,
    )


    norm_b = np.linalg.norm(
        B,
        axis=1,
    )


    denominator = (
        norm_a
        *
        norm_b
    )


    similarity = np.empty(
        len(A),
        dtype=float,
    )


    both_zero = (
        (norm_a == 0)
        &
        (norm_b == 0)
    )


    one_zero = (
        (denominator == 0)
        &
        ~both_zero
    )


    normal = (
        denominator
        != 0
    )


    similarity[
        both_zero
    ] = 1.0


    similarity[
        one_zero
    ] = 0.0


    similarity[
        normal
    ] = (
        numerator[
            normal
        ]
        /
        denominator[
            normal
        ]
    )


    return similarity


def patient_topk_overlap(
    source_shap,
    adapted_shap,
    k,
):

    values = []


    for i in range(
        len(
            source_shap
        )
    ):

        source_top = set(
            np.argsort(
                np.abs(
                    source_shap[
                        i
                    ]
                )
            )[
                -k:
            ]
        )


        adapted_top = set(
            np.argsort(
                np.abs(
                    adapted_shap[
                        i
                    ]
                )
            )[
                -k:
            ]
        )


        values.append(
            len(
                source_top
                &
                adapted_top
            )
            /
            k
        )


    return np.asarray(
        values,
        dtype=float,
    )


# ============================================================
# Build aggregate summaries
# ============================================================

def build_summaries():

    patient_files = sorted(
        CACHE_DIR.glob(
            "*__patient.csv.gz"
        )
    )


    fold_files = sorted(
        CACHE_DIR.glob(
            "*__fold.csv"
        )
    )


    verification_files = sorted(
        CACHE_DIR.glob(
            "*__verification.csv"
        )
    )


    if not patient_files:

        raise RuntimeError(
            "No XAI cache files found."
        )


    patient_df = pd.concat(
        [
            pd.read_csv(path)
            for path
            in patient_files
        ],
        ignore_index=True,
    )


    fold_df = pd.concat(
        [
            pd.read_csv(path)
            for path
            in fold_files
        ],
        ignore_index=True,
    )


    verification_df = pd.concat(
        [
            pd.read_csv(path)
            for path
            in verification_files
        ],
        ignore_index=True,
    )


    raw_features = list(
        canonical.CORE8
    )


    shap_columns = [
        f"shap_{feature}"
        for feature
        in raw_features
    ]


    # --------------------------------------------------------
    # Global feature importance
    # --------------------------------------------------------

    importance_rows = []


    grouped = patient_df.groupby(
        [
            "source",
            "target",
            "seed",
            "model_state",
        ],
        sort=True,
    )


    for (
        source,
        target,
        seed,
        state,
    ), group in grouped:

        importance = {
            feature:
                float(
                    np.mean(
                        np.abs(
                            group[
                                f"shap_{feature}"
                            ]
                            .to_numpy(
                                dtype=float
                            )
                        )
                    )
                )

            for feature
            in raw_features
        }


        ordered = sorted(
            importance.items(),
            key=lambda item:
                item[1],
            reverse=True,
        )


        rank = {
            feature:
                i + 1

            for i, (
                feature,
                _
            )
            in enumerate(
                ordered
            )
        }


        for feature in raw_features:

            importance_rows.append(
                {
                    "source":
                        source,

                    "target":
                        target,

                    "seed":
                        int(seed),

                    "model_state":
                        state,

                    "feature":
                        feature,

                    "mean_abs_shap":
                        importance[
                            feature
                        ],

                    "importance_rank":
                        rank[
                            feature
                        ],
                }
            )


    importance_df = pd.DataFrame(
        importance_rows
    )


    # --------------------------------------------------------
    # Source vs adapted stability
    # --------------------------------------------------------

    stability_rows = []


    jobs = (
        patient_df[
            [
                "source",
                "target",
                "seed",
            ]
        ]
        .drop_duplicates()
    )


    for job in jobs.itertuples(
        index=False
    ):

        source = job.source
        target = job.target
        seed = int(
            job.seed
        )


        job_df = patient_df[
            (
                patient_df[
                    "source"
                ]
                == source
            )
            &
            (
                patient_df[
                    "target"
                ]
                == target
            )
            &
            (
                patient_df[
                    "seed"
                ]
                == seed
            )
        ]


        source_df = (
            job_df[
                job_df[
                    "model_state"
                ]
                == "source_only"
            ]
            .sort_values(
                "row_id"
            )
        )


        source_matrix = (
            source_df[
                shap_columns
            ]
            .to_numpy(
                dtype=float
            )
        )


        source_importance = (
            np.mean(
                np.abs(
                    source_matrix
                ),
                axis=0,
            )
        )


        source_top = set(
            np.argsort(
                source_importance
            )[
                -TOP_K:
            ]
        )


        for method in METHODS:

            adapted_df = (
                job_df[
                    job_df[
                        "model_state"
                    ]
                    == method
                ]
                .sort_values(
                    "row_id"
                )
            )


            if not np.array_equal(
                source_df[
                    "row_id"
                ].to_numpy(),
                adapted_df[
                    "row_id"
                ].to_numpy(),
            ):

                raise RuntimeError(
                    "Patient order mismatch in "
                    "stability analysis."
                )


            adapted_matrix = (
                adapted_df[
                    shap_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )


            adapted_importance = (
                np.mean(
                    np.abs(
                        adapted_matrix
                    ),
                    axis=0,
                )
            )


            adapted_top = set(
                np.argsort(
                    adapted_importance
                )[
                    -TOP_K:
                ]
            )


            rank_correlation = (
                pd.Series(
                    source_importance
                )
                .corr(
                    pd.Series(
                        adapted_importance
                    ),
                    method="spearman",
                )
            )


            patient_difference = (
                adapted_matrix
                -
                source_matrix
            )


            patient_l1 = (
                np.abs(
                    patient_difference
                )
                .sum(
                    axis=1
                )
            )


            patient_l2 = (
                np.linalg.norm(
                    patient_difference,
                    axis=1,
                )
            )


            cosine = (
                cosine_similarity_rows(
                    source_matrix,
                    adapted_matrix,
                )
            )


            topk_overlap = (
                patient_topk_overlap(
                    source_matrix,
                    adapted_matrix,
                    TOP_K,
                )
            )


            probability_shift = (
                np.abs(
                    adapted_df[
                        "probability"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                    -
                    source_df[
                        "probability"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )
            )


            global_l1 = float(
                np.sum(
                    np.abs(
                        adapted_importance
                        -
                        source_importance
                    )
                )
            )


            source_total = float(
                np.sum(
                    source_importance
                )
            )


            relative_global_l1 = (
                global_l1
                /
                source_total
                if source_total > 0
                else np.nan
            )


            stability_rows.append(
                {
                    "source":
                        source,

                    "target":
                        target,

                    "seed":
                        seed,

                    "method":
                        method,

                    "global_importance_spearman":
                        float(
                            rank_correlation
                        )
                        if pd.notna(
                            rank_correlation
                        )
                        else np.nan,

                    "global_top3_overlap_fraction":
                        float(
                            len(
                                source_top
                                &
                                adapted_top
                            )
                            /
                            TOP_K
                        ),

                    "global_top3_jaccard":
                        float(
                            len(
                                source_top
                                &
                                adapted_top
                            )
                            /
                            len(
                                source_top
                                |
                                adapted_top
                            )
                        ),

                    "global_importance_l1_drift":
                        global_l1,

                    "global_importance_relative_l1_drift":
                        relative_global_l1,

                    "patient_l1_drift_mean":
                        float(
                            patient_l1.mean()
                        ),

                    "patient_l1_drift_median":
                        float(
                            np.median(
                                patient_l1
                            )
                        ),

                    "patient_l2_drift_mean":
                        float(
                            patient_l2.mean()
                        ),

                    "patient_cosine_similarity_mean":
                        float(
                            cosine.mean()
                        ),

                    "patient_cosine_similarity_median":
                        float(
                            np.median(
                                cosine
                            )
                        ),

                    "patient_top3_overlap_mean":
                        float(
                            topk_overlap.mean()
                        ),

                    "mean_absolute_probability_shift":
                        float(
                            probability_shift.mean()
                        ),
                }
            )


    stability_df = pd.DataFrame(
        stability_rows
    )


    # --------------------------------------------------------
    # Across-seed descriptive summary
    # --------------------------------------------------------

    metrics = [
        "global_importance_spearman",
        "global_top3_overlap_fraction",
        "global_importance_relative_l1_drift",
        "patient_l1_drift_mean",
        "patient_l2_drift_mean",
        "patient_cosine_similarity_mean",
        "patient_top3_overlap_mean",
        "mean_absolute_probability_shift",
    ]


    summary_parts = []


    for metric in metrics:

        temp = (
            stability_df.groupby(
                [
                    "source",
                    "target",
                    "method",
                ]
            )[
                metric
            ]
            .agg(
                [
                    "mean",
                    "std",
                    "median",
                ]
            )
            .reset_index()
        )


        temp[
            "metric"
        ] = metric


        temp = temp.rename(
            columns={
                "mean":
                    "value_mean",

                "std":
                    "value_sd",

                "median":
                    "value_median",
            }
        )


        summary_parts.append(
            temp
        )


    direction_summary_df = pd.concat(
        summary_parts,
        ignore_index=True,
    )


    # --------------------------------------------------------
    # Write final XAI artifacts
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    patient_df.to_csv(
        OUTPUT_ROOT
        / "patient_shap_values.csv.gz",
        index=False,
        compression="gzip",
    )


    fold_df.to_csv(
        OUTPUT_ROOT
        / "fold_xai_diagnostics.csv",
        index=False,
    )


    verification_df.to_csv(
        OUTPUT_ROOT
        / "prediction_reproduction.csv",
        index=False,
    )


    importance_df.to_csv(
        OUTPUT_ROOT
        / "global_feature_importance.csv",
        index=False,
    )


    stability_df.to_csv(
        OUTPUT_ROOT
        / "explanation_stability_seed.csv",
        index=False,
    )


    direction_summary_df.to_csv(
        OUTPUT_ROOT
        / "explanation_stability_direction_summary.csv",
        index=False,
    )


    print(
        "\n"
        + "=" * 110
    )

    print(
        "XAI SUMMARY BUILT"
    )

    print(
        "=" * 110
    )


    print(
        "Patient SHAP rows:",
        len(
            patient_df
        )
    )

    print(
        "Global importance rows:",
        len(
            importance_df
        )
    )

    print(
        "Stability rows:",
        len(
            stability_df
        )
    )

    print(
        "Prediction checks:",
        len(
            verification_df
        )
    )

    print(
        "Worst prediction reproduction diff:",
        f"{verification_df['max_probability_difference'].max():.12f}"
    )

    print(
        "Worst DeepSHAP additivity error:",
        f"{fold_df['max_additivity_error'].max():.12f}"
    )


# ============================================================
# CLI
# ============================================================

def parse_seed(
    value,
):

    if value == "all":

        return list(
            canonical.SEEDS
        )


    seed = int(
        value
    )


    if (
        seed
        not in canonical.SEEDS
    ):

        raise ValueError(
            f"Seed {seed} not in canonical SEEDS."
        )


    return [
        seed
    ]


def resolve_device(
    value,
):

    if value == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


    return torch.device(
        value
    )


def main():

    if (
        shap.__version__
        != REQUIRED_SHAP_VERSION
    ):

        raise RuntimeError(
            "Frozen XAI environment expects "
            f"SHAP {REQUIRED_SHAP_VERSION}, "
            f"found {shap.__version__}"
        )


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--source",
        default="all",
    )


    parser.add_argument(
        "--target",
        default="all",
    )


    parser.add_argument(
        "--seed",
        default="all",
    )


    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
    )


    parser.add_argument(
        "--overwrite",
        action="store_true",
    )


    args = parser.parse_args()


    domains = list(
        canonical.DOMAINS.keys()
    )


    sources = (
        domains
        if args.source == "all"
        else [
            args.source
        ]
    )


    targets = (
        domains
        if args.target == "all"
        else [
            args.target
        ]
    )


    for domain in (
        sources
        +
        targets
    ):

        if (
            domain
            not in domains
        ):

            raise ValueError(
                f"Unknown domain: {domain}"
            )


    seeds = parse_seed(
        args.seed
    )


    device = resolve_device(
        args.device
    )


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    metadata = {
        "xai_version":
            XAI_VERSION,

        "shap_version":
            shap.__version__,

        "method":
            "DeepExplainer",

        "explanation_scale":
            "model logit",

        "background_source":
            "unlabeled target adaptation fold",

        "background_size":
            BACKGROUND_SIZE,

        "top_k":
            TOP_K,

        "target_labels_used_for_explanation":
            False,

        "source_raw_data_used":
            False,

        "raw_features":
            list(
                canonical.CORE8
            ),

        "models":
            MODEL_STATES,
    }


    with open(
        OUTPUT_ROOT
        / "xai_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


    print(
        "\nDevice:",
        device
    )

    print(
        "SHAP:",
        shap.__version__
    )

    print(
        "XAI version:",
        XAI_VERSION
    )

    print(
        "Background size:",
        BACKGROUND_SIZE
    )


    for source in sources:

        for target in targets:

            if (
                source
                == target
            ):

                continue


            for seed in seeds:

                process_job(
                    source=
                        source,

                    target=
                        target,

                    seed=
                        seed,

                    device=
                        device,

                    overwrite=
                        args.overwrite,
                )


    build_summaries()


    print(
        "\n"
        + "=" * 110
    )

    print(
        "SINGLE-HEAD XAI COMPLETE"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()