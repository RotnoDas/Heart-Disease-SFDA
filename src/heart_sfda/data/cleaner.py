import numpy as np
import pandas as pd


PREDICTORS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


CORE8 = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
]


FULL13 = PREDICTORS.copy()


ALLOWED_CATEGORIES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


def clean_domain(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Standardize sentinel values and create binary target.

    IMPORTANT:
    - Does not impute missing data.
    - Does not scale data.
    - Does not remove rows.
    - Does not use target-domain labels for model training.
    """

    out = df.copy()

    report = {
        "domain": out["domain"].iloc[0],
        "n_rows": len(out),
    }

    # -------------------------------------------------
    # Known invalid / sentinel zero values
    # -------------------------------------------------

    chol_zero = out["chol"].eq(0)

    report["chol_zero_to_nan"] = int(
        chol_zero.sum()
    )

    out.loc[
        chol_zero,
        "chol",
    ] = np.nan


    bp_zero = out["trestbps"].eq(0)

    report["trestbps_zero_to_nan"] = int(
        bp_zero.sum()
    )

    out.loc[
        bp_zero,
        "trestbps",
    ] = np.nan


    # -------------------------------------------------
    # Binary outcome harmonization
    # -------------------------------------------------

    if out["num"].isna().any():
        raise ValueError(
            "Missing values found in outcome 'num'."
        )

    out["target"] = (
        out["num"] > 0
    ).astype("int8")


    # -------------------------------------------------
    # Final basic report
    # -------------------------------------------------

    report["positive_n"] = int(
        out["target"].sum()
    )

    report["negative_n"] = int(
        len(out) - out["target"].sum()
    )

    report["positive_pct"] = float(
        100 * out["target"].mean()
    )

    report["total_predictor_missing"] = int(
        out[PREDICTORS].isna().sum().sum()
    )

    return out, report


def validate_categories(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return validation warnings instead of silently
    changing unexpected values.
    """

    warnings = []

    for feature, allowed in ALLOWED_CATEGORIES.items():

        observed = set(
            df[feature]
            .dropna()
            .astype(int)
            .unique()
        )

        unexpected = observed - allowed

        if unexpected:
            warnings.append(
                f"{feature}: unexpected values "
                f"{sorted(unexpected)}"
            )

    return warnings