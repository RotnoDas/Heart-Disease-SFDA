from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


CORE8_NUMERIC = [
    "age",
    "trestbps",
    "thalach",
    "oldpeak",
]


CORE8_CATEGORICAL = [
    "sex",
    "cp",
    "restecg",
    "exang",
]


def build_linear_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                CORE8_NUMERIC,
            ),
            (
                "categorical",
                categorical_pipeline,
                CORE8_CATEGORICAL,
            ),
        ],
        remainder="drop",
    )


def build_tree_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                CORE8_NUMERIC,
            ),
            (
                "categorical",
                categorical_pipeline,
                CORE8_CATEGORICAL,
            ),
        ],
        remainder="drop",
    )