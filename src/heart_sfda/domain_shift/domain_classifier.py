import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


NUMERIC = [
    "age",
    "trestbps",
    "thalach",
    "oldpeak",
]


CATEGORICAL = [
    "sex",
    "cp",
    "restecg",
    "exang",
]


CORE8 = (
    NUMERIC
    + CATEGORICAL
)


def build_domain_classifier():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
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
                SimpleImputer(
                    strategy="most_frequent"
                ),
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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL,
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=5000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def domain_classifier_auc(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_name: str,
    target_name: str,
):

    source_x = (
        source_df[CORE8]
        .copy()
    )

    target_x = (
        target_df[CORE8]
        .copy()
    )

    source_x["domain_label"] = 0
    target_x["domain_label"] = 1

    combined = pd.concat(
        [
            source_x,
            target_x,
        ],
        ignore_index=True,
    )

    X = combined[CORE8]

    y = combined[
        "domain_label"
    ].to_numpy()

    model = build_domain_classifier()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    probabilities = (
        cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
        )[:, 1]
    )

    auc = roc_auc_score(
        y,
        probabilities,
    )

    return {
        "source": source_name,
        "target": target_name,
        "domain_classifier_auc":
            float(auc),
        "source_n":
            int(len(source_df)),
        "target_n":
            int(len(target_df)),
    }