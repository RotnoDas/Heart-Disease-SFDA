import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


CONTINUOUS_FEATURES = [
    "age",
    "trestbps",
    "thalach",
    "oldpeak",
]


CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "restecg",
    "exang",
]


def ks_shift_table(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_name: str,
    target_name: str,
) -> pd.DataFrame:
    """
    Two-sample Kolmogorov-Smirnov analysis
    for continuous features.

    Missing values are removed feature-wise.
    """

    rows = []

    for feature in CONTINUOUS_FEATURES:

        source_values = (
            source_df[feature]
            .dropna()
            .to_numpy()
        )

        target_values = (
            target_df[feature]
            .dropna()
            .to_numpy()
        )

        if (
            len(source_values) == 0
            or len(target_values) == 0
        ):
            statistic = np.nan
            p_value = np.nan

        else:
            result = ks_2samp(
                source_values,
                target_values,
                alternative="two-sided",
                method="auto",
            )

            statistic = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        rows.append(
            {
                "source": source_name,
                "target": target_name,
                "feature": feature,
                "ks_statistic": statistic,
                "ks_pvalue": p_value,
                "source_n": len(source_values),
                "target_n": len(target_values),
                "source_median": (
                    float(np.median(source_values))
                    if len(source_values)
                    else np.nan
                ),
                "target_median": (
                    float(np.median(target_values))
                    if len(target_values)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def categorical_js_table(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_name: str,
    target_name: str,
) -> pd.DataFrame:
    """
    Jensen-Shannon distance for categorical
    distributions.

    scipy.spatial.distance.jensenshannon returns
    a distance, not raw divergence.

    base=2 -> values lie between 0 and 1.
    """

    rows = []

    for feature in CATEGORICAL_FEATURES:

        categories = sorted(
            set(
                source_df[feature]
                .dropna()
                .unique()
            ).union(
                set(
                    target_df[feature]
                    .dropna()
                    .unique()
                )
            )
        )

        source_dist = (
            source_df[feature]
            .value_counts(
                normalize=True,
                dropna=True,
            )
            .reindex(
                categories,
                fill_value=0.0,
            )
            .to_numpy(dtype=float)
        )

        target_dist = (
            target_df[feature]
            .value_counts(
                normalize=True,
                dropna=True,
            )
            .reindex(
                categories,
                fill_value=0.0,
            )
            .to_numpy(dtype=float)
        )

        if (
            source_dist.sum() == 0
            or target_dist.sum() == 0
        ):
            js_distance = np.nan

        else:
            js_distance = float(
                jensenshannon(
                    source_dist,
                    target_dist,
                    base=2,
                )
            )

        rows.append(
            {
                "source": source_name,
                "target": target_name,
                "feature": feature,
                "js_distance": js_distance,
                "categories": ",".join(
                    map(str, categories)
                ),
            }
        )

    return pd.DataFrame(rows)


def missingness_shift_table(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    features: list[str],
    source_name: str,
    target_name: str,
) -> pd.DataFrame:

    rows = []

    for feature in features:

        source_missing = float(
            source_df[feature]
            .isna()
            .mean()
        )

        target_missing = float(
            target_df[feature]
            .isna()
            .mean()
        )

        rows.append(
            {
                "source": source_name,
                "target": target_name,
                "feature": feature,
                "source_missing_pct":
                    100 * source_missing,
                "target_missing_pct":
                    100 * target_missing,
                "absolute_difference_pct":
                    100
                    * abs(
                        target_missing
                        - source_missing
                    ),
            }
        )

    return pd.DataFrame(rows)