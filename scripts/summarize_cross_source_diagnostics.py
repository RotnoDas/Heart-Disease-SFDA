from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "cross_source"
    / "cross_source_fold_diagnostics.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "cross_source"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def safe_mean(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    if values.notna().sum() == 0:
        return np.nan

    return float(values.mean())


def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "\nRows:",
        len(df)
    )

    print(
        "Methods:",
        sorted(
            df["method"]
            .unique()
        )
    )

    # --------------------------------------------------------
    # Normalize adaptation_skipped
    # --------------------------------------------------------

    if (
        "adaptation_skipped"
        in df.columns
    ):

        df[
            "adaptation_skipped_numeric"
        ] = (
            df[
                "adaptation_skipped"
            ]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": 1.0,
                    "false": 0.0,
                }
            )
        )

    else:

        df[
            "adaptation_skipped_numeric"
        ] = np.nan


    # --------------------------------------------------------
    # Build direction/method summaries
    # --------------------------------------------------------

    group_columns = [
        "source",
        "target",
        "method",
    ]


    rows = []


    for keys, group in (
        df.groupby(
            group_columns,
            dropna=False,
        )
    ):

        (
            source,
            target,
            method,
        ) = keys


        row = {
            "source":
                source,

            "target":
                target,

            "method":
                method,

            "fold_runs":
                len(group),

            "skip_rate_pct":
                100.0
                * safe_mean(
                    group[
                        "adaptation_skipped_numeric"
                    ]
                ),
        }


        optional_columns = {
            "selected_n":
                "mean_selected_n",

            "selected_per_class":
                "mean_selected_per_class",

            "coverage":
                "mean_coverage",

            "pseudo_positive_n":
                "mean_pseudo_positive_n",

            "pseudo_negative_n":
                "mean_pseudo_negative_n",

            "mean_selected_confidence":
                "mean_selected_confidence",

            "reliability_score":
                "mean_reliability_score",

            "teacher_positive_rate":
                "mean_teacher_positive_rate",

            "positive_rate_before":
                "mean_positive_rate_before",

            "positive_rate_after":
                "mean_positive_rate_after",

            "entropy_before":
                "mean_entropy_before",

            "entropy_after":
                "mean_entropy_after",

            "teacher_drift":
                "mean_teacher_drift",

            "mean_abs_teacher_drift":
                "mean_teacher_drift_v1",

            "missingness_rate":
                "mean_missingness_rate",
        }


        for (
            original,
            output,
        ) in optional_columns.items():

            if original in group.columns:

                row[output] = (
                    safe_mean(
                        group[
                            original
                        ]
                    )
                )

            else:

                row[output] = np.nan


        # ----------------------------------------------------
        # Derived pseudo-label imbalance
        # ----------------------------------------------------

        if (
            "pseudo_positive_n"
            in group.columns
            and
            "pseudo_negative_n"
            in group.columns
        ):

            positive = pd.to_numeric(
                group[
                    "pseudo_positive_n"
                ],
                errors="coerce",
            )

            negative = pd.to_numeric(
                group[
                    "pseudo_negative_n"
                ],
                errors="coerce",
            )

            total = (
                positive
                + negative
            )

            valid = (
                total > 0
            )

            if valid.any():

                row[
                    "mean_pseudo_positive_fraction"
                ] = float(
                    (
                        positive[
                            valid
                        ]
                        /
                        total[
                            valid
                        ]
                    ).mean()
                )

            else:

                row[
                    "mean_pseudo_positive_fraction"
                ] = np.nan

        else:

            row[
                "mean_pseudo_positive_fraction"
            ] = np.nan


        # ----------------------------------------------------
        # Prediction movement
        # ----------------------------------------------------

        if (
            "positive_rate_before"
            in group.columns
            and
            "positive_rate_after"
            in group.columns
        ):

            before = pd.to_numeric(
                group[
                    "positive_rate_before"
                ],
                errors="coerce",
            )

            after = pd.to_numeric(
                group[
                    "positive_rate_after"
                ],
                errors="coerce",
            )

            row[
                "mean_positive_rate_change"
            ] = float(
                (
                    after
                    - before
                )
                .mean()
            )

        else:

            row[
                "mean_positive_rate_change"
            ] = np.nan


        if (
            "entropy_before"
            in group.columns
            and
            "entropy_after"
            in group.columns
        ):

            before = pd.to_numeric(
                group[
                    "entropy_before"
                ],
                errors="coerce",
            )

            after = pd.to_numeric(
                group[
                    "entropy_after"
                ],
                errors="coerce",
            )

            row[
                "mean_entropy_change"
            ] = float(
                (
                    after
                    - before
                )
                .mean()
            )

        else:

            row[
                "mean_entropy_change"
            ] = np.nan


        rows.append(
            row
        )


    summary = pd.DataFrame(
        rows
    )


    output_path = (
        OUTPUT_DIR
        / "cross_source_diagnostic_summary.csv"
    )


    summary.to_csv(
        output_path,
        index=False,
    )


    # --------------------------------------------------------
    # Safety-gated methods
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 150
    )

    print(
        "SAFETY-GATE DIAGNOSTICS"
    )

    print(
        "=" * 150
    )


    gate_methods = summary[
        summary[
            "method"
        ].isin(
            [
                "reliability_gated_sfda",
                "conservative_candidate",
            ]
        )
    ]


    gate_columns = [
        "source",
        "target",
        "method",
        "fold_runs",
        "skip_rate_pct",
        "mean_selected_per_class",
        "mean_reliability_score",
        "mean_teacher_positive_rate",
        "mean_missingness_rate",
        "mean_positive_rate_change",
        "mean_entropy_change",
        "mean_teacher_drift",
        "mean_teacher_drift_v1",
    ]


    print(
        gate_methods[
            gate_columns
        ].to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Pseudo-label diagnostics
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 150
    )

    print(
        "PSEUDO-LABEL DIAGNOSTICS"
    )

    print(
        "=" * 150
    )


    pseudo = summary[
        summary["method"]
        == "pseudo_label_sfda"
    ]


    pseudo_columns = [
        "source",
        "target",
        "fold_runs",
        "skip_rate_pct",
        "mean_selected_n",
        "mean_coverage",
        "mean_pseudo_positive_n",
        "mean_pseudo_negative_n",
        "mean_pseudo_positive_fraction",
        "mean_selected_confidence",
    ]


    print(
        pseudo[
            pseudo_columns
        ].to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Entropy diagnostics
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 150
    )

    print(
        "ENTROPY DIAGNOSTICS"
    )

    print(
        "=" * 150
    )


    entropy = summary[
        summary["method"]
        == "entropy_sfda"
    ]


    entropy_columns = [
        "source",
        "target",
        "fold_runs",
        "mean_positive_rate_before",
        "mean_positive_rate_after",
        "mean_positive_rate_change",
        "mean_entropy_before",
        "mean_entropy_after",
        "mean_entropy_change",
    ]


    print(
        entropy[
            entropy_columns
        ].to_string(
            index=False
        )
    )


    print(
        "\nSaved:",
        output_path
    )


if __name__ == "__main__":
    main()