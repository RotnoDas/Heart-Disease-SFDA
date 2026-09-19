from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "domain_shift"

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FILES = {
    "cleveland": "cleveland.csv",
    "hungary": "hungary.csv",
    "switzerland": "switzerland.csv",
    "va_long_beach": "va_long_beach.csv",
}


FEATURES = [
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


def main():

    rows = []

    for domain, filename in FILES.items():

        df = pd.read_csv(
            DATA_DIR / filename
        )

        missing_pct = (
            df[FEATURES]
            .isna()
            .mean()
            * 100
        )

        missing_pct.name = domain

        rows.append(
            missing_pct
        )

    missing_df = pd.DataFrame(
        rows
    )

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    image = ax.imshow(
        missing_df.values,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(FEATURES))
    )

    ax.set_xticklabels(
        FEATURES,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(
        range(len(missing_df.index))
    )

    ax.set_yticklabels(
        missing_df.index
    )

    ax.set_title(
        "Feature Missingness Across Clinical Domains (%)"
    )

    for i in range(
        missing_df.shape[0]
    ):

        for j in range(
            missing_df.shape[1]
        ):

            value = missing_df.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Missing (%)"
    )

    fig.tight_layout()

    output_path = (
        FIG_DIR
        / "missingness_heatmap.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path
    )


if __name__ == "__main__":
    main()