from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "domain_shift"


FILES = {
    "Cleveland": "cleveland.csv",
    "Hungary": "hungary.csv",
    "Switzerland": "switzerland.csv",
    "VA Long Beach": "va_long_beach.csv",
}


def main():

    prevalence = {}

    for domain, filename in FILES.items():

        df = pd.read_csv(
            DATA_DIR / filename
        )

        prevalence[domain] = (
            100 * df["target"].mean()
        )

    series = pd.Series(
        prevalence
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    series.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel(
        "Heart Disease Prevalence (%)"
    )

    ax.set_xlabel(
        "Clinical Domain"
    )

    ax.set_title(
        "Outcome Prevalence Across Domains"
    )

    ax.set_ylim(
        0,
        100
    )

    for i, value in enumerate(
        series.values
    ):

        ax.text(
            i,
            value + 2,
            f"{value:.1f}%",
            ha="center"
        )

    fig.tight_layout()

    output_path = (
        FIG_DIR
        / "target_prevalence.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved:",
        output_path
    )


if __name__ == "__main__":
    main()