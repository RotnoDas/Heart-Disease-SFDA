from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PERF_DIR = ROOT / "results" / "final" / "performance_tables"
XAI_DIR = ROOT / "results" / "explainability" / "xai_single_head"
OUT_DIR = ROOT / "figures" / "presentation"

SINGLE_METRICS_FILE = PERF_DIR / "single_head_seed_level_metrics.csv"
DLAR_METRICS_FILE = PERF_DIR / "dlar_lcl_seed_level_metrics.csv"
DELTA_FILE = ROOT / "results" / "cross_source" / "cross_source_multiseed_deltas.csv"
XAI_STABILITY_FILE = XAI_DIR / "explanation_stability_seed.csv"
XAI_GLOBAL_FILE = XAI_DIR / "global_feature_importance.csv"


# ============================================================
# Frozen method/domain definitions
# ============================================================

METHOD_ORDER = [
    "source_only",
    "pseudo_label_sfda",
    "entropy_sfda",
    "reliability_gated_sfda",
    "conservative_candidate",
]

METHOD_LABELS = {
    "source_only": "Source-only",
    "pseudo_label_sfda": "Pseudo-label",
    "entropy_sfda": "Entropy",
    "reliability_gated_sfda": "Reliability-gated",
    "conservative_candidate": "Conservative",
}

ADAPTED_METHODS = [
    "pseudo_label_sfda",
    "entropy_sfda",
    "reliability_gated_sfda",
    "conservative_candidate",
]

DLAR_METHOD_ORDER = [
    "dual_head_source_only",
    "dlar_lcl",
]

DLAR_LABELS = {
    "dual_head_source_only": "Dual-head source-only",
    "dlar_lcl": "DLAR-LCL",
}

DOMAINS = [
    "cleveland",
    "hungary",
    "switzerland",
    "va_long_beach",
]

DOMAIN_LABELS = {
    "cleveland": "Cleveland",
    "hungary": "Hungary",
    "switzerland": "Switzerland",
    "va_long_beach": "VA Long Beach",
}

DIRECTION_ORDER = [
    (s, t)
    for s in DOMAINS
    for t in DOMAINS
    if s != t
]

DIRECTION_LABELS = {
    (s, t): f"{DOMAIN_LABELS[s]} → {DOMAIN_LABELS[t]}"
    for s, t in DIRECTION_ORDER
}


# ============================================================
# Style
# ============================================================

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================
# Helpers
# ============================================================


def require_file(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def read_optional(path: Path):
    if not path.exists():
        print(f"[SKIP] Missing optional file: {path}")
        return None
    return pd.read_csv(path)


def save_figure(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")


def ordered_existing_methods(df, order):
    present = set(df["method"].astype(str).unique())
    return [m for m in order if m in present]


def direction_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    order = {pair: i for i, pair in enumerate(DIRECTION_ORDER)}
    out = df.copy()
    out["direction_order"] = [
        order.get((s, t), 999)
        for s, t in zip(out["source"], out["target"])
    ]
    return out.sort_values("direction_order")


def pair_balanced_summary(seed_df: pd.DataFrame, metrics, method_order):
    """
    First average seeds within each source-target direction, then summarize
    the 12 directions equally. This is descriptive and avoids patient-count
    weighting across target cohorts.
    """
    direction_means = (
        seed_df.groupby(["source", "target", "method"], as_index=False)[metrics]
        .mean()
    )

    rows = []
    for method in method_order:
        sub = direction_means[direction_means["method"] == method]
        if sub.empty:
            continue

        row = {"method": method, "n_directions": len(sub)}
        for metric in metrics:
            vals = sub[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_sd"] = float(np.nanstd(vals, ddof=1))
        rows.append(row)

    return pd.DataFrame(rows)


def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def method_colors(method_order):
    cmap = plt.get_cmap("tab10")
    return {m: cmap(i % 10) for i, m in enumerate(method_order)}


def bar_metric_panels(
    summary: pd.DataFrame,
    method_order,
    method_labels,
    metric_specs,
    title,
    stem,
):
    """metric_specs: [(column, title, direction_note), ...]"""
    n = len(metric_specs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15.5, 4.2 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    colors = method_colors(method_order)
    methods = [m for m in method_order if m in set(summary["method"])]
    x = np.arange(len(methods))

    for i, (metric, panel_title, note) in enumerate(metric_specs):
        ax = axes[i]
        means = []
        sds = []
        for method in methods:
            row = summary[summary["method"] == method].iloc[0]
            means.append(float(row[f"{metric}_mean"]))
            sds.append(float(row[f"{metric}_sd"]))

        bars = ax.bar(
            x,
            means,
            yerr=sds,
            capsize=4,
            color=[colors[m] for m in methods],
            edgecolor="black",
            linewidth=0.45,
        )
        ax.set_xticks(x, [method_labels[m] for m in methods], rotation=24, ha="right")
        ax.set_title(panel_title, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.7,
        )
        add_panel_label(ax, chr(ord("a") + i) + ")")

        for rect, value in zip(bars, means):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.005,
        "Pair-balanced descriptive mean ± SD across 12 source-target directions; each direction first averaged across 10 source seeds.",
        ha="center",
        fontsize=9.2,
    )
    save_figure(fig, stem)


# ============================================================
# Figure 1: standard classification metrics
# ============================================================


def plot_single_standard_metrics(single_df):
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]
    summary = pair_balanced_summary(single_df, metrics, METHOD_ORDER)

    specs = [
        ("accuracy", "Accuracy", "Higher is better"),
        ("balanced_accuracy", "Balanced Accuracy", "Higher is better"),
        ("precision", "Precision", "Higher is better"),
        ("recall", "Recall / Sensitivity", "Higher is better"),
        ("specificity", "Specificity", "Higher is better"),
        ("f1", "F1 Score", "Higher is better"),
    ]

    bar_metric_panels(
        summary,
        METHOD_ORDER,
        METHOD_LABELS,
        specs,
        "Standard Test Metrics — Single-head Source-Free Adaptation",
        "presentation_01_standard_classification_metrics",
    )


# ============================================================
# Figure 2: advanced discrimination + reliability metrics
# ============================================================


def plot_single_advanced_metrics(single_df):
    metrics = [
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier",
        "ece10",
        "predicted_positive_rate",
    ]
    summary = pair_balanced_summary(single_df, metrics, METHOD_ORDER)

    specs = [
        ("mcc", "Matthews Correlation Coefficient", "Higher is better"),
        ("roc_auc", "ROC-AUC", "Higher is better"),
        ("pr_auc", "PR-AUC / Average Precision", "Higher is better"),
        ("brier", "Brier Score", "Lower is better"),
        ("ece10", "ECE (10 bins)", "Lower is better"),
        ("predicted_positive_rate", "Predicted-positive Rate", "Descriptive threshold behavior"),
    ]

    bar_metric_panels(
        summary,
        METHOD_ORDER,
        METHOD_LABELS,
        specs,
        "Discrimination and Probability Reliability — Single-head Methods",
        "presentation_02_discrimination_reliability_metrics",
    )


# ============================================================
# Figure 3: AUROC negative adaptation / Brier worsening counts
# ============================================================


def coerce_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    out = series.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        bad = sorted(series[out.isna()].astype(str).unique())
        raise ValueError(f"Could not parse boolean values: {bad}")
    return out.astype(bool)


def plot_negative_adaptation_counts(delta_df):
    df = delta_df[delta_df["method"].isin(ADAPTED_METHODS)].copy()
    df["negative_adaptation_auc"] = coerce_bool(df["negative_adaptation_auc"])
    df["brier_worsened"] = coerce_bool(df["brier_worsened"])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    colors = method_colors(ADAPTED_METHODS)
    x = np.arange(len(ADAPTED_METHODS))

    for idx, (flag, title) in enumerate(
        [
            ("negative_adaptation_auc", "AUROC Negative Adaptation"),
            ("brier_worsened", "Brier Worsening"),
        ]
    ):
        ax = axes[idx]
        counts_true = []
        totals = []
        for method in ADAPTED_METHODS:
            sub = df[df["method"] == method]
            counts_true.append(int(sub[flag].sum()))
            totals.append(int(len(sub)))

        counts_false = [t - v for t, v in zip(totals, counts_true)]
        ax.bar(
            x,
            counts_true,
            color=[colors[m] for m in ADAPTED_METHODS],
            edgecolor="black",
            linewidth=0.45,
            label="Failure flag",
        )
        ax.bar(
            x,
            counts_false,
            bottom=counts_true,
            color="lightgray",
            edgecolor="black",
            linewidth=0.45,
            label="No failure flag",
        )
        ax.set_xticks(x, [METHOD_LABELS[m] for m in ADAPTED_METHODS], rotation=22, ha="right")
        ax.set_ylabel("Seed-direction evaluations")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        add_panel_label(ax, chr(ord("a") + idx) + ")")

        for xi, n_true, total in zip(x, counts_true, totals):
            pct = 100 * n_true / total if total else np.nan
            ax.text(xi, n_true + 2, f"{n_true}/{total}\n({pct:.1f}%)", ha="center", fontsize=8.5)

    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle(
        "Negative Adaptation and Probability Degradation — Descriptive Seed-level Counts",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Canonical flags: ΔAUROC < -1e-8; Brier worsening = Brier improvement < -1e-8. Counts are descriptive repeated seed-direction evaluations.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, "presentation_03_negative_adaptation_counts")


# ============================================================
# Figure 4: predicted-positive-rate heatmap (collapse view)
# ============================================================


def plot_predicted_positive_heatmap(single_df):
    direction_means = (
        single_df.groupby(["source", "target", "method"], as_index=False)["predicted_positive_rate"]
        .mean()
    )

    matrix = []
    ylabels = []
    for source, target in DIRECTION_ORDER:
        row = []
        for method in METHOD_ORDER:
            sub = direction_means[
                (direction_means["source"] == source)
                & (direction_means["target"] == target)
                & (direction_means["method"] == method)
            ]
            row.append(float(sub["predicted_positive_rate"].iloc[0]) if not sub.empty else np.nan)
        matrix.append(row)
        ylabels.append(DIRECTION_LABELS[(source, target)])

    matrix = np.asarray(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 8.3))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=24, ha="right")
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.set_title("Threshold Behavior Across Hospital Transfers", fontweight="bold")
    ax.set_xlabel("Model state")
    ax.set_ylabel("Source → Target")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if np.isfinite(val) and val < 0.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.2, color=text_color)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Predicted-positive rate at threshold 0.5")
    fig.text(
        0.5,
        0.01,
        "Values are means across 10 source seeds. Rates near 1.00 expose all-positive threshold collapse in several shifted directions.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, "presentation_04_predicted_positive_rate_heatmap")


# ============================================================
# Figure 5: mean probability shift heatmap
# ============================================================


def plot_probability_shift_heatmap(single_df):
    src = single_df[single_df["method"] == "source_only"][
        ["source", "target", "seed", "mean_probability"]
    ].rename(columns={"mean_probability": "source_mean_probability"})

    adapted = single_df[single_df["method"].isin(ADAPTED_METHODS)].copy()
    merged = adapted.merge(
        src,
        on=["source", "target", "seed"],
        how="left",
        validate="many_to_one",
    )
    merged["delta_mean_probability"] = merged["mean_probability"] - merged["source_mean_probability"]

    direction_means = (
        merged.groupby(["source", "target", "method"], as_index=False)["delta_mean_probability"]
        .mean()
    )

    matrix = []
    ylabels = []
    for source, target in DIRECTION_ORDER:
        row = []
        for method in ADAPTED_METHODS:
            sub = direction_means[
                (direction_means["source"] == source)
                & (direction_means["target"] == target)
                & (direction_means["method"] == method)
            ]
            row.append(float(sub["delta_mean_probability"].iloc[0]) if not sub.empty else np.nan)
        matrix.append(row)
        ylabels.append(DIRECTION_LABELS[(source, target)])

    matrix = np.asarray(matrix, dtype=float)
    vmax = float(np.nanmax(np.abs(matrix)))
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(9.5, 8.3))
    im = ax.imshow(matrix, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax.set_xticks(np.arange(len(ADAPTED_METHODS)), [METHOD_LABELS[m] for m in ADAPTED_METHODS], rotation=24, ha="right")
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.set_title("Adaptation-induced Mean Probability Shift", fontweight="bold")
    ax.set_xlabel("Adaptation method")
    ax.set_ylabel("Source → Target")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Adapted mean probability − source-only mean probability")
    fig.text(
        0.5,
        0.01,
        "Positive values indicate upward confidence/probability shift; negative values indicate downward shift. Means are across 10 source seeds.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, "presentation_05_mean_probability_shift_heatmap")


# ============================================================
# Figure 6: paper-like confusion matrix pages, one per method
# ============================================================


def row_normalized_cm_from_seed_metrics(sub):
    tn = float(sub["tn"].mean())
    fp = float(sub["fp"].mean())
    fn = float(sub["fn"].mean())
    tp = float(sub["tp"].mean())
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    return norm


def plot_confusion_pages(single_df):
    for method in METHOD_ORDER:
        fig, axes = plt.subplots(4, 3, figsize=(10.8, 12.0))
        axes = axes.ravel()
        image = None

        for ax, (source, target) in zip(axes, DIRECTION_ORDER):
            sub = single_df[
                (single_df["source"] == source)
                & (single_df["target"] == target)
                & (single_df["method"] == method)
            ]
            cm = row_normalized_cm_from_seed_metrics(sub)
            image = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=10)

            ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
            ax.set_yticks([0, 1], ["True 0", "True 1"])
            ax.set_title(DIRECTION_LABELS[(source, target)], fontsize=9.5)

        cbar = fig.colorbar(image, ax=axes.tolist(), fraction=0.018, pad=0.02)
        cbar.set_label("Mean row-normalized proportion")
        fig.suptitle(
            f"Cross-Hospital Confusion Matrices — {METHOD_LABELS[method]}",
            fontweight="bold",
            y=0.995,
        )
        fig.text(
            0.5,
            0.008,
            "Each matrix is computed from mean TN/FP/FN/TP across 10 source seeds; threshold = 0.5.",
            ha="center",
            fontsize=8.8,
        )
        save_figure(fig, f"presentation_06_confusion_{method}")


# ============================================================
# Figure 7: XAI stability panels
# ============================================================


def plot_xai_stability(xai_df):
    if xai_df is None:
        return

    metrics = [
        "global_importance_spearman",
        "global_importance_relative_l1_drift",
        "patient_cosine_similarity_mean",
        "patient_l1_drift_mean",
    ]

    direction_means = (
        xai_df.groupby(["source", "target", "method"], as_index=False)[metrics]
        .mean()
    )

    rows = []
    for method in ADAPTED_METHODS:
        sub = direction_means[direction_means["method"] == method]
        if sub.empty:
            continue
        row = {"method": method}
        for metric in metrics:
            vals = sub[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_sd"] = float(np.nanstd(vals, ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows)

    specs = [
        ("global_importance_spearman", "Global Rank Stability", "Higher = more stable"),
        ("global_importance_relative_l1_drift", "Relative Global L1 Drift", "Lower = more stable"),
        ("patient_cosine_similarity_mean", "Patient SHAP Cosine Similarity", "Higher = more stable"),
        ("patient_l1_drift_mean", "Patient-level L1 Drift", "Lower = more stable"),
    ]

    bar_metric_panels(
        summary,
        ADAPTED_METHODS,
        METHOD_LABELS,
        specs,
        "Explanation Stability After Source-Free Adaptation",
        "presentation_07_xai_stability",
    )


# ============================================================
# Figure 8: XAI drift vs Brier degradation, 12 directions/method
# ============================================================


def plot_xai_vs_brier(xai_df, delta_df):
    if xai_df is None:
        return

    left = (
        xai_df.groupby(["source", "target", "method"], as_index=False)[
            "global_importance_relative_l1_drift"
        ]
        .mean()
    )

    right = delta_df[delta_df["method"].isin(ADAPTED_METHODS)].copy()
    right["brier_degradation"] = -right["brier_improvement"].astype(float)
    right = (
        right.groupby(["source", "target", "method"], as_index=False)["brier_degradation"]
        .mean()
    )

    merged = left.merge(
        right,
        on=["source", "target", "method"],
        how="inner",
        validate="one_to_one",
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.2))
    axes = axes.ravel()
    colors = method_colors(ADAPTED_METHODS)

    for i, method in enumerate(ADAPTED_METHODS):
        ax = axes[i]
        sub = merged[merged["method"] == method].copy()
        x = sub["global_importance_relative_l1_drift"].to_numpy(dtype=float)
        y = sub["brier_degradation"].to_numpy(dtype=float)
        rho, p = spearmanr(x, y)

        ax.scatter(x, y, s=55, alpha=0.85, color=colors[method], edgecolor="black", linewidth=0.4)
        ax.axhline(0, linestyle="--", linewidth=1, color="gray")
        ax.set_title(METHOD_LABELS[method], fontweight="bold")
        ax.set_xlabel("Relative global SHAP L1 drift")
        ax.set_ylabel("Brier degradation (+ = worse)")
        ax.grid(alpha=0.2)
        ax.text(
            0.03,
            0.97,
            f"n={len(sub)} directions\nSpearman ρ={rho:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )
        add_panel_label(ax, chr(ord("a") + i) + ")")

    fig.suptitle(
        "Explanation Drift vs Probability-quality Degradation",
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        "Exploratory direction-level association (12 directions per method); no multiplicity correction and no causal interpretation.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, "presentation_08_xai_drift_vs_brier")


# ============================================================
# Figure 9: source-only SHAP attribution share
# ============================================================


def plot_source_shap_importance(global_df):
    if global_df is None:
        return

    df = global_df.copy()
    state_col = "model_state" if "model_state" in df.columns else None
    if state_col is not None:
        df = df[df[state_col].astype(str).str.lower().eq("source_only")].copy()

    required = {"source", "target", "seed", "feature", "mean_abs_shap"}
    missing = required - set(df.columns)
    if missing:
        print(f"[SKIP] Global SHAP file missing columns: {sorted(missing)}")
        return

    denom = df.groupby(["source", "target", "seed"])["mean_abs_shap"].transform("sum")
    df["importance_fraction"] = np.divide(
        df["mean_abs_shap"].to_numpy(dtype=float),
        denom.to_numpy(dtype=float),
        out=np.zeros(len(df), dtype=float),
        where=denom.to_numpy(dtype=float) != 0,
    )

    summary = (
        df.groupby("feature")["importance_fraction"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary = summary.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.barh(summary["feature"], summary["mean"], xerr=summary["std"], capsize=3)
    ax.set_xlabel("Normalized mean |SHAP| share")
    ax.set_title("Source-only SHAP Attribution Profile", fontweight="bold")
    ax.grid(axis="x", alpha=0.2)

    for y, mean in enumerate(summary["mean"]):
        ax.text(mean, y, f" {mean:.3f}", va="center", fontsize=8.7)

    fig.text(
        0.5,
        0.01,
        "Attribution shares are normalized within each source-target-seed explanation job; target backgrounds differ across jobs. SHAP importance is not causal importance.",
        ha="center",
        fontsize=8.8,
    )
    save_figure(fig, "presentation_09_source_only_shap_importance")


# ============================================================
# Figures 10-11: DLAR standard + advanced metrics
# ============================================================


def plot_dlar_metrics(dlar_df):
    if dlar_df is None:
        return

    standard = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]
    standard_summary = pair_balanced_summary(dlar_df, standard, DLAR_METHOD_ORDER)
    standard_specs = [
        ("accuracy", "Accuracy", "Higher is better"),
        ("balanced_accuracy", "Balanced Accuracy", "Higher is better"),
        ("precision", "Precision", "Higher is better"),
        ("recall", "Recall / Sensitivity", "Higher is better"),
        ("specificity", "Specificity", "Higher is better"),
        ("f1", "F1 Score", "Higher is better"),
    ]
    bar_metric_panels(
        standard_summary,
        DLAR_METHOD_ORDER,
        DLAR_LABELS,
        standard_specs,
        "Standard Test Metrics — DLAR-LCL Architecture-matched Comparison",
        "presentation_10_dlar_standard_metrics",
    )

    advanced = [
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier",
        "ece10",
        "predicted_positive_rate",
    ]
    advanced_summary = pair_balanced_summary(dlar_df, advanced, DLAR_METHOD_ORDER)
    advanced_specs = [
        ("mcc", "Matthews Correlation Coefficient", "Higher is better"),
        ("roc_auc", "ROC-AUC", "Higher is better"),
        ("pr_auc", "PR-AUC / Average Precision", "Higher is better"),
        ("brier", "Brier Score", "Lower is better"),
        ("ece10", "ECE (10 bins)", "Lower is better"),
        ("predicted_positive_rate", "Predicted-positive Rate", "Descriptive threshold behavior"),
    ]
    bar_metric_panels(
        advanced_summary,
        DLAR_METHOD_ORDER,
        DLAR_LABELS,
        advanced_specs,
        "Discrimination and Reliability — DLAR-LCL",
        "presentation_11_dlar_discrimination_reliability",
    )


# ============================================================
# Main
# ============================================================


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    single_df = require_file(SINGLE_METRICS_FILE, "single-head seed metrics")
    delta_df = require_file(DELTA_FILE, "single-head delta file")
    dlar_df = read_optional(DLAR_METRICS_FILE)
    xai_df = read_optional(XAI_STABILITY_FILE)
    global_df = read_optional(XAI_GLOBAL_FILE)

    print("\nGenerating presentation figures from frozen result files only.")
    print("No model training or adaptation is performed.\n")

    plot_single_standard_metrics(single_df)
    plot_single_advanced_metrics(single_df)
    plot_negative_adaptation_counts(delta_df)
    plot_predicted_positive_heatmap(single_df)
    plot_probability_shift_heatmap(single_df)
    plot_confusion_pages(single_df)

    if xai_df is not None:
        plot_xai_stability(xai_df)
        plot_xai_vs_brier(xai_df, delta_df)

    if global_df is not None:
        plot_source_shap_importance(global_df)

    if dlar_df is not None:
        plot_dlar_metrics(dlar_df)

    manifest = {
        "source_files": {
            "single_head_seed_metrics": str(SINGLE_METRICS_FILE),
            "cross_source_deltas": str(DELTA_FILE),
            "dlar_seed_metrics": str(DLAR_METRICS_FILE),
            "xai_stability": str(XAI_STABILITY_FILE),
            "xai_global_importance": str(XAI_GLOBAL_FILE),
        },
        "model_training_performed": False,
        "adaptation_performed": False,
        "purpose": "presentation-only visualization of frozen experimental results",
        "canonical_negative_adaptation_tolerance": 1e-8,
        "notes": [
            "Pair-balanced summaries first average 10 seeds within each direction, then summarize 12 directions equally.",
            "DLAR-LCL is shown only against the architecture-matched dual-head source-only reference.",
            "XAI analyses are post-hoc and exploratory; SHAP attribution is not causal importance.",
        ],
    }
    with open(OUT_DIR / "presentation_figure_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 96)
    print("PRESENTATION FIGURES COMPLETE")
    print("=" * 96)
    print("Output directory:", OUT_DIR)


if __name__ == "__main__":
    main()