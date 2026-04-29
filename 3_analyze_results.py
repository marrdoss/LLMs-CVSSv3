"""
Script 3: Analyze CVE Evaluation Results
==========================================
Full quantitative + qualitative analysis:

Quantitative (score regression):
  - MAE  (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - Within-0.5 accuracy
  - Within-1.0 accuracy
  - Overestimation / Underestimation analysis

Qualitative (severity classification):
  - Accuracy
  - Macro F1
  - Per-class Precision / Recall / F1
  - Confusion matrix
  - Boundary crossing errors

Consistency:
  - Score consistency rate (runs within ±1.0 of each other)

Requirements:
    pip install pandas scikit-learn matplotlib seaborn numpy

Usage:
    python 3_analyze_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────
INPUT_FILE   = "cve_results.csv"
REPORT_FILE  = "analysis_report.txt"
SUMMARY_FILE = "analysis_summary.csv"

SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]
MODELS         = ["llama3.1", "mistral"]
PROMPT_TYPES   = ["zero_shot", "few_shot"]

# CVSS v3.1 severity boundaries
SEVERITY_RANGES = {
    "Low":      (0.1, 3.9),
    "Medium":   (4.0, 6.9),
    "High":     (7.0, 8.9),
    "Critical": (9.0, 10.0),
}

CONDITIONS = [
    (m, pt) for m in MODELS for pt in PROMPT_TYPES
]

# ── Helpers ───────────────────────────────────────────────────

def label(model, pt):
    return f"{model} / {pt.replace('_', '-')}"


def normalize_label(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip().capitalize()
    return val if val in SEVERITY_ORDER else np.nan


def severity_rank(sev):
    return {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}.get(sev, 0)


def crosses_boundary(official_score, predicted_score):
    """True if predicted score falls in a different severity bin."""
    def get_bin(s):
        for sev, (lo, hi) in SEVERITY_RANGES.items():
            if lo <= s <= hi:
                return sev
        return None
    return get_bin(official_score) != get_bin(predicted_score)


# ── Quantitative Metrics ──────────────────────────────────────

def compute_regression_metrics(official_scores, predicted_scores, lbl=""):
    pairs = [
        (o, p) for o, p in zip(official_scores, predicted_scores)
        if p is not None and not np.isnan(p)
    ]
    if not pairs:
        return None

    o_arr = np.array([p[0] for p in pairs])
    p_arr = np.array([p[1] for p in pairs])
    errors = np.abs(o_arr - p_arr)

    mae  = round(float(np.mean(errors)), 3)
    rmse = round(float(np.sqrt(np.mean(errors ** 2))), 3)
    w05  = round(float(np.mean(errors <= 0.5) * 100), 1)
    w10  = round(float(np.mean(errors <= 1.0) * 100), 1)
    over = int(np.sum(p_arr > o_arr))
    under= int(np.sum(p_arr < o_arr))
    exact= int(np.sum(p_arr == o_arr))
    boundary = int(sum(crosses_boundary(o, p) for o, p in pairs))

    return {
        "label":             lbl,
        "n_valid":           len(pairs),
        "mae":               mae,
        "rmse":              rmse,
        "within_0.5_%":      w05,
        "within_1.0_%":      w10,
        "overestimated":     over,
        "underestimated":    under,
        "exact":             exact,
        "boundary_errors":   boundary,
    }


# ── Qualitative Metrics ───────────────────────────────────────

def compute_classification_metrics(y_true, y_pred, lbl=""):
    pairs = [
        (t, p) for t, p in zip(y_true, y_pred)
        if not pd.isna(t) and not pd.isna(p)
    ]
    if not pairs:
        return None

    yt = [p[0] for p in pairs]
    yp = [p[1] for p in pairs]

    acc = round(accuracy_score(yt, yp) * 100, 1)
    pr, re, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=SEVERITY_ORDER, average=None, zero_division=0
    )
    macro_f1 = round(float(np.mean(f1)) * 100, 1)

    result = {
        "label":     lbl,
        "n_valid":   len(yt),
        "accuracy":  acc,
        "macro_f1":  macro_f1,
    }
    for i, sev in enumerate(SEVERITY_ORDER):
        result[f"P_{sev}"]  = round(pr[i] * 100, 1)
        result[f"R_{sev}"]  = round(re[i] * 100, 1)
        result[f"F1_{sev}"] = round(f1[i] * 100, 1)

    return result


# ── Plots ─────────────────────────────────────────────────────

def plot_score_scatter(df, model, pt, ax):
    """Scatter: official vs predicted CVSS score."""
    score_col = f"{model}_{pt}_final_score"
    if score_col not in df.columns:
        return
    valid = df[[score_col, "Official_CVSS"]].dropna()
    ax.scatter(valid["Official_CVSS"], valid[score_col],
               alpha=0.6, edgecolors="k", linewidths=0.5, s=50)
    lims = [0, 10]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xlabel("Official CVSS Score")
    ax.set_ylabel("Predicted CVSS Score")
    ax.set_title(f"{model} / {pt.replace('_','-')}")
    ax.legend(fontsize=7)


def plot_confusion_matrix(y_true, y_pred, title, ax):
    pairs = [(t, p) for t, p in zip(y_true, y_pred)
             if not pd.isna(t) and not pd.isna(p)]
    if not pairs:
        return
    yt = [p[0] for p in pairs]
    yp = [p[1] for p in pairs]
    cm = confusion_matrix(yt, yp, labels=SEVERITY_ORDER)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SEVERITY_ORDER, yticklabels=SEVERITY_ORDER,
                ax=ax, cbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Official")
    ax.set_title(title, fontsize=9)


def plot_error_distribution(df, model, pt, ax):
    """Histogram of absolute score errors."""
    score_col = f"{model}_{pt}_final_score"
    if score_col not in df.columns:
        return
    valid = df[[score_col, "Official_CVSS"]].dropna()
    errors = (valid[score_col] - valid["Official_CVSS"]).values
    ax.hist(errors, bins=20, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction Error (Predicted − Official)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution\n{model} / {pt.replace('_','-')}", fontsize=9)


# ── Main ──────────────────────────────────────────────────────

def analyze():
    print("=" * 60)
    print("  CVE CVSS Score Prediction — Full Analysis")
    print("=" * 60)

    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"\n[ERROR] {INPUT_FILE} not found. Run 2_run_evaluation.py first.")
        return

    print(f"\n[+] Loaded {len(df)} rows")

    official_scores   = df["Official_CVSS"].tolist()
    official_labels   = [normalize_label(s) for s in df["Official_Severity"]]

    report_lines      = []
    regression_rows   = []
    classification_rows = []
    consistency_rows  = []

    report_lines += [
        "=" * 60,
        "  CVE CVSS Score Prediction — Analysis Report",
        "=" * 60,
        "",
        "── Dataset Summary ──",
        f"  Total CVEs : {len(df)}",
        f"  Mean CVSS  : {df['Official_CVSS'].mean():.2f}",
        f"  Min CVSS   : {df['Official_CVSS'].min()}",
        f"  Max CVSS   : {df['Official_CVSS'].max()}",
        "",
        "  Distribution:",
    ]
    for sev in SEVERITY_ORDER:
        n = (df["Official_Severity"] == sev).sum()
        report_lines.append(f"    {sev:10s}: {n} CVEs")

    # ── Per condition ─────────────────────────────────────────
    print("\n[+] Computing metrics per condition...")

    for model, pt in CONDITIONS:
        lbl        = label(model, pt)
        score_col  = f"{model}_{pt}_final_score"
        label_col  = f"{model}_{pt}_final_label"
        consist_col= f"{model}_{pt}_consistent"

        if score_col not in df.columns:
            print(f"  [SKIP] {lbl}")
            continue

        pred_scores = [
            float(s) if not pd.isna(s) else None
            for s in df[score_col]
        ]
        pred_labels = [normalize_label(s) for s in df[label_col]]

        # Regression metrics
        reg = compute_regression_metrics(official_scores, pred_scores, lbl)
        if reg:
            regression_rows.append(reg)

        # Classification metrics
        clf = compute_classification_metrics(official_labels, pred_labels, lbl)
        if clf:
            classification_rows.append(clf)

        # Consistency — two metrics
        score_col_c = f"{model}_{pt}_score_consistent"
        label_col_c = f"{model}_{pt}_label_consistent"

        n_score_consistent = int(df[score_col_c].sum()) if score_col_c in df.columns else 0
        n_label_consistent = int(df[label_col_c].sum()) if label_col_c in df.columns else 0

        score_consist_rate = round(n_score_consistent / len(df) * 100, 1)
        label_consist_rate = round(n_label_consistent / len(df) * 100, 1)

        consistency_rows.append({
            "label":                  lbl,
            "n_score_consistent":     n_score_consistent,
            "score_consistency_rate": score_consist_rate,
            "n_label_consistent":     n_label_consistent,
            "label_consistency_rate": label_consist_rate,
            "n_total":                len(df),
        })

        # Report section
        report_lines += [
            "",
            f"── {lbl} ──",
            "  [ Quantitative — Score Prediction ]",
            f"  MAE              : {reg['mae'] if reg else 'N/A'}",
            f"  RMSE             : {reg['rmse'] if reg else 'N/A'}",
            f"  Within ±0.5      : {reg['within_0.5_%'] if reg else 'N/A'}%",
            f"  Within ±1.0      : {reg['within_1.0_%'] if reg else 'N/A'}%",
            f"  Overestimated    : {reg['overestimated'] if reg else 'N/A'} CVEs",
            f"  Underestimated   : {reg['underestimated'] if reg else 'N/A'} CVEs",
            f"  Boundary Errors  : {reg['boundary_errors'] if reg else 'N/A'} CVEs",
            "",
            "  [ Qualitative — Severity Classification ]",
            f"  Accuracy         : {clf['accuracy'] if clf else 'N/A'}%",
            f"  Macro F1         : {clf['macro_f1'] if clf else 'N/A'}%",
            "",
            "  Per-class F1:",
        ]
        if clf:
            for sev in SEVERITY_ORDER:
                report_lines.append(
                    f"    {sev:10s}: F1={clf[f'F1_{sev}']}%  "
                    f"P={clf[f'P_{sev}']}%  R={clf[f'R_{sev}']}%"
                )

        report_lines += [
            "",
            "  [ Consistency ]",
            f"  Score Consistency : {score_consist_rate}%  "
            f"({n_score_consistent}/{len(df)} CVEs — all runs within ±1.0)",
            f"  Label Consistency : {label_consist_rate}%  "
            f"({n_label_consistent}/{len(df)} CVEs — all runs same severity label)",
            "  NOTE: Label consistency is stricter — catches boundary cases",
            "        e.g. scores 6.0 vs 7.0 are within ±1.0 but cross Medium/High boundary",
        ]

    # ── Save summary CSVs ─────────────────────────────────────
    reg_df   = pd.DataFrame(regression_rows)
    clf_df   = pd.DataFrame(classification_rows)
    cons_df  = pd.DataFrame(consistency_rows)

    reg_df.to_csv("regression_summary.csv", index=False)
    clf_df.to_csv("classification_summary.csv", index=False)
    cons_df.to_csv("consistency_summary.csv", index=False)
    print("[✓] Summary CSVs saved")

    # ── Figures ───────────────────────────────────────────────
    print("[+] Generating figures...")

    # Figure 1: Scatter plots (official vs predicted)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    for i, (model, pt) in enumerate(CONDITIONS):
        plot_score_scatter(df, model, pt, axes[i])
    plt.suptitle("Official vs Predicted CVSS Score", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("fig_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✓] fig_scatter.png saved")

    # Figure 2: Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (model, pt) in enumerate(CONDITIONS):
        label_col  = f"{model}_{pt}_final_label"
        pred_labels= [normalize_label(s) for s in df[label_col]] if label_col in df.columns else []
        title      = f"{model} / {pt.replace('_','-')}"
        plot_confusion_matrix(official_labels, pred_labels, title, axes[i])
    plt.suptitle("Confusion Matrices — Severity Classification", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_confusion.png", dpi=150)
    plt.close()
    print("[✓] fig_confusion.png saved")

    # Figure 3: Error distributions
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, (model, pt) in enumerate(CONDITIONS):
        plot_error_distribution(df, model, pt, axes[i])
    plt.suptitle("Prediction Error Distribution (Predicted − Official)", fontsize=13)
    plt.tight_layout()
    plt.savefig("fig_errors.png", dpi=150)
    plt.close()
    print("[✓] fig_errors.png saved")

    # Figure 4: MAE comparison bar chart
    if not reg_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
        bars = ax.bar(reg_df["label"], reg_df["mae"], color=colors[:len(reg_df)])
        ax.set_ylabel("Mean Absolute Error (MAE)")
        ax.set_title("MAE by Condition (lower is better)")
        for bar, val in zip(bars, reg_df["mae"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig("fig_mae.png", dpi=150)
        plt.close()
        print("[✓] fig_mae.png saved")

    # Figure 5: Accuracy + both consistency metrics
    if not clf_df.empty and not cons_df.empty:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

        bars1 = ax1.bar(clf_df["label"], clf_df["accuracy"], color=colors[:len(clf_df)])
        ax1.set_ylabel("Accuracy (%)"); ax1.set_ylim(0, 100)
        ax1.set_title("Severity Classification Accuracy")
        ax1.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars1, clf_df["accuracy"]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val}%", ha="center", fontsize=8)

        bars2 = ax2.bar(cons_df["label"], cons_df["score_consistency_rate"],
                        color=colors[:len(cons_df)])
        ax2.set_ylabel("Rate (%)"); ax2.set_ylim(0, 100)
        ax2.set_title("Score Consistency\n(runs within ±1.0)")
        ax2.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars2, cons_df["score_consistency_rate"]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val}%", ha="center", fontsize=8)

        bars3 = ax3.bar(cons_df["label"], cons_df["label_consistency_rate"],
                        color=colors[:len(cons_df)])
        ax3.set_ylabel("Rate (%)"); ax3.set_ylim(0, 100)
        ax3.set_title("Label Consistency\n(all runs same severity label)")
        ax3.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars3, cons_df["label_consistency_rate"]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val}%", ha="center", fontsize=8)

        plt.tight_layout()
        plt.savefig("fig_accuracy_consistency.png", dpi=150)
        plt.close()
        print("[✓] fig_accuracy_consistency.png saved")

    # ── Write report ──────────────────────────────────────────
    report_text = "\n".join(report_lines)
    with open(REPORT_FILE, "w") as f:
        f.write(report_text)

    print(f"\n[✓] Full report saved to : {REPORT_FILE}")
    print("\n" + report_text)
    print("\n" + "=" * 60)
    print("  Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    analyze()
