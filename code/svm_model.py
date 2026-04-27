"""
svm_dementia_benchmark.py
=========================
SVM benchmark model for dementia risk prediction from MIMIC-IV EHR data.
Used as a classical ML baseline alongside Transformer and CNN deep learning models.

Key design principle:
    SVMs require a fixed-length feature vector per patient. Unlike sequence models,
    they cannot process raw time series directly. This script aggregates temporal
    signals (labs, vitals, OMR) into per-patient summary statistics, then combines
    them with static demographic and comorbidity features into a single flat vector.

Feature groups:
    1. Static     — age, sex, number of admissions, insurance type
    2. Comorbidity — Elixhauser index computed from ICD codes
    3. OMR stats  — mean/std/trend of BP, weight, BMI across outpatient visits
    4. Lab stats  — mean/std/min/max of key labs (BMP, CBC, lipids)
    5. Vital stats — mean/std of HR, SpO2, RR, temperature from chartevents
                     (aggregated; only loaded if --use_chartevents flag is set,
                     as the file is very large)
"""

import argparse
import os
import warnings
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    brier_score_loss, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import loguniform, uniform
from scipy import stats

warnings.filterwarnings("ignore")


# ICD-9 dementia codes
DEMENTIA_ICD9 = [
    "290", "2900", "29010", "29011", "29012", "29013",
    "29020", "29021", "2903", "29040", "29041", "29042", "29043",
    "2940", "29410", "29411", "2948",
    "3310", "3311", "3312"
]

# ICD-10 dementia codes
DEMENTIA_ICD10 = [
    "F00", "F000", "F001", "F002", "F009",
    "F01", "F010", "F011", "F012", "F018", "F019",
    "F02", "F020", "F021", "F022", "F023", "F028",
    "F03", "F030", "F031", "F032", "F038", "F039",
    "G30", "G300", "G301", "G308", "G309",
    "G31", "G310", "G311", "G312", "G3109", "G3183", "G3184", "G3185",
    "G311", "G312", "G318"
]




def assemble_features(
    labels:         pd.DataFrame,
    vital_feat:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all feature tables on subject_id and attach dementia labels.
    Missing values are filled with column medians (computed on train set later).
    """
    df = labels.copy()
    if vital_feat is not None and len(vital_feat.columns) > 1:
        df = df.merge(vital_feat, on="subject_id", how="left")

    print(f"\n  Final feature matrix: {df.shape[0]:,} patients × {df.shape[1]-2} features")
    print(f"  Missing values: {df.isnull().sum().sum():,} cells ({100*df.isnull().mean().mean():.1f}% sparse)")
    return df



def bootstrap_ci(y_true, y_prob, metric_fn, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for a scalar metric."""
    rng = np.random.default_rng(42)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            pass
    lo = np.percentile(scores, 100 * (1 - ci) / 2)
    hi = np.percentile(scores, 100 * (1 + ci) / 2)
    return np.mean(scores), lo, hi



def plot_results(results_list: list, y_true_dict: dict, y_prob_dict: dict, output_dir: str):
    """Plot ROC curves, PR curves, and confusion matrices for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"SVM": "#0D9488", "Logistic Regression": "#1B3A5C", "SVM (Tuned)": "#14B8A6"}

    # ROC
    ax = axes[0]
    for name in y_prob_dict:
        fpr, tpr, _ = roc_curve(y_true_dict[name], y_prob_dict[name])
        auroc = roc_auc_score(y_true_dict[name], y_prob_dict[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auroc:.3f})", color=colors.get(name, "gray"), lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Test Set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Precision-Recall
    ax = axes[1]
    for name in y_prob_dict:
        prec_vals, rec_vals, _ = precision_recall_curve(y_true_dict[name], y_prob_dict[name])
        auprc = average_precision_score(y_true_dict[name], y_prob_dict[name])
        ax.plot(rec_vals, prec_vals, label=f"{name} (AP={auprc:.3f})", color=colors.get(name, "gray"), lw=2)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Test Set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.suptitle("Dementia Risk Prediction — SVM Benchmark Results", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "svm_benchmark_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plots saved to: {plot_path}")

    results_df = pd.DataFrame(results_list)
    table_path = os.path.join(output_dir, "svm_benchmark_results.csv")
    results_df.to_csv(table_path, index=False)
    print(f"  Results table saved to: {table_path}")


def build_svm(num_features=6, timestep=48):
        base_svm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                SVC(class_weight="balanced", random_state=42, probability=False),
                cv=3, method="isotonic"
            ))
        ])


        cv = StratifiedKFold(n_splits=5, shuffle=False)   # maintain temporal order
        search = RandomizedSearchCV(
            base_svm, num_features, n_iter=args.n_iter,
            scoring="roc_auc", cv=cv, n_jobs=-1,
            random_state=42, verbose=1
        )

        return base_svm
