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


# ICU chartevents itemids for vital signs
VITAL_ITEMIDS = {
    "heart_rate":   220045,
    "spo2":         220277,
    "resp_rate":    220210,
    "temperature":  223761,   # Fahrenheit
    "sbp":          220179,
    "dbp":          220180,
    "gcs_total":    223900,
}

def load_admissions(mimic_dir: str) -> pd.DataFrame:
    """Load admissions.csv and parse timestamps."""
    path = os.path.join(mimic_dir, "hosp", "admissions.csv")
    df = pd.read_csv(path, parse_dates=["admittime", "dischtime", "deathtime", "edregtime"])
    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400
    print(f"  Admissions loaded: {len(df):,} rows, {df['subject_id'].nunique():,} patients")
    return df


def load_patients(mimic_dir: str) -> pd.DataFrame:
    """Load patients.csv."""
    path = os.path.join(mimic_dir, "hosp", "patients.csv")
    df = pd.read_csv(path)
    print(f"  Patients loaded: {len(df):,} rows")
    return df


def load_diagnoses(mimic_dir: str) -> pd.DataFrame:
    """Load diagnoses_icd.csv."""
    path = os.path.join(mimic_dir, "hosp", "diagnoses_icd.csv")
    df = pd.read_csv(path, dtype={"icd_code": str})
    df["icd_code"] = df["icd_code"].str.strip().str.upper()
    print(f"  Diagnoses loaded: {len(df):,} rows")
    return df


def load_omr(mimic_dir: str) -> pd.DataFrame:
    """Load omr.csv.gz — outpatient measurement records."""
    path = os.path.join(mimic_dir, "hosp", "omr.csv.gz")
    df = pd.read_csv(path, compression="gzip", parse_dates=["chartdate"])
    df["result_name"] = df["result_name"].str.strip().str.lower()
    print(f"  OMR loaded: {len(df):,} rows")
    return df


def load_chartevents(mimic_dir: str, itemids: list) -> pd.DataFrame:
    path = os.path.join(mimic_dir, "icu", "chartevents.csv.gz")
    print(f"  Loading chartevents (chunked — this may take several minutes)...")
    chunks = []
    for chunk in pd.read_csv(
        path, compression="gzip",
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        chunksize=1_000_000,
        dtype={"valuenum": float}
    ):
        chunks.append(chunk[chunk["itemid"].isin(itemids)])
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Chartevents filtered: {len(df):,} rows")
    return df

def build_dementia_labels(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """
    Assign binary dementia label per patient.
    A patient is positive (label=1) if any of their ICD codes match
    the dementia code lists above.

    Returns a DataFrame with columns: [subject_id, dementia_label]
    """
    icd9_prefix_set  = set(DEMENTIA_ICD9)
    icd10_prefix_set = set(DEMENTIA_ICD10)

    def matches_dementia(code: str, version: str) -> bool:
        if not isinstance(code, str):
            return False
        code = code.upper().replace(".", "")
        prefix_set = icd9_prefix_set if str(version) == "9" else icd10_prefix_set
        return any(code.startswith(p) for p in prefix_set)

    dx = diagnoses.copy()
    dx["is_dementia"] = dx.apply(
        lambda r: matches_dementia(r["icd_code"], r.get("icd_version", "10")), axis=1
    )
    labels = (
        dx.groupby("subject_id")["is_dementia"]
        .max()
        .reset_index()
        .rename(columns={"is_dementia": "dementia_label"})
    )
    labels["dementia_label"] = labels["dementia_label"].astype(int)
    n_pos = labels["dementia_label"].sum()
    n_total = len(labels)
    print(f"  Dementia labels: {n_pos:,} positive / {n_total:,} total ({100*n_pos/n_total:.1f}% prevalence)")
    return labels

def build_static_features(admissions: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """
    Build static per-patient features from admissions + patients tables.

    Features:
        - anchor_age         : age at anchor year (proxy for current age)
        - gender_male        : binary (1 = male)
        - n_admissions       : total number of hospital admissions
        - mean_los           : mean length of stay across admissions
        - n_ed_visits        : number of admissions via ED
        - insurance_medicare : binary
        - insurance_medicaid : binary
        - days_span          : days between first and last admission (disease trajectory length)
    """
    # Per-patient admission stats
    adm_stats = admissions.groupby("subject_id").agg(
        n_admissions=("hadm_id", "count"),
        mean_los=("los_days", "mean"),
        first_admit=("admittime", "min"),
        last_admit=("admittime", "max"),
    ).reset_index()
    adm_stats["days_span"] = (adm_stats["last_admit"] - adm_stats["first_admit"]).dt.days

    # ED visits
    ed_visits = (
        admissions[admissions["admission_type"].str.contains("EMERGENCY|EW EMER", na=False, case=False)]
        .groupby("subject_id")
        .size()
        .reset_index(name="n_ed_visits")
    )
    # Demographics from patients
    demo = patients[["subject_id", "anchor_age", "gender"]].copy()
    demo["gender_male"] = (demo["gender"].str.upper() == "M").astype(int)

    # Merge everything
    feat = (
        demo.drop(columns=["gender"])
        .merge(adm_stats[["subject_id", "n_admissions", "mean_los", "days_span"]], on="subject_id", how="left")
        .merge(ed_visits, on="subject_id", how="left")
    )
    feat["n_ed_visits"] = feat["n_ed_visits"].fillna(0)
    feat["days_span"] = feat["days_span"].fillna(0)

    print(f"  Static features: {len(feat):,} patients, {feat.shape[1]-1} features")
    return feat


def build_omr_features(omr: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate outpatient measurements into per-patient summary statistics.

    For each measurement type (BP systolic, BP diastolic, weight, BMI, eGFR):
        - mean, std, min, max
        - linear trend slope (captures deterioration over time)
        - count (number of readings)
    """
    omr = omr.dropna(subset=["result_value"]).copy()

    # Parse BP — stored as "120/80" string
    bp_mask = omr["result_name"].str.contains("blood pressure", na=False)
    bp = omr[bp_mask].copy()
    if len(bp) > 0:
        bp_split = bp["result_value"].str.extract(r"(\d+)/(\d+)").astype(float)
        bp["sbp"] = bp_split[0]
        bp["dbp"] = bp_split[1]

    # Numeric measurements
    numeric_mask = omr["result_name"].isin(["weight (lbs)", "weight (kg)", "bmi (kg/m2)", "height (inches)"])
    num = omr[numeric_mask].copy()
    num["result_value"] = pd.to_numeric(num["result_value"], errors="coerce")

    def agg_measurement(df, value_col, name):
        if len(df) == 0:
            return pd.DataFrame(columns=["subject_id", f"{name}_mean", f"{name}_std",
                                         f"{name}_min", f"{name}_max", f"{name}_count", f"{name}_trend"])
        stats = df.groupby("subject_id")[value_col].agg(
            **{f"{name}_mean": "mean", f"{name}_std": "std",
               f"{name}_min": "min",  f"{name}_max": "max",
               f"{name}_count": "count"}
        ).reset_index()

        # Compute linear trend per patient (slope of value over time)
        trends = []
        df_sorted = df.sort_values("chartdate")
        for subj, grp in df_sorted.groupby("subject_id"):
            if len(grp) >= 3:
                x = (grp["chartdate"] - grp["chartdate"].min()).dt.days.values.astype(float)
                y = grp[value_col].dropna().values
                if len(y) == len(x) and np.std(x) > 0:
                    slope, *_ = np.polyfit(x, y, 1)
                else:
                    slope = 0.0
            else:
                slope = 0.0
            trends.append({"subject_id": subj, f"{name}_trend": slope})
        trend_df = pd.DataFrame(trends)
        return stats.merge(trend_df, on="subject_id", how="left")

    omr_features = None

    # Blood pressure
    if len(bp) > 0:
        sbp_feat = agg_measurement(bp, "sbp", "omr_sbp")
        dbp_feat = agg_measurement(bp, "dbp", "omr_dbp")
        omr_features = sbp_feat.merge(dbp_feat, on="subject_id", how="outer") if omr_features is None else omr_features.merge(sbp_feat, on="subject_id", how="outer").merge(dbp_feat, on="subject_id", how="outer")

    # Weight, BMI
    for name_str, col_name in [("weight (lbs)", "omr_weight"), ("bmi (kg/m2)", "omr_bmi")]:
        subset = num[num["result_name"] == name_str]
        feat = agg_measurement(subset, "result_value", col_name)
        if omr_features is None:
            omr_features = feat
        else:
            omr_features = omr_features.merge(feat, on="subject_id", how="outer")

    if omr_features is None:
        print("  OMR: no usable data found, skipping OMR features")
        return pd.DataFrame(columns=["subject_id"])

    print(f"  OMR features: {len(omr_features):,} patients, {omr_features.shape[1]-1} features")
    return omr_features

    """
    Aggregate lab values into per-patient summary statistics.

    For each lab (sodium, potassium, creatinine, etc.):
        - mean, std, min, max, last value, count
        - trend slope (rate of change)
        - flag: any critically abnormal reading (outside reference range)
    """
    # Reference ranges for critical value flagging
    CRITICAL_RANGES = {
        50983: (120, 160),   # sodium
        50971: (2.5, 6.5),   # potassium
        50912: (0.0, 10.0),  # creatinine
        51222: (7.0, 99.0),  # hemoglobin
        50931: (50, 500),    # glucose
    }

    id_to_name = {v: k for k, v in LAB_ITEMIDS.items()}
    results = []

    for itemid, lab_name in id_to_name.items():
        subset = labevents[labevents["itemid"] == itemid][["subject_id", "charttime", "valuenum"]].dropna()
        if len(subset) == 0:
            continue

        agg = subset.groupby("subject_id")["valuenum"].agg(
            mean="mean", std="std", min="min", max="max", count="count"
        ).reset_index()
        agg.columns = ["subject_id"] + [f"lab_{lab_name}_{s}" for s in ["mean", "std", "min", "max", "count"]]

        # Last recorded value (most recent)
        last_val = (
            subset.sort_values("charttime")
            .groupby("subject_id")["valuenum"]
            .last()
            .reset_index()
            .rename(columns={"valuenum": f"lab_{lab_name}_last"})
        )

        # Critical value flag
        if itemid in CRITICAL_RANGES:
            lo, hi = CRITICAL_RANGES[itemid]
            critical = (
                subset.groupby("subject_id")["valuenum"]
                .apply(lambda x: int(((x < lo) | (x > hi)).any()))
                .reset_index()
                .rename(columns={"valuenum": f"lab_{lab_name}_critical_flag"})
            )
            agg = agg.merge(last_val, on="subject_id", how="left").merge(critical, on="subject_id", how="left")
        else:
            agg = agg.merge(last_val, on="subject_id", how="left")

        # Trend slope
        trends = []
        for subj, grp in subset.sort_values("charttime").groupby("subject_id"):
            if len(grp) >= 3:
                x = (grp["charttime"] - grp["charttime"].min()).dt.total_seconds().values / 86400
                y = grp["valuenum"].values
                if np.std(x) > 0:
                    slope, *_ = np.polyfit(x, y, 1)
                else:
                    slope = 0.0
            else:
                slope = 0.0
            trends.append({"subject_id": subj, f"lab_{lab_name}_trend": slope})
        trend_df = pd.DataFrame(trends)
        agg = agg.merge(trend_df, on="subject_id", how="left")

        results.append(agg)

    if not results:
        print("  Labs: no lab data found")
        return pd.DataFrame(columns=["subject_id"])

    lab_features = results[0]
    for r in results[1:]:
        lab_features = lab_features.merge(r, on="subject_id", how="outer")

    print(f"  Lab features: {len(lab_features):,} patients, {lab_features.shape[1]-1} features")
    return lab_features

def build_vital_features(chartevents: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ICU vital signs into per-patient summary statistics.
    Same aggregation strategy as labs: mean, std, min, max, last, count, trend.
    """
    id_to_name = {v: k for k, v in VITAL_ITEMIDS.items()}
    results = []

    for itemid, vital_name in id_to_name.items():
        subset = chartevents[chartevents["itemid"] == itemid][["subject_id", "charttime", "valuenum"]].dropna()
        if len(subset) == 0:
            continue

        agg = subset.groupby("subject_id")["valuenum"].agg(
            mean="mean", std="std", min="min", max="max", count="count"
        ).reset_index()
        agg.columns = ["subject_id"] + [f"vital_{vital_name}_{s}" for s in ["mean", "std", "min", "max", "count"]]

        last_val = (
            subset.sort_values("charttime")
            .groupby("subject_id")["valuenum"]
            .last()
            .reset_index()
            .rename(columns={"valuenum": f"vital_{vital_name}_last"})
        )
        agg = agg.merge(last_val, on="subject_id", how="left")
        results.append(agg)

    if not results:
        print("  Vitals: no chartevents data found")
        return pd.DataFrame(columns=["subject_id"])

    vital_features = results[0]
    for r in results[1:]:
        vital_features = vital_features.merge(r, on="subject_id", how="outer")

    print(f"  Vital features: {len(vital_features):,} patients, {vital_features.shape[1]-1} features")
    return vital_features


def assemble_features(
    labels:         pd.DataFrame,
    static_feat:    pd.DataFrame,
    omr_feat:       pd.DataFrame,
    vital_feat:     pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merge all feature tables on subject_id and attach dementia labels.
    Missing values are filled with column medians (computed on train set later).
    """
    df = labels.copy()
    df = df.merge(static_feat, on="subject_id", how="left")
    df = df.merge(omr_feat,    on="subject_id", how="left")
    if vital_feat is not None and len(vital_feat.columns) > 1:
        df = df.merge(vital_feat, on="subject_id", how="left")

    print(f"\n  Final feature matrix: {df.shape[0]:,} patients × {df.shape[1]-2} features")
    print(f"  Missing values: {df.isnull().sum().sum():,} cells ({100*df.isnull().mean().mean():.1f}% sparse)")
    return df

def train_test_split(df: pd.DataFrame, admissions: pd.DataFrame,
                   train_frac=0.70, val_frac=0.15):

    admissions = admissions.groupby("subject_id")

    n = len(admissions)
    train_cut = int(n * train_frac)
    val_cut   = int(n * (train_frac + val_frac))

    train_subj = set(admissions.iloc[:train_cut]["subject_id"])
    val_subj   = set(admissions.iloc[train_cut:val_cut]["subject_id"])
    test_subj  = set(admissions.iloc[val_cut:]["subject_id"])

    train_df = df[df["subject_id"].isin(train_subj)].copy()
    val_df   = df[df["subject_id"].isin(val_subj)].copy()
    test_df  = df[df["subject_id"].isin(test_subj)].copy()

    print(f"\n  Temporal split —"
          f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        prev = 100 * split_df["dementia_label"].mean()
        print(f"    {split_name} dementia prevalence: {prev:.1f}%")

    return train_df, val_df, test_df



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


def evaluate_model(name: str, y_true, y_prob, y_pred, threshold=0.5):
    """Print a formatted evaluation report."""
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)

    # Bootstrap CIs
    _, auroc_lo, auroc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score)
    _, auprc_lo, auprc_hi = bootstrap_ci(y_true, y_prob, average_precision_score)

    print(f"\n  ── {name} ──")
    print(f"  AUROC         : {auroc:.4f}  (95% CI: {auroc_lo:.4f}–{auroc_hi:.4f})")
    print(f"  AUPRC         : {auprc:.4f}  (95% CI: {auprc_lo:.4f}–{auprc_hi:.4f})")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall/Sens.  : {rec:.4f}")
    print(f"  F1 Score      : {f1:.4f}")
    print(f"  Brier Score   : {brier:.4f}  (lower = better calibration)")

    return {
        "model": name, "auroc": auroc, "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
        "auprc": auprc, "auprc_lo": auprc_lo, "auprc_hi": auprc_hi,
        "precision": prec, "recall": rec, "f1": f1, "brier": brier
    }


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


def main(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    print("\n" + "="*70)
    print("  DEMENTIA RISK PREDICTION — SVM BENCHMARK")
    print("="*70)

    print("\n[1/6] Loading MIMIC-IV data...")
    admissions = load_admissions(args.mimic_dir)
    patients   = load_patients(args.mimic_dir)
    diagnoses  = load_diagnoses(args.mimic_dir)
    omr        = load_omr(args.mimic_dir)

    chartevents = None
    if args.use_chartevents:
        chartevents = load_chartevents(args.mimic_dir, list(VITAL_ITEMIDS.values()))

    print("\n[2/6] Building labels and comorbidity features...")
    labels     = build_dementia_labels(diagnoses)

    print("\n[3/6] Engineering features...")
    static_feat = build_static_features(admissions, patients)
    omr_feat    = build_omr_features(omr)
    vital_feat  = build_vital_features(chartevents) if chartevents is not None else None

    print("\n[4/6] Assembling feature matrix and splitting data...")
    master = assemble_features(labels, static_feat, omr_feat, vital_feat)

    train_df, val_df, test_df = train_test_split(master, admissions)

    # Prepare arrays — drop non-feature columns
    drop_cols = ["subject_id", "dementia_label"]
    feature_cols = [c for c in master.columns if c not in drop_cols]

    X_train = train_df[feature_cols].values
    y_train = train_df["dementia_label"].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df["dementia_label"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["dementia_label"].values

    # Impute missing values with training set medians
    print(f"\n  Imputing {len(feature_cols)} features with training set medians...")
    train_medians = np.nanmedian(X_train, axis=0)
    for arr in [X_train, X_val, X_test]:
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(train_medians, inds[1])

    # Save feature names for interpretability
    pd.Series(feature_cols).to_csv(os.path.join(output_dir, "feature_names.csv"), index=False, header=False)

    print("\n[5/6] Training models...")
    results_list = []
    y_true_dict  = {}
    y_prob_dict  = {}

    print("\n  Training Logistic Regression (baseline)...")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1.0,
            solver="lbfgs", random_state=42
        ))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    res = evaluate_model("Logistic Regression", y_test, lr_prob, lr_pred)
    results_list.append(res)
    y_true_dict["Logistic Regression"] = y_test
    y_prob_dict["Logistic Regression"] = lr_prob
    joblib.dump(lr_pipeline, os.path.join(output_dir, "lr_baseline.joblib"))


    print("\n  Training SVM (RBF kernel, default C=1.0)...")
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", random_state=42),
            cv=3, method="isotonic"
        ))
    ])

    svm_pipeline.fit(X_train, y_train)
    svm_prob = svm_pipeline.predict_proba(X_test)[:, 1]
    svm_pred = (svm_prob >= 0.5).astype(int)
    res = evaluate_model("SVM", y_test, svm_prob, svm_pred)
    results_list.append(res)
    y_true_dict["SVM"] = y_test
    y_prob_dict["SVM"] = svm_prob
    joblib.dump(svm_pipeline, os.path.join(output_dir, "svm_default.joblib"))

    if args.tune:
        print("\n  Running randomized hyperparameter search for SVM...")
        print("  (This may take 20-60 minutes depending on dataset size)")

        param_dist = {
            "clf__estimator__C":     loguniform(1e-2, 1e3),
            "clf__estimator__gamma": loguniform(1e-5, 1e0),
            "clf__estimator__kernel": ["rbf", "poly", "sigmoid"],
        }

        base_svm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                SVC(class_weight="balanced", random_state=42, probability=False),
                cv=3, method="isotonic"
            ))
        ])

        # Use val set for scoring during search
        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])

        cv = StratifiedKFold(n_splits=5, shuffle=False)   # maintain temporal order
        search = RandomizedSearchCV(
            base_svm, param_dist, n_iter=args.n_iter,
            scoring="roc_auc", cv=cv, n_jobs=-1,
            random_state=42, verbose=1
        )
        search.fit(X_trainval, y_trainval)

        best_svm = search.best_estimator_
        print(f"\n  Best hyperparameters: {search.best_params_}")
        print(f"  Best CV AUROC: {search.best_score_:.4f}")

        svm_tuned_prob = best_svm.predict_proba(X_test)[:, 1]
        svm_tuned_pred = (svm_tuned_prob >= 0.5).astype(int)
        res = evaluate_model("SVM (Tuned)", y_test, svm_tuned_prob, svm_tuned_pred)
        results_list.append(res)
        y_true_dict["SVM (Tuned)"] = y_test
        y_prob_dict["SVM (Tuned)"] = svm_tuned_prob
        joblib.dump(best_svm, os.path.join(output_dir, "svm_tuned.joblib"))

        # Save search results
        search_df = pd.DataFrame(search.cv_results_)
        search_df.to_csv(os.path.join(output_dir, "svm_hp_search_results.csv"), index=False)

   
    print("\n[6/6] Saving plots and results...")
    plot_results(results_list, y_true_dict, y_prob_dict, output_dir)

    # Final summary table
    results_df = pd.DataFrame(results_list)
    print("\n" + "="*70)
    print("  FINAL RESULTS SUMMARY")
    print("="*70)
    print(results_df[["model", "auroc", "auprc", "precision", "recall", "f1", "brier"]].to_string(index=False))

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
    print(f"  Models saved to: {output_dir}")
    print("="*70 + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Dementia Benchmark — MIMIC-IV")

    parser.add_argument(
        "--mimic_dir", type=str, required=True,
        help="Root directory of MIMIC-IV (should contain hosp/ and icu/ subdirs)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./svm_results",
        help="Directory for saving models, plots, and result CSVs (default: ./svm_results)"
    )
    parser.add_argument(
        "--use_chartevents", action="store_true",
        help="Include ICU vital sign features from chartevents.csv.gz (large file — adds ~10-30 min)"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run randomized hyperparameter search for the SVM (adds significant runtime)"
    )
    parser.add_argument(
        "--n_iter", type=int, default=30,
        help="Number of hyperparameter combinations to try when --tune is set (default: 30)"
    )

    args = parser.parse_args()
    main(args)