"""
data_utils.py
=============
Cohort-aware MIMIC-IV data pipeline for dementia risk prediction.

Key design changes vs. the previous version:

1. Cohort filtered to patients with adequate ICU vital-sign data.
   Patients in `diagnoses_icd.csv` who have NO usable chart events (most
   non-ICU patients) are excluded. Previously they were included with
   their input filled by the global column mean — i.e. a constant tensor
   paired with a binary label, which dilutes whatever real signal exists.

2. Demographics (anchor_age, sex) are loaded from patients.csv and
   concatenated as constant channels broadcast across time. These are far
   more predictive of dementia than 48h of vitals.

3. Missingness mask added as 6 additional channels (1 if a vital was
   observed at that hour, 0 otherwise). In ICU data, *whether* something
   was charted is informative on its own.

4. Final input shape is (B, T, 14):
     channels  0..5   normalised vitals (clipped to [0,1])
     channels  6..11  missingness mask
     channels 12..13  age, sex (broadcast across time)
   Your model code does NOT need to change. `build_transformer/cnn/bilstm`
   read `meta["n_vitals"]`; this file now sets it to 14.

5. `meta["pos_weight"]` is now precomputed from the training split.
   Pass it to `BCEWithLogitsLoss(pos_weight=...)` in train.py.

6. Cache filename includes a config hash, so changing any constant here
   automatically invalidates the cache.

Label is unchanged: any patient with F01/F02/F03 anywhere in
diagnoses_icd is positive. This is a "prevalent dementia" task. If you
later want incident-dementia-after-admission, that requires joining
admissions.csv and re-anchoring the vitals window per admission.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
HOSP_PATH = "/oscar/data/shared/ursa/mimic-iv/hosp/3.1"
ICU_PATH  = "/oscar/data/shared/ursa/mimic-iv/icu/3.1"

DIAGNOSES_PATH   = os.path.join(HOSP_PATH, "diagnoses_icd.csv")
PATIENTS_PATH    = os.path.join(HOSP_PATH, "patients.csv")
CHARTEVENTS_PATH = os.path.join(ICU_PATH,  "chartevents.csv")

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN          = 48        # hourly time steps per patient
N_VITALS  = 6         # raw vital sign channels
N_DEMO_CHANNELS  = 2         # age + sex
TOTAL_CHANNELS   = N_VITALS * 2 + N_DEMO_CHANNELS  # vitals + mask + demo = 14

CHUNK_SIZE       = 500_000
SEED             = 42

# Patients with fewer than this many filtered chart-event rows are dropped.
# Most non-ICU patients have 0; this keeps anyone with at least a brief stay.
MIN_CHART_OBSERVATIONS = 10

DEMENTIA_CODES = ("F01", "F02", "F03")

VITAL_ITEM_IDS = {
    220045: "heart_rate",
    220179: "sbp",
    220180: "dbp",
    223761: "temperature",
    220277: "spo2",
    220210: "resp_rate",
}

# Clip then min-max scale to [0, 1]
VITAL_BOUNDS = {
    "heart_rate":  (20,  250),
    "sbp":         (50,  250),
    "dbp":         (20,  180),
    "temperature": (90,  108),   # Fahrenheit
    "spo2":        (50,  100),
    "resp_rate":   (4,   60),
}

AGE_BOUNDS = (18, 100)

# ── Cache key (auto-invalidates when config changes) ──────────────────────────
_CONFIG_BLOB = json.dumps({
    "seq_len":   SEQ_LEN,
    "n_vital":   N_VITALS,
    "min_obs":   MIN_CHART_OBSERVATIONS,
    "vitals":    list(VITAL_ITEM_IDS.values()),
    "bounds":    {k: list(v) for k, v in VITAL_BOUNDS.items()},
    "age":       list(AGE_BOUNDS),
    "codes":     list(DEMENTIA_CODES),
    "channels":  TOTAL_CHANNELS,
}, sort_keys=True)
_CONFIG_HASH = hashlib.md5(_CONFIG_BLOB.encode()).hexdigest()[:8]
CACHE_PATH   = f"./cache/processed_{_CONFIG_HASH}.pt"


# ── Cohort & labels ───────────────────────────────────────────────────────────
def get_subject_labels(diagnoses_path: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns [subject_id, label]:
      label = 1 if patient ever has F01/F02/F03 ICD code, else 0.
      Negatives are randomly downsampled to match positive count.
    """
    print("Loading diagnoses...")
    df = pd.read_csv(diagnoses_path, usecols=["subject_id", "icd_code"])
    df["icd_code"] = df["icd_code"].astype(str)

    mask = df["icd_code"].str.startswith(DEMENTIA_CODES, na=False)
    positive_ids = df.loc[mask, "subject_id"].unique()

    all_ids      = df["subject_id"].unique()
    negative_ids = np.setdiff1d(all_ids, positive_ids)

    rng = np.random.default_rng(SEED)
    n_pos = len(positive_ids)
    negative_ids = rng.choice(
        negative_ids,
        size=min(n_pos, len(negative_ids)),
        replace=False,
    )

    pos_df = pd.DataFrame({"subject_id": positive_ids, "label": 1})
    neg_df = pd.DataFrame({"subject_id": negative_ids, "label": 0})
    labels = (
        pd.concat([pos_df, neg_df], ignore_index=True)
          .sample(frac=1, random_state=SEED)
          .reset_index(drop=True)
    )

    print(f"  Positives (dementia): {n_pos:,}")
    print(f"  Negatives (controls): {len(negative_ids):,}")
    return labels


# ── Demographics ──────────────────────────────────────────────────────────────
def load_demographics(patients_path: str, subject_ids: np.ndarray) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by subject_id with columns [age, sex],
    both scaled into [0, 1].
      age : anchor_age clipped to AGE_BOUNDS, then min-max scaled
      sex : F → 0.0, M → 1.0, anything else → 0.5
    Subjects missing from patients.csv get NaN; the caller handles that.
    """
    print("Loading patients (demographics)...")
    df = pd.read_csv(patients_path, usecols=["subject_id", "anchor_age", "gender"])
    df = df[df["subject_id"].isin(subject_ids)].copy()

    lo, hi = AGE_BOUNDS
    age = pd.to_numeric(df["anchor_age"], errors="coerce").astype(float)
    age = age.clip(lower=lo, upper=hi)
    df["age"] = (age - lo) / (hi - lo)

    sex_map = {"F": 0.0, "M": 1.0}
    df["sex"] = df["gender"].map(sex_map).fillna(0.5)

    return df.set_index("subject_id")[["age", "sex"]]


# ── Vitals ────────────────────────────────────────────────────────────────────
def build_vitals_matrix(chartevents_path: str, subject_ids: np.ndarray):
    """
    Reads chartevents in chunks, filters to target subjects and vital itemids,
    then constructs:
      vitals : (n_subjects, SEQ_LEN, N_VITAL_SIGNALS) — float32, clipped + min-max to [0,1]
      mask   : (n_subjects, SEQ_LEN, N_VITAL_SIGNALS) — float32, 1 if observed, 0 if imputed
      n_obs  : (n_subjects,) — int32, total filtered chart-event rows per subject
    """
    print("Loading chartevents (chunked)...")
    item_ids    = list(VITAL_ITEM_IDS.keys())
    vital_names = list(VITAL_ITEM_IDS.values())
    subject_set = set(subject_ids.tolist())

    records = []
    rows_seen = 0
    for chunk in pd.read_csv(
        chartevents_path,
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        rows_seen += len(chunk)
        sub = chunk[
            chunk["subject_id"].isin(subject_set) &
            chunk["itemid"].isin(item_ids)
        ]
        if not sub.empty:
            records.append(sub)
        print(f"  Read {rows_seen:,} rows  ({len(records)} chunks kept)...", end="\r")
    print()

    if not records:
        raise ValueError("No matching chartevents rows. Check paths/itemids.")

    df = pd.concat(records, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"])
    df["vital"]     = df["itemid"].map(VITAL_ITEM_IDS)
    print(f"  chartevents rows after filtering: {len(df):,}")

    n_subj = len(subject_ids)
    vitals = np.full((n_subj, SEQ_LEN, N_VITALS), np.nan, dtype=np.float32)
    mask   = np.zeros((n_subj, SEQ_LEN, N_VITALS), dtype=np.float32)
    n_obs  = np.zeros(n_subj, dtype=np.int32)

    # One groupby up front is much faster than the previous per-subject filter.
    groups = df.groupby("subject_id", sort=False)
    sid_to_idx = {sid: i for i, sid in enumerate(subject_ids.tolist())}

    for sid, pat in groups:
        i = sid_to_idx.get(sid)
        if i is None:
            continue
        n_obs[i] = len(pat)

        t0 = pat["charttime"].min()
        hours = ((pat["charttime"] - t0).dt.total_seconds() / 3600).astype(int)
        pat = pat.assign(hour=hours)
        pat = pat[(pat["hour"] >= 0) & (pat["hour"] < SEQ_LEN)]
        if pat.empty:
            continue

        pivot = (
            pat.groupby(["hour", "vital"])["valuenum"]
               .mean()
               .unstack("vital")
               .reindex(index=range(SEQ_LEN), columns=vital_names)
        )

        for j, v in enumerate(vital_names):
            lo, hi = VITAL_BOUNDS[v]
            col = pivot[v].values.astype(np.float32)
            observed = ~np.isnan(col)
            mask[i, :, j] = observed.astype(np.float32)
            col = np.clip(col, lo, hi)
            col = (col - lo) / (hi - lo)
            vitals[i, :, j] = col

    # Fill NaNs: per-patient ffill+bfill, then global column means as last resort.
    for i in range(n_subj):
        seq = pd.DataFrame(vitals[i], columns=vital_names).ffill().bfill()
        vitals[i] = seq.values

    col_means = np.nanmean(vitals.reshape(-1, N_VITAL_SIGNALS), axis=0)
    for j in range(N_VITAL_SIGNALS):
        nan_idx = np.isnan(vitals[:, :, j])
        vitals[:, :, j][nan_idx] = col_means[j]

    print(f"  Vitals matrix shape: {vitals.shape}")
    return vitals, mask, n_obs


# ── Assemble final input ──────────────────────────────────────────────────────
def build_full_input(
    vitals: np.ndarray,
    mask: np.ndarray,
    demo_df: pd.DataFrame,
    subject_ids: np.ndarray,
) -> np.ndarray:
    """
    Concatenate vitals + mask + (broadcast) demographics into a single tensor.

    Output: (n_subjects, SEQ_LEN, TOTAL_CHANNELS) float32
      ch  0.. 5  : normalised vitals
      ch  6..11  : missingness mask
      ch 12..13  : age, sex (constant across time)
    """
    n = len(subject_ids)
    out = np.zeros((n, SEQ_LEN, TOTAL_CHANNELS), dtype=np.float32)

    out[:, :, 0:N_VITALS]                   = vitals
    out[:, :, N_VITALS:2 * N_VITALS] = mask

    demo = demo_df.reindex(subject_ids)
    age  = demo["age"].fillna(0.5).values.astype(np.float32)   # neutral default
    sex  = demo["sex"].fillna(0.5).values.astype(np.float32)

    out[:, :, 2 * N_VITALS]     = age[:, None]   # broadcast over T
    out[:, :, 2 * N_VITALS + 1] = sex[:, None]

    return out


# ── Dataset / DataLoader ──────────────────────────────────────────────────────
class DementiaDataset(Dataset):
    def __init__(self, x, labels):
        self.x      = x      if isinstance(x, torch.Tensor)      else torch.tensor(x,      dtype=torch.float32)
        self.labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]


def get_dataloaders(
    batch_size: int = 64,
    val_size:   float = 0.15,
    test_size:  float = 0.15,
    force_rebuild: bool = False,
):
    """
    Returns:
      train_loader, val_loader, test_loader, meta

    meta keys:
      seq_len          : 48
      n_vitals         : 14   (kept under this name for compat with build_*())
      n_vital_signals  : 6
      vital_names      : list[str]
      pos_weight       : float — n_neg / n_pos in the TRAINING split.
                         Pass to nn.BCEWithLogitsLoss(pos_weight=torch.tensor([...])).
      n_train/val/test : ints
    """
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    if os.path.exists(CACHE_PATH) and not force_rebuild:
        print(f"Loading cached tensors from {CACHE_PATH}")
        blob      = torch.load(CACHE_PATH, weights_only=False)
        x         = blob["x"]
        labels    = blob["labels"]
        idx_train = blob["train"]
        idx_val   = blob["val"]
        idx_test  = blob["test"]
        meta      = blob["meta"]
    else:
        # 1. Cohort + labels
        label_df    = get_subject_labels(DIAGNOSES_PATH)
        subject_ids = label_df["subject_id"].values
        labels_arr  = label_df["label"].values.astype(np.float32)

        # 2. Vitals + missingness mask
        vitals, mask, n_obs = build_vitals_matrix(CHARTEVENTS_PATH, subject_ids)

        # 3. Drop patients with too little real data
        keep = n_obs >= MIN_CHART_OBSERVATIONS
        dropped = int((~keep).sum())
        if dropped:
            print(f"  Dropped {dropped:,} patients with < {MIN_CHART_OBSERVATIONS} chart events")
        subject_ids = subject_ids[keep]
        labels_arr  = labels_arr[keep]
        vitals      = vitals[keep]
        mask        = mask[keep]
        print(f"  Cohort after filtering: {len(subject_ids):,} "
              f"({int(labels_arr.sum()):,} pos / {int((1 - labels_arr).sum()):,} neg)")

        # 4. Demographics
        demo_df = load_demographics(PATIENTS_PATH, subject_ids)

        # 5. Assemble full (B, T, 14) tensor
        x_arr = build_full_input(vitals, mask, demo_df, subject_ids)

        # 6. Stratified split
        idx = np.arange(len(labels_arr))
        idx_train, idx_tmp, y_train, y_tmp = train_test_split(
            idx, labels_arr,
            test_size=val_size + test_size,
            stratify=labels_arr,
            random_state=SEED,
        )
        rel_test = test_size / (val_size + test_size)
        idx_val, idx_test = train_test_split(
            idx_tmp, test_size=rel_test, stratify=y_tmp, random_state=SEED,
        )

        # 7. pos_weight from TRAIN ONLY (do not peek at val/test)
        n_pos = float(labels_arr[idx_train].sum())
        n_neg = float(len(idx_train) - n_pos)
        pos_weight = n_neg / max(n_pos, 1.0)

        meta = {
            "seq_len":         SEQ_LEN,
            "n_vitals":        TOTAL_CHANNELS,    # what model constructors read
            "n_vital_signals": N_VITALS,
            "vital_names":     list(VITAL_ITEM_IDS.values()),
            "pos_weight":      pos_weight,
            "n_train":         len(idx_train),
            "n_val":           len(idx_val),
            "n_test":          len(idx_test),
        }

        x      = torch.tensor(x_arr,      dtype=torch.float32)
        labels = torch.tensor(labels_arr, dtype=torch.float32)

        torch.save({
            "x":      x,
            "labels": labels,
            "train":  idx_train,
            "val":    idx_val,
            "test":   idx_test,
            "meta":   meta,
        }, CACHE_PATH)
        print(f"Cached → {CACHE_PATH}")

    def make_loader(indices, shuffle):
        ds = DementiaDataset(x[indices], labels[indices])
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
        )

    train_loader = make_loader(idx_train, shuffle=True)
    val_loader   = make_loader(idx_val,   shuffle=False)
    test_loader  = make_loader(idx_test,  shuffle=False)

    print(
        f"\nSplits — Train: {meta['n_train']:,}  "
        f"Val: {meta['n_val']:,}  Test: {meta['n_test']:,}"
    )
    print(f"pos_weight (use in BCEWithLogitsLoss): {meta['pos_weight']:.3f}")
    print(
        f"Input channels: {meta['n_vitals']}  "
        f"({meta['n_vital_signals']} vitals + {meta['n_vital_signals']} mask + "
        f"{N_DEMO_CHANNELS} demo)"
    )
    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    train_loader, val_loader, test_loader, meta = get_dataloaders()
    x, y = next(iter(train_loader))
    print(f"\nBatch x shape : {x.shape}     # (B, {SEQ_LEN}, {TOTAL_CHANNELS})")
    print(f"Batch y shape : {y.shape}    y mean: {y.float().mean():.3f}")
    print(f"meta = {meta}")