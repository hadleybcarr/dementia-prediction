import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

HOSP_PATH = "/oscar/data/shared/ursa/mimic-iv/hosp/3.1"
ICU_PATH  = "/oscar/data/shared/ursa/mimic-iv/icu/3.1"

DIAGNOSES_PATH  = os.path.join(HOSP_PATH, "diagnoses_icd.csv")
PATIENTS_PATH   = os.path.join(HOSP_PATH, "patients.csv")
CHARTEVENTS_PATH = os.path.join(ICU_PATH,  "chartevents.csv")

SEQ_LEN    = 48          # number of time steps per patient (hours)
N_VITALS   = 6           # number of vital sign channels
CHUNK_SIZE = 500_000     # rows per chunk when reading chartevents
SEED       = 42

# Dementia ICD-10 prefixes (positive class)
DEMENTIA_CODES = ("F01", "F02", "F03")

# Vital sign item IDs to extract from chartevents
VITAL_ITEM_IDS = {
    220045: "heart_rate",
    220179: "sbp",
    220180: "dbp",
    223761: "temperature",
    220277: "spo2",
    220210: "resp_rate",
}

# Per-vital normalization bounds (clip then scale to [0,1])
VITAL_BOUNDS = {
    "heart_rate":  (20,  250),
    "sbp":         (50,  250),
    "dbp":         (20,  180),
    "temperature": (90,  108),   # Fahrenheit
    "spo2":        (50,  100),
    "resp_rate":   (4,   60),
}


def get_subject_labels(diagnoses_path: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [subject_id, label].
    label=1 if patient has any dementia diagnosis (F01/F02/F03), else 0.
    Patients without dementia are sampled as controls (same size as positives).
    """
    print("Loading diagnoses...")
    df = pd.read_csv(diagnoses_path, usecols=["subject_id", "icd_code"])

    mask = df["icd_code"].str.startswith(DEMENTIA_CODES, na=False)
    positive_ids = df.loc[mask, "subject_id"].unique()

    all_ids = df["subject_id"].unique()
    negative_ids = np.setdiff1d(all_ids, positive_ids)

    rng = np.random.default_rng(SEED)
    n_pos = len(positive_ids)
    negative_ids = rng.choice(negative_ids, size=min(n_pos, len(negative_ids)), replace=False)

    pos_df = pd.DataFrame({"subject_id": positive_ids, "label": 1})
    neg_df = pd.DataFrame({"subject_id": negative_ids, "label": 0})
    labels = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=SEED)

    print(f"  Positives (dementia): {n_pos:,}")
    print(f"  Negatives (controls): {len(negative_ids):,}")
    return labels



def build_vitals_matrix(chartevents_path: str, subject_ids: np.ndarray) -> np.ndarray:
    """
    Reads chartevents in chunks, filters to target subjects and vital item IDs,
    then creates a (n_subjects, SEQ_LEN, N_VITALS) array.

    Missing values are filled with the per-vital mean (forward-fill first, then mean).
    """
    print("Loading chartevents (chunked)...")
    item_ids = list(VITAL_ITEM_IDS.keys())
    vital_names = list(VITAL_ITEM_IDS.values())
    subject_set = set(subject_ids.tolist())

    records = []
    for chunk in pd.read_csv(
        chartevents_path,
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        sub = chunk[
            chunk["subject_id"].isin(subject_set) &
            chunk["itemid"].isin(item_ids)
        ].copy()
        if not sub.empty:
            records.append(sub)
        print(f"  Processed {len(records) * CHUNK_SIZE:,} rows...", end="\r")

    if not records:
        raise ValueError("No chartevents found for the target subjects. Check paths/IDs.")

    df = pd.concat(records, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"])
    df["vital"] = df["itemid"].map(VITAL_ITEM_IDS)
    print(f"\n  chartevents rows after filtering: {len(df):,}")

    # For each subject, take up to SEQ_LEN hourly buckets from first observation
    vitals_out = np.full((len(subject_ids), SEQ_LEN, N_VITALS), np.nan, dtype=np.float32)

    for i, sid in enumerate(subject_ids):
        pat = df[df["subject_id"] == sid].copy()
        if pat.empty:
            continue

        t0 = pat["charttime"].min()
        pat["hour"] = ((pat["charttime"] - t0).dt.total_seconds() / 3600).astype(int)
        pat = pat[(pat["hour"] >= 0) & (pat["hour"] < SEQ_LEN)]

        pivot = (
            pat.groupby(["hour", "vital"])["valuenum"]
            .mean()
            .unstack("vital")
            .reindex(index=range(SEQ_LEN), columns=vital_names)
        )

        # Clip and normalise each vital
        for j, v in enumerate(vital_names):
            lo, hi = VITAL_BOUNDS[v]
            col = pivot[v].values.astype(np.float32)
            col = np.clip(col, lo, hi)
            col = (col - lo) / (hi - lo)          # scale to [0, 1]
            vitals_out[i, :, j] = col

    # Fill NaNs: forward-fill per patient, then global mean
    for i in range(len(subject_ids)):
        seq = pd.DataFrame(vitals_out[i], columns=vital_names)
        seq = seq.ffill().bfill()
        vitals_out[i] = seq.values

    col_means = np.nanmean(vitals_out.reshape(-1, N_VITALS), axis=0)
    for j in range(N_VITALS):
        nan_mask = np.isnan(vitals_out[:, :, j])
        vitals_out[:, :, j][nan_mask] = col_means[j]

    print(f"  Vitals matrix shape: {vitals_out.shape}")
    return vitals_out


class DementiaDataset(Dataset):
    def __init__(self, vitals: np.ndarray, labels: np.ndarray):
        self.vitals  = torch.tensor(vitals,  dtype=torch.float32)
        self.labels  = torch.tensor(labels,  dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.vitals[idx], self.labels[idx]


def get_dataloaders(batch_size: int = 64, val_size: float = 0.15, test_size: float = 0.15):
    """
    Returns:
      train_loader, val_loader, test_loader  : PyTorch DataLoaders
      meta : dict with n_icd_codes, seq_len, n_vitals, mlb (for decoding)

    Example:
      train_loader, val_loader, test_loader, meta = get_dataloaders()
      n_icd = meta["n_icd_codes"]   # pass to model constructors
    """
    label_df = get_subject_labels(DIAGNOSES_PATH)
    subject_ids = label_df["subject_id"].values
    labels      = label_df["label"].values

    vitals_matrix = build_vitals_matrix(CHARTEVENTS_PATH, subject_ids)

    idx = np.arange(len(labels))
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        idx, labels, test_size=val_size + test_size, stratify=labels, random_state=SEED
    )
    rel_test = test_size / (val_size + test_size)
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=rel_test, stratify=y_tmp, random_state=SEED
    )

    def make_loader(indices, shuffle):
        ds = DementiaDataset(
            vitals_matrix[indices],
            labels[indices],
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

    train_loader = make_loader(idx_train, shuffle=True)
    val_loader   = make_loader(idx_val,   shuffle=False)
    test_loader  = make_loader(idx_test,  shuffle=False)

    meta = {
        "seq_len":     SEQ_LEN,
        "n_vitals":    N_VITALS,
        "label_df":    label_df,
    }

    print(f"\nData splits — Train: {len(idx_train):,} | Val: {len(idx_val):,} | Test: {len(idx_test):,}")
    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    train_loader, val_loader, test_loader, meta = get_dataloaders()
    vitals, labels = next(iter(train_loader))
    print(f"Batch vitals shape : {vitals.shape}")    # (64, 48, 6)    # (64, 200)
    print(f"Batch labels shape : {labels.shape}")    # (64,)