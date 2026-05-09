"""
viz_chartevent_frequency.py
===========================
Visualisations for how frequently chart events are recorded in the
cohort produced by data_utils.py.

Reads the cached tensor file (./cache/processed_<hash>.pt) created by
get_dataloaders() and plots six panels:

  1. Histogram: total observed (hour x vital) cells per patient
  2. Mean observation rate per hour across the 48 h window (overall)
  3. Per-vital observation rate over time   (heat-map style line plot)
  4. Distribution of #vitals observed in any given hour
  5. Coverage by class: dementia+ vs dementia-
  6. Per-patient observation count, split by class

The mask lives in channels [n_vital_signals : 2*n_vital_signals] of x,
so we derive everything from that without reloading chartevents.csv.
"""

import os
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
VITAL_NAMES = ["heart_rate", "sbp", "dbp", "temperature", "spo2", "resp_rate"]


def find_cache(cache_dir: str = "/cache") -> str:
    matches = sorted(
        glob.glob(os.path.join(cache_dir, "processed.pt")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No processed_*.pt cache found in {cache_dir}. "
            "Run data_utils.get_dataloaders() first."
        )
    return matches[0]


def load_mask_and_labels(cache_path: str):
    blob = torch.load(cache_path, weights_only=False)
    print(blob)
    x = blob["x"]                       # (N, T, C)
    labels = blob["labels"].numpy()     # (N,)
    meta = blob["meta"]

    n_vital = meta["n_vital_signals"]
    # Mask channels are [n_vital : 2*n_vital]
    mask = x[:, :, n_vital : 2 * n_vital].numpy()  # (N, T, n_vital)

    return mask, labels, meta


def make_figure(mask: np.ndarray, labels: np.ndarray, meta: dict, out_path: str):
    N, T, V = mask.shape
    print(f"Cohort: {N:,} patients   T={T} h   {V} vitals")

    pos = labels == 1
    neg = labels == 0

    # ── derived stats ─────────────────────────────────────────────────────
    obs_per_patient = mask.sum(axis=(1, 2))          # (N,)
    obs_rate_by_hour = mask.mean(axis=(0, 2))        # (T,)
    obs_rate_by_hour_per_vital = mask.mean(axis=0)   # (T, V)
    vitals_seen_per_hour = mask.sum(axis=2).ravel()  # (N*T,)

    coverage_by_class = {
        "dementia+": mask[pos].mean(),
        "dementia-": mask[neg].mean(),
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))

    # 1. Histogram of total observed cells per patient
    ax = axes[0, 0]
    ax.hist(obs_per_patient, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(np.median(obs_per_patient), color="k", linestyle="--",
               label=f"median = {np.median(obs_per_patient):.0f}")
    ax.set_title("Observed (hour × vital) cells per patient")
    ax.set_xlabel(f"# observed cells (max = {T*V})")
    ax.set_ylabel("Patients")
    ax.legend()

    # 2. Mean observation rate per hour
    ax = axes[0, 1]
    ax.plot(np.arange(T), obs_rate_by_hour, color="#C44E52", linewidth=2)
    ax.fill_between(np.arange(T), 0, obs_rate_by_hour, color="#C44E52", alpha=0.15)
    ax.set_ylim(0, 1)
    ax.set_title("Mean observation rate per hour (averaged over vitals)")
    ax.set_xlabel("Hour from first chart event")
    ax.set_ylabel("Fraction of patients × vitals observed")

    # 3. Per-vital observation rate over time
    ax = axes[1, 0]
    names = meta.get("vital_names", VITAL_NAMES)
    for j in range(V):
        ax.plot(np.arange(T), obs_rate_by_hour_per_vital[:, j],
                label=names[j], linewidth=1.5)
    ax.set_ylim(0, 1)
    ax.set_title("Observation rate per hour, by vital")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Fraction of patients observed")
    ax.legend(fontsize=8, ncol=2)

    # 4. Distribution of #vitals observed in any given hour
    ax = axes[1, 1]
    counts, _ = np.histogram(vitals_seen_per_hour, bins=np.arange(V + 2) - 0.5)
    ax.bar(np.arange(V + 1), counts / counts.sum(),
           color="#55A868", edgecolor="white")
    ax.set_xticks(np.arange(V + 1))
    ax.set_title("Per-hour count: how many of the 6 vitals were observed?")
    ax.set_xlabel("# vitals observed in that hour")
    ax.set_ylabel("Fraction of patient-hours")

    # 5. Coverage by class (mean over patient × hour × vital)
    ax = axes[2, 0]
    classes = list(coverage_by_class.keys())
    vals = [coverage_by_class[c] for c in classes]
    bars = ax.bar(classes, vals, color=["#8172B2", "#937860"], edgecolor="white")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_title("Average mask density by class")
    ax.set_ylabel("Mean fraction observed")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)

    # 6. Per-patient observation count by class (overlaid hist)
    ax = axes[2, 1]
    bins = np.linspace(0, obs_per_patient.max(), 40)
    ax.hist(obs_per_patient[pos], bins=bins, alpha=0.55,
            label=f"dementia+ (n={pos.sum():,})", color="#8172B2")
    ax.hist(obs_per_patient[neg], bins=bins, alpha=0.55,
            label=f"dementia- (n={neg.sum():,})", color="#937860")
    ax.set_title("Total observed cells per patient, by class")
    ax.set_xlabel("# observed cells")
    ax.set_ylabel("Patients")
    ax.legend()

    fig.suptitle(
        f"Chart-event recording frequency  "
        f"(N={N:,}  ·  T={T} h  ·  {V} vitals)",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Numeric summary printed for the report
    print("\n── summary ──")
    print(f"obs_per_patient  median={np.median(obs_per_patient):.0f}  "
          f"mean={obs_per_patient.mean():.1f}  "
          f"max={obs_per_patient.max():.0f}  (theoretical max={T*V})")
    print(f"hour-1 mean rate : {obs_rate_by_hour[0]:.3f}")
    print(f"hour-{T-1} mean rate: {obs_rate_by_hour[-1]:.3f}")
    print(f"coverage dementia+ : {coverage_by_class['dementia+']:.3f}")
    print(f"coverage dementia- : {coverage_by_class['dementia-']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None,
                    help="Path to processed_<hash>.pt (default: latest in ./cache)")
    ap.add_argument("--out", default="chartevent_frequency.png",
                    help="Output figure path")
    args = ap.parse_args()

    cache_path = Path("cache/processed_d0af355e.pt")
    mask, labels, meta = load_mask_and_labels(cache_path)
    make_figure(mask, labels, meta, args.out)


if __name__ == "__main__":
    main()