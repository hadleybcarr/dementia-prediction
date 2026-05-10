"""
viz_vitals.py
=============
Visualises the vital-sign and demographic data actually stored in the
data_utils cache. Reads only the vital channels [0:N_VITALS] and the
two demographic channels (age, sex), so it works regardless of whether
mask channels were saved.

Six panels:
  1. Distribution of normalised values, per vital
  2. Mean ± IQR trajectory over time, per vital
  3. Class-conditional mean trajectory (dementia+ vs dementia-)
  4. Per-patient mean by vital, boxplot by class
  5. Age distribution by class
  6. Sex breakdown by class
"""

import os
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def find_cache(cache_dir: str = "./cache") -> str:
    matches = sorted(
        glob.glob(os.path.join(cache_dir, "processed_*.pt")),
        key=os.path.getmtime, reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No processed_*.pt cache in {cache_dir}. "
            "Run data_utils.get_dataloaders() first."
        )
    return matches[0]


def load_data(cache_path: str):
    """Pulls vitals, age, sex, labels from cache. Layout-agnostic — works with
    or without mask channels."""
    blob   = torch.load(cache_path, weights_only=False)
    x      = blob["x"].numpy()
    labels = blob["labels"].numpy()
    meta   = blob["meta"]

    n_vital = meta["n_vital_signals"]
    n_total = meta["n_vitals"]

    vitals = x[:, :, :n_vital]                          # (N, T, V)

    if n_total == 2 * n_vital + 2:                       # mask present
        age_idx, sex_idx = 2 * n_vital, 2 * n_vital + 1
    elif n_total == n_vital + 2:                         # no mask
        age_idx, sex_idx = n_vital, n_vital + 1
    else:
        raise ValueError(
            f"Unexpected channel layout: n_total={n_total}, n_vital={n_vital}"
        )

    # Demographics are broadcast across time; sample at t=0
    age = x[:, 0, age_idx]
    sex = x[:, 0, sex_idx]
    return vitals, age, sex, labels, meta


def make_figure(vitals, age, sex, labels, meta, out_path):
    N, T, V = vitals.shape
    vital_names = meta.get("vital_names", [f"vital_{i}" for i in range(V)])
    if len(vital_names) != V:
        vital_names = [f"vital_{i}" for i in range(V)]

    pos = labels == 1
    neg = labels == 0
    hours = np.arange(T)

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))

    # 1. Per-vital value distribution
    ax = axes[0, 0]
    for j in range(V):
        ax.hist(vitals[:, :, j].ravel(), bins=40, alpha=0.45,
                label=vital_names[j])
    ax.set_title("Distribution of normalised vital values")
    ax.set_xlabel("normalised value  [0, 1]")
    ax.set_ylabel("count  (patient-hours)")
    ax.legend(fontsize=8)

    # 2. Per-vital mean ± IQR trajectory
    ax = axes[0, 1]
    for j in range(V):
        v = vitals[:, :, j]
        mean = v.mean(axis=0)
        q25  = np.percentile(v, 25, axis=0)
        q75  = np.percentile(v, 75, axis=0)
        line, = ax.plot(hours, mean, label=vital_names[j], linewidth=1.5)
        ax.fill_between(hours, q25, q75, alpha=0.12, color=line.get_color())
    ax.set_title("Vital trajectories — mean line, IQR band")
    ax.set_xlabel("hour")
    ax.set_ylabel("normalised value")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2)

    # 3. Class-conditional mean per vital
    ax = axes[1, 0]
    for j in range(V):
        line_p, = ax.plot(hours, vitals[pos, :, j].mean(axis=0),
                          label=f"{vital_names[j]}", linestyle="-", linewidth=1.5)
        ax.plot(hours, vitals[neg, :, j].mean(axis=0),
                color=line_p.get_color(), linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_title("Class mean — solid = dementia+, dashed = dementia-")
    ax.set_xlabel("hour")
    ax.set_ylabel("normalised value")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)

    # 4. Per-patient mean of each vital, boxplot by class
    ax = axes[1, 1]
    pos_means = vitals[pos].mean(axis=1)
    neg_means = vitals[neg].mean(axis=1)
    width     = 0.35
    positions = np.arange(V)
    bp_pos = ax.boxplot([pos_means[:, j] for j in range(V)],
                        positions=positions - width/2, widths=width,
                        patch_artist=True,
                        boxprops=dict(facecolor="#8172B2", alpha=0.8),
                        medianprops=dict(color="black"))
    bp_neg = ax.boxplot([neg_means[:, j] for j in range(V)],
                        positions=positions + width/2, widths=width,
                        patch_artist=True,
                        boxprops=dict(facecolor="#937860", alpha=0.8),
                        medianprops=dict(color="black"))
    ax.set_xticks(positions)
    ax.set_xticklabels(vital_names, rotation=25, ha="right")
    ax.set_title("Per-patient mean by vital and class")
    ax.set_ylabel("mean normalised value")
    ax.legend([bp_pos["boxes"][0], bp_neg["boxes"][0]],
              ["dementia+", "dementia-"], fontsize=8)

    # 5. Age distribution by class
    ax = axes[2, 0]
    bins = np.linspace(0, 1, 30)
    ax.hist(age[pos], bins=bins, alpha=0.55,
            label=f"dementia+ (n={int(pos.sum()):,})", color="#8172B2")
    ax.hist(age[neg], bins=bins, alpha=0.55,
            label=f"dementia- (n={int(neg.sum()):,})", color="#937860")
    ax.set_title("Age by class  (normalised; 0=18 yrs, 1=100 yrs)")
    ax.set_xlabel("normalised age")
    ax.set_ylabel("patients")
    ax.legend()

    # 6. Sex breakdown by class (F=0.0, missing=0.5, M=1.0)
    ax = axes[2, 1]
    cats = ["F (0.0)", "missing (0.5)", "M (1.0)"]
    def _counts(s):
        return [
            int(((s >= -0.1) & (s < 0.25)).sum()),
            int(((s >= 0.25) & (s < 0.75)).sum()),
            int(((s >= 0.75) & (s <= 1.1)).sum()),
        ]
    pos_c, neg_c = _counts(sex[pos]), _counts(sex[neg])
    xpos = np.arange(len(cats))
    ax.bar(xpos - width/2, pos_c, width, color="#8172B2", label="dementia+")
    ax.bar(xpos + width/2, neg_c, width, color="#937860", label="dementia-")
    ax.set_xticks(xpos)
    ax.set_xticklabels(cats)
    ax.set_title("Sex by class")
    ax.set_ylabel("patients")
    ax.legend()

    fig.suptitle(
        f"Vital-sign data summary  "
        f"(N={N:,} · T={T} h · {V} vitals)",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Quick numeric summary
    print("\n── per-vital summary (normalised) ──")
    print(f"{'vital':<14}  {'mean':>6}  {'sd':>6}  {'p10':>6}  {'p90':>6}")
    for j in range(V):
        v = vitals[:, :, j].ravel()
        print(f"{vital_names[j]:<14}  "
              f"{v.mean():>6.3f}  {v.std():>6.3f}  "
              f"{np.percentile(v,10):>6.3f}  {np.percentile(v,90):>6.3f}")
    print(f"\nclass split: dementia+ = {int(pos.sum())}, dementia- = {int(neg.sum())}")
    print(f"age:  pos mean={age[pos].mean():.3f}  neg mean={age[neg].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None,
                    help="Path to processed_<hash>.pt (default: latest in ./cache)")
    ap.add_argument("--out", default="vitals_summary.png",
                    help="Output figure path")
    args = ap.parse_args()

    cache_path = Path("cache/processed_7469a882.pt")
    vitals, age, sex, labels, meta = load_data(cache_path)
    make_figure(vitals, age, sex, labels, meta, args.out)


if __name__ == "__main__":
    main()