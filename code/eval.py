import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    brier_score_loss,
)

try:
    import matplotlib.pyplot as plt
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_predictions(model: nn.Module, loader, device=DEVICE):
    """
    Run model inference on a DataLoader.

    Returns:
      probs  : np.ndarray (N,) — predicted probabilities
      labels : np.ndarray (N,) — true binary labels
    """
    model.eval()
    all_probs  = []
    all_labels = []

    for vitals, icd, label in loader:
        vitals = vitals.to(device)
        icd    = icd.to(device)

        logits = model(vitals, icd)
        probs  = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(label.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute a full set of binary classification metrics.

    Args:
      probs     : predicted probabilities in [0, 1]
      labels    : true binary labels (0 or 1)
      threshold : decision threshold for binary predictions (default 0.5)

    Returns:
      dict of metric names → values
    """
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # precision
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "auroc":       roc_auc_score(labels, probs),
        "auprc":       average_precision_score(labels, probs),
        "accuracy":    (tp + tn) / len(labels),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv":         ppv,
        "npv":         npv,
        "f1":          f1_score(labels, preds, zero_division=0),
        "brier":       brier_score_loss(labels, probs),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n":  len(labels),
        "n_pos": int(labels.sum()),
        "n_neg": int((1 - labels).sum()),
    }


def evaluate_model(model: nn.Module, loader, threshold: float = 0.5, name: str = "Model") -> dict:
    """
    Evaluate a model on a DataLoader and print a formatted summary.

    Returns:
      dict of metrics
    """
    probs, labels = get_predictions(model, loader)
    metrics = compute_metrics(probs, labels, threshold)

    print(f"\n── {name} ──────────────────────────────────────────")
    print(f"  N={metrics['n']:,}  (Pos={metrics['n_pos']:,}, Neg={metrics['n_neg']:,})")
    print(f"  AUROC        : {metrics['auroc']:.4f}")
    print(f"  AUPRC        : {metrics['auprc']:.4f}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Sensitivity  : {metrics['sensitivity']:.4f}   (Recall / TPR)")
    print(f"  Specificity  : {metrics['specificity']:.4f}   (TNR)")
    print(f"  PPV          : {metrics['ppv']:.4f}   (Precision)")
    print(f"  NPV          : {metrics['npv']:.4f}")
    print(f"  F1 Score     : {metrics['f1']:.4f}")
    print(f"  Brier Score  : {metrics['brier']:.4f}   (lower = better calibrated)")
    print(f"  Confusion Matrix → TP={metrics['tp']}, TN={metrics['tn']}, "
          f"FP={metrics['fp']}, FN={metrics['fn']}")

    metrics["probs"]  = probs
    metrics["labels"] = labels
    return metrics


def compare_models(models: dict, loader, threshold: float = 0.5) -> dict:
    """
    Evaluate multiple models and print a comparison table.

    Args:
      models : dict of {name: model}  e.g. {"Transformer": model1, "CNN": model2}
      loader : test DataLoader

    Returns:
      dict of {name: metrics_dict}
    """
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, loader, threshold, name)

    # Summary table
    print("\n\n── Comparison Summary ──────────────────────────────────")
    header = f"{'Model':<14}{'AUROC':>8}{'AUPRC':>8}{'F1':>8}{'Acc':>8}{'Sens':>8}{'Spec':>8}"
    print(header)
    print("─" * len(header))
    for name, m in results.items():
        print(
            f"{name:<14}"
            f"{m['auroc']:>8.4f}"
            f"{m['auprc']:>8.4f}"
            f"{m['f1']:>8.4f}"
            f"{m['accuracy']:>8.4f}"
            f"{m['sensitivity']:>8.4f}"
            f"{m['specificity']:>8.4f}"
        )
    return results


def plot_roc_curves(results: dict, save_path: str = None):
    """
    Plot ROC curves for multiple models on the same axes.

    Args:
      results   : dict from compare_models()
      save_path : optional file path to save the figure (e.g. "roc.png")
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2E5F8A", "#E06C2E", "#3DAF6A", "#9B59B6"]

    # ROC
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUROC=0.50)")
    for (name, m), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(m["labels"], m["probs"])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUROC={m['auroc']:.3f})")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # Precision-Recall
    ax = axes[1]
    for (name, m), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(m["labels"], m["probs"])
        ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AUPRC={m['auprc']:.3f})")
    prevalence = results[list(results.keys())[0]]["n_pos"] / results[list(results.keys())[0]]["n"]
    ax.axhline(prevalence, color="k", linestyle="--", alpha=0.4, label=f"Random (AUPRC={prevalence:.2f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.suptitle("Dementia Risk Prediction — Model Comparison", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_training_history(histories: dict, save_path: str = None):
    """
    Plot training/validation loss curves for multiple models.

    Args:
      histories : dict of {model_name: history_dict}
                  (history_dict from train.py contains "train_loss", "val_loss", etc.)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    colors = ["#2E5F8A", "#E06C2E", "#3DAF6A"]

    for ax, metric, ylabel in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss (BCE)", "Accuracy"]
    ):
        for (name, hist), color in zip(histories.items(), colors):
            epochs = range(1, len(hist[metric[0]]) + 1)
            ax.plot(epochs, hist[metric[0]], color=color, linestyle="--", alpha=0.6, label=f"{name} train")
            ax.plot(epochs, hist[metric[1]], color=color, linestyle="-",  linewidth=2, label=f"{name} val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Training {ylabel}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


@torch.no_grad()
def plot_attention_weights(bilstm_model, vitals_batch: torch.Tensor,
                           icd_batch: torch.Tensor, patient_idx: int = 0,
                           vital_names=None, save_path: str = None):
    """
    Visualise the temporal attention weights of the BiLSTM for a single patient.

    Args:
      bilstm_model : trained DementiaBiLSTM instance
      vitals_batch : (batch, seq_len, n_vitals) tensor
      icd_batch    : (batch, n_icd_codes) tensor
      patient_idx  : which patient in the batch to plot (default 0)
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    bilstm_model.eval()
    vitals_batch = vitals_batch.to(DEVICE)
    icd_batch    = icd_batch.to(DEVICE)

    _, attn_weights = bilstm_model.forward_with_attention(vitals_batch, icd_batch)
    weights = attn_weights[patient_idx].cpu().numpy()   # (seq_len,)
    vitals  = vitals_batch[patient_idx].cpu().numpy()   # (seq_len, n_vitals)

    if vital_names is None:
        vital_names = ["HR", "SBP", "DBP", "Temp", "SpO2", "RespR"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Attention weights
    axes[0].bar(range(len(weights)), weights, color="#2E5F8A", alpha=0.8)
    axes[0].set_ylabel("Attention Weight")
    axes[0].set_title(f"BiLSTM Temporal Attention — Patient {patient_idx}")
    axes[0].grid(axis="y", alpha=0.3)

    # Vital signs heatmap
    im = axes[1].imshow(vitals.T, aspect="auto", cmap="RdYlGn",
                        vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_yticks(range(len(vital_names)))
    axes[1].set_yticklabels(vital_names)
    axes[1].set_xlabel("Time Step (hours)")
    axes[1].set_title("Normalised Vital Signs")
    plt.colorbar(im, ax=axes[1], label="Normalised value")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("evaluate.py loaded. Import and call evaluate_model() / compare_models().")
    print("\nExample usage in Jupyter:\n")
    print("  from evaluate import evaluate_model, compare_models, plot_roc_curves")
    print("  metrics = evaluate_model(model, test_loader, name='Transformer')")
    print("  results = compare_models({'Transformer': m1, 'CNN': m2, 'BiLSTM': m3}, test_loader)")
    print("  plot_roc_curves(results, save_path='roc_curves.png')")