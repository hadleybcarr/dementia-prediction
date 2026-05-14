import json
import matplotlib.pyplot as plt
from pathlib import Path


METRICS = [
    "train_loss", "val_loss",
    "train_acc", "val_acc",
    "train_auc",
    "train_precision",
    "t_recall",
    "train_f1",
]


def load_history(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_metric(history_dict, metric, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))

    for model_name, hist in history_dict.items():
        if metric not in hist:
            continue
        values = hist[metric]
        epochs = list(range(1, len(values) + 1))
        plt.plot(epochs, values, label=model_name)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_all(history_dict, out_dir="comparison_plots"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Loss curves
    plot_metric(
        history_dict,
        "train_loss",
        "Training Loss Comparison",
        "Loss",
        out_dir / "train_loss.png",
    )

    plot_metric(
        history_dict,
        "val_loss",
        "Validation Loss Comparison",
        "Loss",
        out_dir / "val_loss.png",
    )

    # 2. Accuracy
    plot_metric(
        history_dict,
        "train_acc",
        "Training Accuracy Comparison",
        "Accuracy",
        out_dir / "train_acc.png",
    )

    plot_metric(
        history_dict,
        "val_acc",
        "Validation Accuracy Comparison",
        "Accuracy",
        out_dir / "val_acc.png",
    )

    # 3. AUC
    plot_metric(
        history_dict,
        "train_auc",
        "Training AUC Comparison",
        "AUC",
        out_dir / "train_auc.png",
    )

    # 4. Precision / Recall / F1
    plot_metric(
        history_dict,
        "train_precision",
        "Training Precision Comparison",
        "Precision",
        out_dir / "train_precision.png",
    )

    plot_metric(
        history_dict,
        "t_recall",
        "Training Recall Comparison",
        "Recall",
        out_dir / "train_recall.png",
    )

    plot_metric(
        history_dict,
        "train_f1",
        "Training F1 Comparison",
        "F1 Score",
        out_dir / "train_f1.png",
    )


def plot_combined_final_metrics(history_dict, out_dir="comparison_plots"):
    """
    Bar chart comparing FINAL epoch performance across models.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    final_metrics = ["val_acc", "val_auc", "train_f1"]

    for metric in final_metrics:
        plt.figure(figsize=(6, 4))

        models = []
        values = []

        for model_name, hist in history_dict.items():
            if metric in hist and len(hist[metric]) > 0:
                models.append(model_name)
                values.append(hist[metric][-1])

        plt.bar(models, values)
        plt.title(f"Final {metric}")
        plt.ylabel(metric)
        plt.ylim(0, 1.0)
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / f"final_{metric}.png", dpi=150)
        plt.close()


def plot_generalization_gap(history_dict, out_dir="comparison_plots"):
    """
    Shows overfitting: train vs val loss gap.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(8, 5))

    for model_name, hist in history_dict.items():
        if "train_loss" in hist and "val_loss" in hist:
            train = hist["train_loss"]
            val = hist["val_loss"]
            gap = [v - t for t, v in zip(train, val)]
            epochs = range(1, len(gap) + 1)
            plt.plot(epochs, gap, label=model_name)

    plt.axhline(0, linestyle="--", color="black", linewidth=1)
    plt.title("Generalization Gap (Val Loss - Train Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Gap")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "generalization_gap.png", dpi=150)
    plt.close()


def run_all(transformer_path, cnn_path, bilstm_path):
    history_dict = {
        "transformer": load_history(transformer_path),
        "cnn": load_history(cnn_path),
        "bilstm": load_history(bilstm_path),
    }

    plot_all(history_dict)
    plot_combined_final_metrics(history_dict)
    plot_generalization_gap(history_dict)

    print("Saved all comparison plots → comparison_plots/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer")
    parser.add_argument("--cnn")
    parser.add_argument("--bilstm")

    args = parser.parse_args()

    run_all(args.transformer, args.cnn, args.bilstm)