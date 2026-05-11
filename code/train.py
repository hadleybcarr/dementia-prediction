import argparse
import os
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import SWALR, AveragedModel, update_bn
import json 
from sklearn.metrics import roc_auc_score

from data_utils import get_dataloaders
from transformer import build_transformer
from cnn import build_cnn
from bi_lstm import build_bilstm
from svm_model import svm_train
import shap 
import matplotlib.pyplot as plt

CHECKPOINT_DIR = Path("./checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name: str, meta: dict) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "transformer":
        return build_transformer(meta)
    elif model_name == "cnn":
        print("Building CNN...")
        return build_cnn(meta)
    elif model_name in ("bilstm", "bi-lstm", "bi_lstm"):
        return build_bilstm(meta)
    elif model_name in "svm":
        pass
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose: transformer | cnn | bilstm")


def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    print("Training", model,"...")
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    n_correct  = 0
    n_total    = 0
    all_probs, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for vitals, labels in loader:
            vitals  = vitals.to(device)
            labels  = labels.to(device)

            logits = model(vitals)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds       = (torch.sigmoid(logits) >= 0.5).float()
            n_correct  += (preds == labels).sum().item()
            n_total    += len(labels)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / n_total
    accuracy = n_correct / n_total

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else float("nan")

    print("AUROC is", auc)
    
    #with open("auroc.json", "wb") as f:
    #    json.dump(auc)

    return avg_loss, accuracy, auc



def train(
    model_name: str = "transformer",
    epochs:     int = 60,
    lr:         float = 1e-4,
    batch_size: int = 64,
    patience:   int = 7,        # early stopping patience
    pos_weight: float = 1.0,    # increase if dataset is imbalanced
    save_best:  bool = True,
):
    """
    Train a dementia risk model.

    Args:
      model_name  : "transformer", "cnn", "svm", or "bilstm"
      epochs      : max training epochs
      lr          : learning rate
      batch_size  : mini-batch size
      patience    : early stopping (stop if val loss doesn't improve for N epochs)
      pos_weight  : weight for positive class in BCEWithLogitsLoss (set > 1 if imbalanced)
      save_best   : whether to save the best checkpoint

    Returns:
      dict with training history and best val metrics
    """
    print(f"\n{'='*60}")
    print(f"  Model : {model_name.upper()}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    train_loader, val_loader, test_loader, meta = get_dataloaders(batch_size=batch_size)

    if model_name == "svm":
        return svm_train(train_loader, val_loader, test_loader)

    model = get_model(model_name, meta).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}\n")

    n_pos = sum(labels.sum().item() for _, labels in train_loader)
    n_neg = sum((1 - labels).sum().item() for _, labels in train_loader)
    pos_weight = n_neg / max(n_pos, 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE)
    )

    warmup_epochs = max(2,epochs //10)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=lr / 100)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    #scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader),
#)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"best_{model_name}.pt"

    history_default = {
                "svm": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
                "cnn": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
                "transformer": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
                "bilstm": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
                }         
    try:
        with open("history.json", "r") as file:
            history = json.load(file)
    except(FileNotFoundError, json.JSONDecodeError):
        history = history_default
        with open("history.json", "w") as f:
            json.dump(history, f, indent=2)
      
    best_val_auc = -float('inf')
    patience_counter = 0
    swa_model = AveragedModel(model)
    swa_start = int(epochs * 0.75)
    swa_scheduler = SWALR(optimizer, swa_lr = lr/10)


    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_auc = run_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss,   val_acc, val_auc   = run_epoch(model, val_loader,   criterion, None,      DEVICE)

            
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        history[model_name]["train_loss"].append(train_loss)
        history[model_name]["val_loss"].append(val_loss)
        history[model_name]["train_acc"].append(train_acc)
        history[model_name]["val_acc"].append(val_acc)


        print(history)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"| Train loss: {train_loss:.4f}  acc: {train_acc:.3f}"
            f"| Val loss: {val_loss:.4f}  acc: {val_acc:.3f}"
            f"| {elapsed:.1f}s"
        )

        # Early stopping + best checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            if save_best:
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(),
                     "val_loss": val_loss, "val_acc": val_acc, "val_auc": val_auc, "meta": meta},
                    ckpt_path,
                )
                print(f"  ✓ Saved best checkpoint → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if save_best and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"\nLoaded best checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    update_bn(train_loader, swa_model, device=DEVICE)
    test_loss, test_acc = run_epoch(swa_model, test_loader, criterion, None, DEVICE)
    print(f"\nTest  loss: {test_loss:.4f}  |  Test  acc: {test_acc:.3f}")
    
    with open("history.json", "w") as f:
            json.dump(history, f)

    for name, param in model.named_parameters():
         print(f"{name}: mean={param.mean().item():.4f}, "
          f"std={param.std().item():.4f}, "
          f"min={param.min().item():.4f}, "
          f"max={param.max().item():.4f}")
         
    if model != "svm":
        run_shap_analysis(model, train_loader, test_loader, meta, model_name, out_path=f"shap_{model_name}.png")

    return {
        "model":      model,
        "history":    history,
        "test_loss":  test_loss,
        "test_acc":   test_acc,
        "meta":       meta,
        "ckpt_path":  str(ckpt_path),
    }


def load_model(model_name: str, ckpt_path: str, device=DEVICE) -> tuple:
    """
    Load a saved checkpoint.

    Returns:
      model, meta
    """
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta  = ckpt["meta"]
    model = get_model(model_name, meta).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {model_name} from epoch {ckpt['epoch']}")
    return model, meta


def run_shap_analysis(model, train_loader, test_loader, meta, 
                      model_name, n_background: int=100, n_explain: int=200, out_path: str=None,device=DEVICE,):
    
    model = model.to(device).eval()
    vital_names = meta.get("vital_names", ["heart_rate", "sbp", "dbp", "spo2", "resp_rate"])
    n_v = meta["n_vital_signals"]
    channel_names = (
        list(vital_names)
        + ["age", "sex"]
    )

    def collect(loader,n):
        xs = []
        seen = 0
        for x, _ in loader:
            xs.append(x)
            seen += x.shape[0]
            if seen >= n:
                break
        return torch.cat(xs, dim=0)[:n].to(device)
    
    background = collect(train_loader, n_background)
    explain = collect(test_loader, n_explain)

    class _Wrap(nn.Module):
        def __init__(self,m): super().__init__(); self.m = m
        def forward(self, x):
            out = self.m(x)
            return out.unsqueeze(-1) if out.ndim == 1 else out
        
    wrapped = _Wrap(model).to(device).eval()
    explainer = shap.GradientExplainer(wrapped, background)
    shap_vals = explainer.shap_values(explain)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    if shap_vals.ndim == 4 and shap_vals.shape[-1] == 1:
        shap_vals = shap_vals.squeeze(-1)

    abs_shap = np.abs(shap_vals)
    per_channel = abs_shap.mean(axis=(0,1))
    per_time_chan = abs_shap.mean(axis=0)

    order = np.argsort(per_channel)[::-1]
    for k in order:
        print(f"  {channel_names[k]:<14}  {per_channel[k]:.5f}")

    fig, axes = plt.subplots(1,2, figsize=(14,6), gridspec_kw={"width_ratios":[1,1.2]})
    names_sorted = [channel_names[i] for i in order][::-1]
    vals_sorted = per_channel[order][::-1]
    axes[0].barh(names_sorted, vals_sorted, color="#4C72B0")
    axes[0].set_title("Mean |SHAP| per channel")
    axes[0].set_xlabel("Mean |SHAP|")

    im = axes[1].imshow(per_time_chan.T, aspect="auto", cmap="viridis")
    axes[1].set_yticks(range(len(channel_names)))
    axes[1].set_yticklabels(channel_names)
    axes[1].set_xlabel("Hour")
    axes[1].set_title("Mean |SHAP| per (hour, channel)")
    plt.colorbar(im, ax=axes[1])

    fig.suptitle(f"SHAP feature importance — {model_name}", y=1.02)
    fig.tight_layout()

    if out_path is None:
        out_path = f"shap_{model_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    return {
        "channel_names":  channel_names,
        "per_channel":    per_channel,
        "per_time_chan":  per_time_chan,
        "shap_values":    shap_vals,
    }
    

                             

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a dementia risk model on MIMIC-IV")
    parser.add_argument("--model",      type=str,   default="transformer",
                        choices=["transformer", "cnn", "bilstm", "svm"],
                        help="Which model architecture to train")
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--patience",   type=int,   default=7)
    parser.add_argument("--pos_weight", type=float, default=1.0,
                        help="Positive class weight for BCEWithLogitsLoss")
    args = parser.parse_args()

    results = train(
        model_name  = args.model,
        epochs      = args.epochs,
        lr          = args.lr,
        batch_size  = args.batch_size,
        patience    = args.patience,
        pos_weight  = args.pos_weight,
    )