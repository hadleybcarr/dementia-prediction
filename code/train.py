import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_utils import get_dataloaders
from transformer import build_transformer
from cnn import build_cnn
from bi_lstm import build_bilstm
from svm_model import build_svm

CHECKPOINT_DIR = Path("./checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name: str, meta: dict) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "transformer":
        return build_transformer(meta)
    elif model_name == "cnn":
        return build_cnn(meta)
    elif model_name in ("bilstm", "bi-lstm", "bi_lstm"):
        return build_bilstm(meta)
    elif model_name in ("svm"):
        return build_svm(meta)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose: transformer | cnn | bilstm")


def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    n_correct  = 0
    n_total    = 0

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

    avg_loss = total_loss / n_total
    accuracy = n_correct / n_total
    return avg_loss, accuracy



def train(
    model_name: str = "transformer",
    epochs:     int = 30,
    lr:         float = 1e-4,
    batch_size: int = 64,
    patience:   int = 7,        # early stopping patience
    pos_weight: float = 1.0,    # increase if dataset is imbalanced
    save_best:  bool = True,
):
    """
    Train a dementia risk model.

    Args:
      model_name  : "transformer", "cnn", or "bilstm"
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

    model = get_model(model_name, meta).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}\n")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE)
    )


    #TODO: CHANGE LOSS AND OPTIMIZER!!
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"best_{model_name}.pt"

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      DEVICE)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"| Train loss: {train_loss:.4f}  acc: {train_acc:.3f}"
            f"| Val loss: {val_loss:.4f}  acc: {val_acc:.3f}"
            f"| {elapsed:.1f}s"
        )

        # Early stopping + best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_best:
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(),
                     "val_loss": val_loss, "val_acc": val_acc, "meta": meta},
                    ckpt_path,
                )
                print(f"  ✓ Saved best checkpoint → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if save_best and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print(f"\nLoaded best checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, DEVICE)
    print(f"\nTest  loss: {test_loss:.4f}  |  Test  acc: {test_acc:.3f}")

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
    ckpt  = torch.load(ckpt_path, map_location=device)
    meta  = ckpt["meta"]
    model = get_model(model_name, meta).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {model_name} from epoch {ckpt['epoch']}")
    return model, meta



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a dementia risk model on MIMIC-IV")
    parser.add_argument("--model",      type=str,   default="transformer",
                        choices=["transformer", "cnn", "bilstm"],
                        help="Which model architecture to train")
    parser.add_argument("--epochs",     type=int,   default=30)
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