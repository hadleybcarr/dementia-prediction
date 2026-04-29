import numpy as np
import torch 
import matplotlib.pyplot as plt
import argparse
import json 


if __name__ == "__main__":
    print("Running your function...")
    parser = argparse.ArgumentParser(description="Train a dementia risk model on MIMIC-IV")
    parser.add_argument("--graph",      type=str,   default="transformer",
                        choices=["transformer", "cnn", "bilstm", "svm"],
                        help="Which model architecture to graph")
   
    args = parser.parse_args()
    print(args.graph)
    with open("history.json") as f:
            h = json.load(f)
            model_stats = h[args.graph]
    epochs = range(1, len(h["train_loss"]) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, model_stats["train_loss"], label="Train Loss")
    #plt.plot(epochs, h["val_acc"], label="Val acuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Dementia CNN Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("train_loss.png", dpi=150)

