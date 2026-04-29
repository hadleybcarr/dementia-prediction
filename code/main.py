import numpy as np
import torch 
import matplotlib.pyplot as plt
import argparse
import json 

def all_graphs(graph_type:str, graph_title):
    with open("history.json") as f:
        h = json.load(f)
    epochs = range(1, len(h["cnn"]["train_loss"]) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, h["cnn"][graph_type], label="CNN"+graph_title)
    plt.plot(epochs, h["transformer"][graph_type], label="Transformer"+graph_title)
    plt.plot(epochs, h["bilstm"][graph_type], label="BiLSTM"+graph_title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("All Models"+graph_title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{graph_type}.png", dpi=150)
     

if __name__ == "__main__":
    print("Running your function...")
    parser = argparse.ArgumentParser(description="Train a dementia risk model on MIMIC-IV")
    parser.add_argument("--graph",      type=str,   default="transformer",
                        choices=["all", "transformer", "cnn", "bilstm", "svm"],
                        help="Which model architecture to graph")
   
    args = parser.parse_args()
    print(args.graph)
    if args.graph == "all":
        all_graphs("train_loss", "Training Loss")
        all_graphs("val_loss", "Val Loss")
        all_graphs("train_acc", "Training Accuracy")
        all_graphs("val_acc", "Validation Accuracy")
