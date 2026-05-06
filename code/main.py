import numpy as np
import torch 
import matplotlib.pyplot as plt
import argparse
import json 

def pad_to_length(arr, target_len):
    return arr + [np.nan] * (target_len - len(arr))

def all_graphs(graph_type:str, graph_title):
    with open("history.json") as f:
        h = json.load(f)
        #print(h)
    
    epochs = range(1, 15+1)
    plt.figure(figsize=(8,5))

    for model in ["cnn", "transformer", "bilstm"]:
        values = h[model][graph_type]
        if len(values) == 0:
            continue
        padded = pad_to_length(values, 30)
        plt.plot(epochs, padded, label=f"{model.upper()} {graph_title}")

    plt.xlabel("Epoch")
    plt.ylabel(graph_title)
    plt.title("All Models "+graph_title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{graph_type}.png", dpi=150)
    plt.close()
     

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
