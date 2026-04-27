import numpy as np
import torch 
import matplotlib.pyplot as plt
import argparse
import json 


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dementia Prediction from EHR Data")
    parser.add_argument(
                        '--task', 
                        required=True,
                        choices=['cnn', 
                                 'bidirectional_lstm',
                                 'transformer',
                                 'graph'
                        ]
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("Running your function...")
    args = parse_args()
    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available() else "cpu"
    )
    print(args.task)
    if args.task == 'graph':
          with open("history.json") as f:
                h = json.load(f)
          epochs = range(1, len(h["train_loss"]) + 1)
          plt.figure(figsize=(8,5))
          plt.plot(epochs, h["train_loss"], label="Train Loss")
          #plt.plot(epochs, h["val_acc"], label="Val acuracy")
          plt.xlabel("Epoch")
          plt.ylabel("Accuracy")
          plt.title("Dementia CNN Training Loss")
          plt.legend()
          plt.grid(alpha=0.3)
          plt.savefig("train_loss.png", dpi=150)

    else:
            pass 
    