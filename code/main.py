import numpy as np
import torch 
import matplotlib as plt
import argparse
from cnn import train, test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dementia Prediction from EHR Data")
    parser.add_argument(
                        '--task', 
                        required=True,
                        choices=['cnn', 
                                 'bidirectional_lstm',
                                 'transformer'
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
    if args.task == 'cnn': 
          print("Running CNN...")
    elif args.task == 'bidirectional_lstm':
            pass
    else:
            pass 
    