import numpy as np
import torch 
import matplotlib as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dementia Prediction from EHR Data")
    parser.add_argument('--task', required=True,
                        choices=['cnn', 
                                 'bidirectional_lstm',
                                 'transformer'
                        ]
    return parser.parse_args()
    )

if __name__ == "main":
    args = parse_args()
    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available() else "cpu"
    )
    print(args.task)
    if args.task == 'cnn': 
            pass
    elif args.task == 'bidirectional_lstm':
            pass
    else:
            pass 
    