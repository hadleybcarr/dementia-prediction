import torch
import torch.nn as nn
import torch.nn.functional as F


class DementiaCNN(nn.Module):

    def __init__(self, n_features=6, n_timesteps=48):
        super().__init__()
        self.vital_conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(64,128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.25), 
            nn.Linear(128,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0,2,1)
        return self.vital_conv(x).squeeze(-1)


def build_cnn(meta:dict):
    seq_length = meta["seq_len"]
    num_vitals = meta["n_vitals"]
    label_df = meta["label_df"]
    return DementiaCNN(n_features=num_vitals, n_timesteps=48)

if __name__ == "__main__":
    model = DementiaCNN(n_vitals=6, seq_len=48)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    vitals = torch.randn(8, 48, 6)
    out    = model(vitals)
    print(f"Output shape: {out.shape}")    # (8,)
    print(f"Sample logits: {out[:3].detach()}")