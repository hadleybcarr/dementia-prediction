import torch
import torch.nn as nn
import torch.nn.functional as F


class DementiaCNN(nn.Module):
    def __init__(self, n_features=14, n_timesteps=48):
        super().__init__()
        self.n_temporal = n_features - 2          # vitals + mask
        self.n_demo     = 2                       # age + sex

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.n_temporal, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),                         # (B, 128)
        )
        self.head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128 + self.n_demo, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 14)
        temporal = x[:, :, :self.n_temporal].permute(0, 2, 1)   # (B, 12, T)
        demo     = x[:, 0, self.n_temporal:]                    # (B, 2) — same at every t
        feat     = self.temporal_conv(temporal)                 # (B, 128)
        out      = self.head(torch.cat([feat, demo], dim=-1))   # (B, 1)
        return out.squeeze(-1)


def build_cnn(meta:dict):
    seq_length = meta["seq_len"]
    num_vitals = meta["n_vitals"]
    #label_df = meta["label_df"]
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