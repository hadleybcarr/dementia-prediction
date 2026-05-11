import torch
import torch.nn as nn
import torch.nn.functional as F


class ResTCNBlock(nn.Module):
    def __init__(self, channels:int, dilation: int, dropout:float=0.1, kernel_size:int=3):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout(dropout)
        self.pool1 = nn.AdaptiveAvgPool1d()
        self.pool2 = nn.AdaptiveAvgPool1d()
        self.act = nn.GELU()

    def forward(self,x):
        r = x
        x = self.pool1(self.act(self.norm1(self.conv1(x))))
        x = self.drop(x)
        x = self.pool2(self.act(self.norm2(self.conv2(x))))
        x = self.drop(x)
        return r + x 

class DementiaCNN(nn.Module):
    def __init__(self, n_features=14, n_timesteps=24, n_demo: int=2, mask_idx=None, channels:int=96, dilations=(1,2,4), dropout:float=0.2):
        super().__init__()
        self.n_features = n_features
        self.n_demo = n_demo
        self.n_temporal = n_features - n_demo       
        self.mask_idx = mask_idx

        self.stem == nn.Sequential(
            nn.Conv1d(self.n_temporal, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(*[
            ResTCNBlock(channels, dilation=d, dropout=dropout) for d in dilations
        ])

        demo_dim = 32
        self.demo_mlp = nn.Sequential(
            nn.Linear(n_demo, demo_dim), 
            nn.GELU(),
            nn.Linear(demo_dim, demo_dim)
        )

        fused_dim = 2*channels + demo_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128,1),
        )

    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 14)
        temporal = x[:, :, :self.n_temporal].permute(0, 2, 1)   # (B, 12, T)
        demo     = x[:, 0, self.n_temporal:]                    # (B, 2) — same at every t
        h = self.stem(temporal)
        h = self.blocks(h)
        demo_h = self.demo_mlp(demo)
        out = self.head(torch.cat([h, demo_h]), dim=-1)
        return out.squeeze(-1)


def build_cnn(meta:dict):
    seq_length = meta["seq_len"]
    num_vitals = meta["n_vitals"]
    #label_df = meta["label_df"]
    return DementiaCNN(n_features=num_vitals, n_demo=2, n_timesteps=24, channels=96, dilations=(1,2,4), dropout=0.2)

if __name__ == "__main__":
    model = DementiaCNN(n_vitals=5, seq_len=24)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    vitals = torch.randn(8, 48, 6)
    out    = model(vitals)
    print(f"Output shape: {out.shape}")    # (8,)
    print(f"Sample logits: {out[:3].detach()}")