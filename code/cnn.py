import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, residual: bool = False):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU()
        self.residual = residual
        self.proj = nn.Conv1d(in_ch, out_ch, 1, bias=False) if residual and in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.conv(x)))
        if self.residual:
            res = self.proj(x) if self.proj is not None else x
            out = out + res
        return out




class DementiaCNN(nn.Module):
    """
    Args:
      n_icd_codes  : size of the multi-hot ICD vector
      hidden_ch    : channels in the shared conv block after concatenation (default 128)
      icd_dim      : ICD embedding output size (default 64)
      dropout      : dropout probability (default 0.3)
    """

    def __init__(
        self,
        n_icd_codes: int = 200,
        hidden_ch:   int = 128,
        icd_dim:     int = 64,
        dropout:     float = 0.3,
    ):
        super().__init__()


        self.conv = nn.Sequential(
            ConvBlock(128, hidden_ch, kernel=3, residual=True),
            ConvBlock(hidden_ch, hidden_ch, kernel=3, residual=True),
            nn.Dropout(dropout),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.icd_embed = nn.Sequential(
            nn.Linear(n_icd_codes, icd_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(icd_dim * 2, icd_dim),
            nn.LayerNorm(icd_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_ch + icd_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, vitals: torch.Tensor, icd: torch.Tensor) -> torch.Tensor:
        """
        Args:
          vitals : (batch, seq_len, n_vitals)
          icd    : (batch, n_icd_codes)
        Returns:
          logits : (batch,)
        """
        # Conv1d expects (batch, channels, length)
        x = vitals.permute(0, 2, 1)             # (B, n_vitals, T)


        # Align lengths (branches may differ by 1 due to padding/pooling)
        min_len = min(s.size(2), m.size(2), l.size(2))
        x = torch.cat([s[:, :, :min_len],
                        m[:, :, :min_len],
                        l[:, :, :min_len]], dim=1)   # (B, 3*branch_ch, T')

        x = self.conv(x)                  # (B, hidden_ch, T')
        x = self.gap(x).squeeze(-1)              # (B, hidden_ch)

        # ICD embedding
        icd_out = self.icd_embed(icd)            # (B, icd_dim)

        # Classify
        combined = torch.cat([x, icd_out], dim=-1)
        logits   = self.classifier(combined).squeeze(-1)   # (B,)
        return logits


def build_cnn(meta: dict, **kwargs) -> DementiaCNN:
    """
    Convenience constructor using metadata returned by get_dataloaders().

    Example:
      model = build_cnn(meta, branch_ch=64, hidden_ch=128)
    """
    return DementiaCNN(
        n_vitals    = meta["n_vitals"],
        n_icd_codes = meta["n_icd_codes"],
        **kwargs,
    )


if __name__ == "__main__":
    model = DementiaCNN(n_vitals=6, n_icd_codes=200)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    vitals = torch.randn(8, 48, 6)
    icd    = torch.randint(0, 2, (8, 200)).float()
    out    = model(vitals, icd)
    print(f"Output shape: {out.shape}")    # (8,)
    print(f"Sample logits: {out[:3].detach()}")