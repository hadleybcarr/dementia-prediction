"""
transformer_model.py
====================
Transformer encoder for dementia risk prediction.

Architecture:
  1. Linear input projection: vital signs → d_model
  2. Learnable positional encoding
  3. N × TransformerEncoderLayer (multi-head self-attention + FFN)
  4. [CLS] token pooling
  5. ICD embedding: Linear(n_icd_codes → icd_dim)
  6. Concatenate [CLS] + ICD → classifier head → sigmoid

Input:
  vitals : (batch, seq_len, n_vitals)
  icd    : (batch, n_icd_codes)

Output:
  (batch,) — probability of dementia
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class DementiaTransformer(nn.Module):
    """
    Transformer-based dementia risk classifier.

    Args:
      n_vitals     : number of vital sign channels (default 6)
      d_model      : internal embedding dimension (default 128)
      n_heads      : number of attention heads (default 8)
      n_layers     : number of encoder layers (default 4)
      dim_ff       : feed-forward hidden size (default 256)
      dropout      : dropout probability (default 0.1)
      seq_len      : sequence length — used for positional encoding (default 48)
    """

    def __init__(
        self,
        n_vitals:    int = 6,
        d_model:     int = 128,
        n_heads:     int = 8,
        n_layers:    int = 4,
        dim_ff:      int = 256,
        icd_dim:     int = 64,
        dropout:     float = 0.1,
        seq_len:     int = 48,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_vitals, d_model),
            nn.LayerNorm(d_model),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.icd_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(icd_dim * 2, icd_dim),
            nn.LayerNorm(icd_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model + icd_dim, 128),
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
          logits : (batch,)  — raw scores (apply sigmoid for probabilities)
        """
        batch = vitals.size(0)

        # Project vitals to d_model
        x = self.input_proj(vitals)                       # (B, T, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch, -1, -1)        # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                  # (B, T+1, d_model)

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer encoder
        x = self.encoder(x)                               # (B, T+1, d_model)

        # Extract [CLS] representation
        cls_out = x[:, 0, :]                              # (B, d_model)

        # Embed ICD codes
        icd_out = self.icd_embed(icd)                     # (B, icd_dim)

        # Concatenate and classify
        combined = torch.cat([cls_out, icd_out], dim=-1)  # (B, d_model + icd_dim)
        logits   = self.classifier(combined).squeeze(-1)  # (B,)
        return logits


def build_transformer(meta: dict, **kwargs) -> DementiaTransformer:
    """
    Convenience constructor using metadata returned by get_dataloaders().

    Example:
      model = build_transformer(meta, d_model=128, n_layers=4)
    """
    return DementiaTransformer(
        n_vitals    = meta["n_vitals"],
        n_icd_codes = meta["n_icd_codes"],
        seq_len     = meta["seq_len"],
        **kwargs,
    )


if __name__ == "__main__":
    model = DementiaTransformer(n_vitals=6, n_icd_codes=200, seq_len=48)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    vitals = torch.randn(8, 48, 6)
    icd    = torch.randint(0, 2, (8, 200)).float()
    out    = model(vitals, icd)
    print(f"Output shape: {out.shape}")    # (8,)
    print(f"Sample logits: {out[:3].detach()}")