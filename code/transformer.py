"""
transformer_model.py
====================
Transformer encoder for dementia risk prediction.

Architecture:
  1. Linear input projection: vital signs → d_model
  2. Learnable positional encoding
  3. N × TransformerEncoderLayer (multi-head self-attention + FFN)
  4. [CLS] token pooling

Input:
  vitals : (batch, seq_len, n_vitals)

Output:
  (batch,) — probability of dementia
"""

import math
import torch
import torch.nn as nn

class ConvStem(nn.Module):
    def __init__(self, in_ch:int, d_model:int, kernel_size:int=3, dropout:float=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d_model, kernel_size, padding=pad),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size, padding=pad),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.net(x)
        return x.transpose(1,2)
    
class AttentionPool(nn.Module):
    def __init__(self, d_model:int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1,1,d_model))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

    def forward(self, x, key_padding_mask=None):
        q = self.q.expand(x.size(0), -1,-1)
        out, _ = self.attn(q, x,x,key_padding_mask=None, need_weights=False)
        return out.squeeze(1)
    
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
      n_vitals     : number of vital sign channels (default 5)
      d_model      : internal embedding dimension (default 128)
      n_heads      : number of attention heads (default 8)
      n_layers     : number of encoder layers (default 4)
      dim_ff       : feed-forward hidden size (default 256)
      dropout      : dropout probability (default 0.1)
      seq_len      : sequence length — used for positional encoding (default 48)
    """

    def __init__(self,
        n_vitals:int = 5,n_demo: int=2,d_model:int = 96,n_heads:int = 8,n_features=6,
        n_layers:int = 3,dim_ff:int = 256,dropout:float = 0.2,seq_len:int = 24
    ):
        super().__init__()

        self.n_features = n_features
        self.n_demo = n_demo
        self.n_temporal = n_features - n_demo

        self.input_proj = nn.Sequential(
            nn.Linear(self.n_temporal, d_model),
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
            norm_first=True,      
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        demo_dim = 32
        self.demo_mlp = nn.Sequential(
            nn.Linear(n_demo, demo_dim), 
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(demo_dim, demo_dim)
        )
        
        fused_dim = 128 + demo_dim #128 is the # of channels from classifier
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128,1),
        )

    def forward(self, vitals: torch.Tensor) -> torch.Tensor:
        """
        Args:
          vitals : (batch, seq_len, n_vitals)
        Returns:
          logits : (batch,)  — raw scores (apply sigmoid for probabilities)
        """
        print("Vitals shape", vitals.shape)
        temporal = vitals[:, :, :self.n_temporal]  # (B, 12, T)
        demo     = vitals[:, 0, self.n_temporal:] 

        x = self.input_proj(temporal)                       # (B, T, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)       # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                  # (B, T+1, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)                               # (B, T+1, d_model)
        cls_out = x[:, 0, :]                              # (B, d_model)
        logits   = self.classifier(cls_out).squeeze(-1)  # (B,)
        d_logits = self.demo_mlp(demo)
        out = self.head(torch.cat([logits, d_logits], dim=-1))
        return out.squeeze(-1)



def build_transformer(meta: dict, **kwargs) -> DementiaTransformer:
    """
    Convenience constructor using metadata returned by get_dataloaders().

    Example:
      model = build_transformer(meta, d_model=128, n_layers=4)
    """
    return DementiaTransformer(
        n_vitals    = meta["n_vitals"],
        seq_len     = meta["seq_len"],
        **kwargs,
    )


if __name__ == "__main__":
    model = DementiaTransformer(n_vitals=6, seq_len=24)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    vitals = torch.randn(8, 24, 6)
    out    = model(vitals,)
    print(f"Output shape: {out.shape}")    # (8,)
    print(f"Sample logits: {out[:3].detach()}")