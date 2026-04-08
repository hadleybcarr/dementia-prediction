"""
Bidirectional LSTM with temporal attention for dementia risk prediction.

Input:
  vitals : (batch, seq_len, n_vitals)
  icd    : (batch, n_icd_codes)

Output:
  (batch,) — raw logit (apply sigmoid for probability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Soft attention over the time dimension.
    Learns a scalar weight for each time step, then takes a weighted sum
    of all hidden states.

    Input  : (batch, seq_len, hidden_dim)
    Output : (batch, hidden_dim), (batch, seq_len)  [context, attention weights]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T, H)
        scores  = self.attn(x).squeeze(-1)         # (B, T)
        weights = torch.softmax(scores, dim=-1)    # (B, T)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, H)
        return context, weights


class DementiaBiLSTM(nn.Module):
    """
    Bidirectional LSTM with temporal attention for dementia risk classification.

    Args:
      n_vitals     : number of vital sign channels (default 6)
      n_icd_codes  : size of the multi-hot ICD vector
      hidden_dim   : LSTM hidden size per direction (default 128; full BiLSTM = 256)
      n_layers     : number of stacked LSTM layers (default 3)
      icd_dim      : ICD embedding output size (default 64)
      dropout      : dropout probability (default 0.3)
    """

    def __init__(
        self,
        n_vitals:    int = 6,
        n_icd_codes: int = 200,
        hidden_dim:  int = 128,
        n_layers:    int = 3,
        icd_dim:     int = 64,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.input_proj = nn.Sequential(
            nn.Linear(n_vitals, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if n_layers > 1 else 0.0,
        )

        bilstm_out_dim = hidden_dim * 2   # bidirectional doubles output

        self.attention = TemporalAttention(bilstm_out_dim)

        self.post_attn_norm = nn.LayerNorm(bilstm_out_dim)
        self.dropout = nn.Dropout(dropout)

        self.icd_embed = nn.Sequential(
            nn.Linear(n_icd_codes, icd_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(icd_dim * 2, icd_dim),
            nn.LayerNorm(icd_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim + icd_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, vitals: torch.Tensor, icd: torch.Tensor):
        """
        Args:
          vitals : (batch, seq_len, n_vitals)
          icd    : (batch, n_icd_codes)
        Returns:
          logits : (batch,)

        Optional: also returns attention weights if you want to inspect
                  which time steps the model focused on.
        """
        # Project input
        x = self.input_proj(vitals)              # (B, T, hidden_dim)

        # BiLSTM
        lstm_out, _ = self.lstm(x)               # (B, T, hidden_dim*2)

        # Temporal attention pooling
        context, attn_weights = self.attention(lstm_out)  # (B, hidden_dim*2)

        context = self.post_attn_norm(context)
        context = self.dropout(context)

        # ICD embedding
        icd_out = self.icd_embed(icd)            # (B, icd_dim)

        # Concatenate and classify
        combined = torch.cat([context, icd_out], dim=-1)
        logits   = self.classifier(combined).squeeze(-1)   # (B,)
        return logits

    def forward_with_attention(self, vitals: torch.Tensor, icd: torch.Tensor):
        """
        Same as forward(), but also returns attention weights for interpretability.

        Returns:
          logits       : (batch,)
          attn_weights : (batch, seq_len) — which time steps the model focused on
        """
        x = self.input_proj(vitals)
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        context = self.post_attn_norm(context)
        context = self.dropout(context)
        icd_out = self.icd_embed(icd)
        combined = torch.cat([context, icd_out], dim=-1)
        logits   = self.classifier(combined).squeeze(-1)
        return logits, attn_weights


def build_bilstm(meta: dict, **kwargs) -> DementiaBiLSTM:
    """
    Convenience constructor using metadata returned by get_dataloaders().

    Example:
      model = build_bilstm(meta, hidden_dim=128, n_layers=3)
    """
    return DementiaBiLSTM(
        n_vitals    = meta["n_vitals"],
        n_icd_codes = meta["n_icd_codes"],
        **kwargs,
    )


if __name__ == "__main__":
    model = DementiaBiLSTM(n_vitals=6, n_icd_codes=200)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Smoke test
    vitals = torch.randn(8, 48, 6)
    icd    = torch.randint(0, 2, (8, 200)).float()

    logits = model(vitals, icd)
    print(f"Output shape: {logits.shape}")    # (8,)

    logits, weights = model.forward_with_attention(vitals, icd)
    print(f"Attention weights shape: {weights.shape}")   # (8, 48)
    print(f"Attention sums (should ≈ 1): {weights.sum(dim=-1)[:3].detach()}")