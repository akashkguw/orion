from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyDecoderOnly(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(4096, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_mult,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, T]
        b, t = idx.shape
        device = idx.device
        x = self.tok(idx) + self.pos(torch.arange(t, device=device))[None, :, :]
        # causal mask: [T, T] with True = masked (PyTorch expects float mask or bool depending)
        causal = torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)
        x = self.blocks(x, mask=causal)
        x = self.ln(x)
        logits = self.head(x)  # [B, T, V]
        return logits


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V], targets: [B, T]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
