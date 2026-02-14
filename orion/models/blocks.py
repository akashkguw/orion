from __future__ import annotations

import torch
import torch.nn as nn

from orion.attention.base import AttentionBackend, AttentionConfig, build_attention_backend


class DecoderBlock(nn.Module):
    """
    LN → Attention → resid → LN → MLP → resid
    """

    def __init__(self, d_model: int, n_heads: int, attn_cfg: AttentionConfig, mlp_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn: AttentionBackend = build_attention_backend(attn_cfg)

    def forward(self, x: torch.Tensor, *, attn_mask=None) -> torch.Tensor:
        raise NotImplementedError("Not implemented.")
