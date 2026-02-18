from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionConfig


class DenseAttention:
    """Full causal attention — every token attends to all previous tokens.

    Implements AttentionBackend protocol. Not an nn.Module (no learnable params).
    """

    def __init__(self, cfg: AttentionConfig):
        self.cfg = cfg

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, attn_mask=None
    ) -> torch.Tensor:
        # is_causal applies a lower-triangular mask; also auto-dispatches to
        # FlashAttention on supported GPUs
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
