from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionConfig


class DenseAttention:
    """Full causal attention — every token attends to all previous tokens.

    Implements AttentionBackend protocol. Not an nn.Module (no learnable params).
    https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html
    """

    def __init__(self, cfg: AttentionConfig):
        self.cfg = cfg

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, attn_mask=None
    ) -> torch.Tensor:
        # is_causal: lower-triangular mask, auto-dispatches to FlashAttention on GPU
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
