from __future__ import annotations

import torch

from .base import AttentionConfig
from .mask.builder import SparseMask


class SparseAttention:
    def __init__(self, cfg: AttentionConfig):
        self.cfg = cfg

    def forward(self, q, k, v, *, attn_mask: SparseMask | None = None) -> torch.Tensor:
        """
        Compute masked attention using SparseMask contract.
        """
        raise NotImplementedError("Not implemented")
