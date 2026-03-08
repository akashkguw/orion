from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class AttentionConfig:
    backend: str  # "dense" | "window" | "sparse"
    window_size: int | None = None  # W (local window size)
    expander_degree: int | None = None  # d (number of long-range expander neighbors)
    # Sparse backend implementation:
    # - "auto": use fused sparse kernel when available, else gather path
    # - "gather": always use explicit gather/scatter sparse path
    # - "flex": require fused sparse kernel path
    sparse_impl: str = "flex"
    # Block size used by torch flex_attention block masks
    sparse_block_size: int = 128
    # Optional probe metrics for fused sparse path:
    # every N forward calls, run a small gather probe to estimate entropy/mass.
    sparse_probe_every: int = 0
    sparse_probe_tokens: int = 256
    # Optional probe metrics for window backend:
    # every N forward calls, run a bounded probe to estimate entropy/score.
    window_probe_every: int = 50
    window_probe_tokens: int = 256


class AttentionBackend(Protocol):
    def forward(
        self,
        q: torch.Tensor,  # [B, H, T, Dh]
        k: torch.Tensor,  # [B, H, T, Dh]
        v: torch.Tensor,  # [B, H, T, Dh]
        *,
        attn_mask: Any | None = None,  # backend-specific (indices/COO/etc.)
    ) -> torch.Tensor:  # [B, H, T, Dh]
        ...


def build_attention_backend(cfg: AttentionConfig) -> AttentionBackend:
    """
    Factory used by the model. Must return an object implementing AttentionBackend.
    """
    backend = cfg.backend.lower()
    if backend == "dense":
        from .dense import DenseAttention

        return DenseAttention(cfg)
    if backend == "window":
        from .window import WindowAttention

        return WindowAttention(cfg)
    if backend == "sparse":
        from .sparse import SparseAttention

        return SparseAttention(cfg)
    raise ValueError(f"Unknown attention backend: {cfg.backend}")
