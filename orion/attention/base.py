from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class AttentionConfig:
    backend: str  # "dense" | "window" | "sparse"
    window_size: int | None = None  # W (local window size)
    expander_degree: int | None = None  # d (number of long-range expander neighbors)
    # add fields as needed (dropout, qk_norm toggle, etc.)


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
