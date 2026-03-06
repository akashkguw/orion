from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionConfig


class WindowAttention:
    """Causal sliding-window attention. Implements AttentionBackend.

    Each token at position i attends only to tokens in [i-W+1, i] (clamped to 0).
    Restricts the receptive field to W tokens, reducing memory and compute
    from O(T²) to O(T·W) compared to dense attention.

    References:
    - Longformer: https://arxiv.org/abs/2004.05150
    - Mistral sliding window: https://arxiv.org/abs/2310.06825
    """

    def __init__(self, cfg: AttentionConfig):
        self.cfg = cfg
        self.W = cfg.window_size or 64  # default to 64 if not set in config

        # Cache the mask so it's only built once per (T, device, dtype) combination
        # instead of being reallocated on every forward pass.
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, attn_mask=None
    ) -> torch.Tensor:
        """Compute causal sliding-window attention.

        Args:
            q, k, v: [B, H, T, Dh] — query, key, value tensors
            attn_mask: ignored (mask is built internally from window_size)

        Returns:
            out: [B, H, T, Dh]
        """
        _B, _H, T, _Dh = q.shape
        cache_key = (T, q.device, q.dtype)
        if cache_key not in self._mask_cache:
            self._mask_cache[cache_key] = _build_window_mask(
                T, self.W, device=q.device, dtype=q.dtype
            )
        mask = self._mask_cache[cache_key]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def _build_window_mask(T: int, W: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a [1, 1, T, T] additive attention mask for causal sliding-window attention.

    Position i attends to j iff:
      - j <= i       (causal — no future leakage)
      - i - j < W   (within window — no tokens older than W steps)

    Returns a float mask: 0.0 where allowed, -inf where blocked.
    Shaped [1, 1, T, T] so it broadcasts over [B, H, T, T].
    """
    rows = torch.arange(T, device=device).unsqueeze(1)  # [T, 1] — query positions
    cols = torch.arange(T, device=device).unsqueeze(0)  # [1, T] — key positions

    causal = cols <= rows  # j <= i: no future tokens
    in_window = (rows - cols) < W  # i - j < W: within sliding window

    allowed = causal & in_window

    mask = torch.zeros(T, T, device=device, dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
