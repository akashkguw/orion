from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionConfig


class WindowAttention:
    """Sliding window attention with causal masking.

    Attends only to positions within a local window of size W,
    reducing complexity from O(T²) to O(T*W).
    """

    def __init__(self, cfg: AttentionConfig):
        self.cfg = cfg
        self.window_size = cfg.window_size or 64

    def forward(
        self,
        q: torch.Tensor,  # [B, H, T, Dh]
        k: torch.Tensor,  # [B, H, T, Dh]
        v: torch.Tensor,  # [B, H, T, Dh]
        *,
        attn_mask=None,
    ) -> torch.Tensor:
        """
        Compute sliding window attention with causal masking.

        Args:
            q, k, v: [B, H, T, Dh] query, key, value tensors
            attn_mask: Ignored (uses window + causal mask)

        Returns:
            [B, H, T, Dh] attention output
        """
        B, H, T, Dh = q.shape
        device = q.device

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (Dh**0.5)  # [B, H, T, T]

        # Apply window mask: only attend to positions within window_size
        window_mask = torch.ones(T, T, device=device, dtype=torch.bool)
        for i in range(T):
            window_start = max(0, i - self.window_size)
            window_mask[i, :window_start] = False
            window_mask[i, i + 1 :] = False  # Causal

        scores = scores.masked_fill(~window_mask[None, None, :, :], float("-inf"))

        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)

        output = torch.matmul(attn_weights, v)  # [B, H, T, Dh]
        return output
