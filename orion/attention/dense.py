from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionConfig


class DenseAttention:
    """Full causal attention. Implements AttentionBackend.

    https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html
    """

    def __init__(self, cfg: AttentionConfig):
        self.cfg = cfg
        self.last_attn_weights: torch.Tensor | None = None  # Store for metrics
        self.last_attn_entropy: float = 0.0
        self.last_attn_entropy_normalized: float = 0.0
        self.causal_mask_cache: dict[tuple, torch.Tensor] = {}  # Cache causal masks

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Compute attention manually to capture weights for metrics
        # q, k, v: [B, H, T, Dh]
        B, H, T, Dh = q.shape
        scale = Dh**-0.5
        device = q.device

        # Compute attention scores: [B, H, T, T]
        scores = torch.einsum("bhtd,bhsd->bhts", q, k) * scale

        # Apply causal mask (cached to avoid reallocation)
        cache_key = (T, str(device))
        if cache_key not in self.causal_mask_cache:
            causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            self.causal_mask_cache[cache_key] = causal_mask
        else:
            causal_mask = self.causal_mask_cache[cache_key]

        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Apply padding mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:  # [B, T] key padding mask (True=keep)
                key_ok = attn_mask.to(torch.bool)
                # Broadcast to [B, 1, 1, T] to mask key positions
                scores = scores.masked_fill(~key_ok[:, None, None, :], float("-inf"))
            elif attn_mask.dim() == 4:  # [B, H, T, T] or [B, 1, T, T] full mask
                mask4 = attn_mask.to(torch.bool)
                if mask4.shape[1] == 1 and H > 1:
                    mask4 = mask4.expand(B, H, T, T)
                scores = scores.masked_fill(~mask4, float("-inf"))

        # Softmax over keys
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)

        # Store for metrics (detached to avoid graph retention)
        self.last_attn_weights = attn_weights.detach()

        # Compute and store attention metrics
        self._compute_attention_metrics(attn_weights.detach())

        # Aggregate values: [B, H, T, Dh]
        output = torch.einsum("bhts,bhsd->bhtd", attn_weights, v)

        return output

    def _compute_attention_metrics(self, attn_weights: torch.Tensor) -> None:
        """Compute and store attention metrics from weights.

        Args:
            attn_weights: [B, H, T, T] attention weights (detached)
        """
        import math

        if attn_weights.numel() == 0:
            self.last_attn_entropy = 0.0
            self.last_attn_entropy_normalized = 0.0
            return

        degree = attn_weights.shape[-1]

        # Entropy: -sum(p * log(p))
        attn_weights_safe = torch.clamp(attn_weights, min=1e-10)
        entropy = -(attn_weights * torch.log(attn_weights_safe)).sum(dim=-1).mean().item()
        max_entropy = math.log(degree) if degree > 1 else 1.0
        entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        self.last_attn_entropy = entropy
        self.last_attn_entropy_normalized = entropy_normalized
