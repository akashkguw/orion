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
        self.W = int(cfg.window_size) if cfg.window_size is not None else 64
        if self.W < 1:
            raise ValueError(f"window_size must be >= 1, got {self.W}")

        # Cache the mask so it's only built once per (T, device, dtype) combination
        # instead of being reallocated on every forward pass.
        self._mask_cache: dict[tuple, torch.Tensor] = {}

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, attn_mask=None
    ) -> torch.Tensor:
        """Compute causal sliding-window attention.

        Args:
            q, k, v: [B, H, T, Dh] — query, key, value tensors
            attn_mask: Optional [B, T] key padding or [B, H, T, T]/[B, 1, T, T] full mask
                using True=keep, False=mask semantics.

        Returns:
            out: [B, H, T, Dh]
        """
        B, H, T, _Dh = q.shape
        cache_key = (T, q.device, q.dtype)
        if cache_key not in self._mask_cache:
            self._mask_cache[cache_key] = _build_window_mask(
                T, self.W, device=q.device, dtype=q.dtype
            )
        window_mask = self._mask_cache[cache_key]
        combined_mask = window_mask

        if attn_mask is not None:
            extra_mask = _build_external_additive_mask(
                attn_mask=attn_mask,
                B=B,
                H=H,
                T=T,
                device=q.device,
                dtype=q.dtype,
            )
            combined_mask = window_mask + extra_mask

        return F.scaled_dot_product_attention(q, k, v, attn_mask=combined_mask)


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


def _build_external_additive_mask(
    *,
    attn_mask: torch.Tensor,
    B: int,
    H: int,
    T: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build additive mask from external attention mask with True=keep semantics."""
    if attn_mask.dim() == 2:
        # Key padding mask [B, T]: True=keep.
        if attn_mask.shape != (B, T):
            raise ValueError(
                f"2D attn_mask must have shape [{B}, {T}], got {tuple(attn_mask.shape)}"
            )
        key_ok = attn_mask.to(torch.bool)
        mask = torch.zeros((B, 1, 1, T), device=device, dtype=dtype)
        mask.masked_fill_(~key_ok[:, None, None, :], float("-inf"))
        return mask

    if attn_mask.dim() == 4:
        # Full mask [B, H, T, T] or [B, 1, T, T]: True=keep.
        if attn_mask.shape[0] != B or attn_mask.shape[2:] != (T, T):
            raise ValueError(
                f"4D attn_mask must have shape [{B}, {H}|1, {T}, {T}], got {tuple(attn_mask.shape)}"
            )
        mask4 = attn_mask.to(torch.bool)
        if mask4.shape[1] == 1 and H > 1:
            mask4 = mask4.expand(B, H, T, T)
        elif mask4.shape[1] != H:
            raise ValueError(f"4D attn_mask head dimension must be 1 or {H}, got {mask4.shape[1]}")
        mask = torch.zeros((B, H, T, T), device=device, dtype=dtype)
        mask.masked_fill_(~mask4, float("-inf"))
        return mask

    raise ValueError(f"attn_mask must be 2D or 4D, got {attn_mask.dim()}D")
