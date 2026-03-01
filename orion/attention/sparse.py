from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionConfig


def build_sparse_indices(
    n: int, window_size: int, expander_degree: int, head_idx: int, device: str
) -> torch.Tensor:
    """Build sparse attention indices combining local window + expander edges.

    Combines:
    1. Local window: [q-window_size, ..., q-1, q] (dense local context)
    2. Expander edges: d long-range neighbors using modular arithmetic

    Args:
        n: Sequence length
        window_size: Size of local sliding window
        expander_degree: Number of long-range expander neighbors
        head_idx: Head index for per-head variation (deterministic seed)
        device: Device to place indices on

    Returns:
        indices: [n, window_size + expander_degree] tensor with attention indices
    """
    indices_list = []

    for q in range(n):
        neighbors = set()

        # 1. Local window: [q-window_size, ..., q-1, q]
        for i in range(max(0, q - window_size + 1), q + 1):
            neighbors.add(i)

        # 2. Expander edges using modular arithmetic with per-head offset
        # Use quadratic residues mod a prime-like structure for regularity
        if n > 1:
            # Choose a modulus close to n for better distribution
            mod = max(2, n // 2)
            # Per-head offset for variation
            head_offset = (head_idx * 7) % mod  # Use prime-like multiplier
            for s in range(1, expander_degree + 1):
                # Quadratic residue with per-head offset: (s * s + head_offset) mod mod
                offset = ((s * s) + head_offset) % mod
                k = q - offset
                if k >= 0:
                    neighbors.add(k)

        # Convert to sorted list and pad/truncate to exact degree
        neighbors = sorted(list(neighbors), reverse=True)  # Sort descending for locality
        target_degree = window_size + expander_degree

        # Dedup and refill strategy: keep unique neighbors, pad with window if needed
        if len(neighbors) < target_degree:
            # Pad with additional window positions if not enough unique neighbors
            for i in range(max(0, q - window_size - expander_degree), max(0, q - window_size + 1)):
                if i not in neighbors and len(neighbors) < target_degree:
                    neighbors.append(i)
            neighbors = sorted(neighbors, reverse=True)

        # Truncate to exact degree
        neighbors = neighbors[:target_degree]

        # Pad with -1 (invalid) if still not enough
        while len(neighbors) < target_degree:
            neighbors.append(-1)

        indices_list.append(neighbors)

    indices = torch.tensor(indices_list, dtype=torch.long, device=device)
    return indices


class SparseAttention:
    """Sparse attention with local window + structured long-range expander edges.

    Combines:
    - Local window: dense context for short-range dependencies
    - Expander edges: modular arithmetic for long-range connectivity
    - Per-head variation: different seeds per head for diverse sparse patterns
    - Proper masking: respects padding and causality
    - Deduplication: ensures consistent degree across tokens

    Complexity: O(T * (W + d) * Dh) where W=window_size, d=expander_degree
    """

    def __init__(self, cfg: AttentionConfig):
        self.cfg = cfg
        self.window_size = cfg.window_size or 64
        self.expander_degree = cfg.expander_degree or 8
        self.indices_cache: dict = {}

    def _get_indices(self, n: int, h: int, device: str) -> torch.Tensor:
        """Get or build sparse indices for a given sequence length and head.

        Args:
            n: Sequence length
            h: Head index (for per-head variation)
            device: Device

        Returns:
            indices: [n, window_size + expander_degree] tensor
        """
        cache_key = (n, h, self.window_size, self.expander_degree, str(device))
        if cache_key not in self.indices_cache:
            self.indices_cache[cache_key] = build_sparse_indices(
                n, self.window_size, self.expander_degree, h, device
            )
        return self.indices_cache[cache_key]

    def forward(
        self,
        q: torch.Tensor,  # [B, H, T, Dh]
        k: torch.Tensor,  # [B, H, T, Dh]
        v: torch.Tensor,  # [B, H, T, Dh]
        *,
        attn_mask=None,
    ) -> torch.Tensor:
        """Compute sparse attention.

        Args:
            q, k, v: [B, H, T, Dh] query, key, value tensors
            attn_mask: Optional [B, T] or [B, H, T, T] mask for padding/segments

        Returns:
            [B, H, T, Dh] attention output
        """
        B, H, T, Dh = q.shape
        device = q.device

        # Build indices for each head (per-head variation)
        # Shape: [H, T, window_size + expander_degree]
        indices_per_head = torch.stack([self._get_indices(T, h, device) for h in range(H)], dim=0)

        # Expand for batch: [B, H, T, degree]
        degree = indices_per_head.shape[-1]
        indices_expanded = indices_per_head[None, :, :, :].expand(B, H, T, degree)

        # Gather K and V using indices
        # Reshape for gather: [B*H, T, Dh] -> [B*H, T, degree, Dh]
        k_flat = k.reshape(B * H, T, Dh)
        v_flat = v.reshape(B * H, T, Dh)
        indices_flat = indices_expanded.reshape(B * H, T, degree)

        # Gather with -1 handling (invalid indices)
        # Clamp invalid indices to 0 for gather, then mask later
        indices_clamped = indices_flat.clamp(min=0)

        # Expand for gather: [B*H, T, degree, Dh]
        indices_for_gather = indices_clamped[:, :, :, None].expand(B * H, T, degree, Dh)

        k_sparse = torch.gather(
            k_flat[:, :, None, :].expand(B * H, T, degree, Dh), dim=1, index=indices_for_gather
        )
        v_sparse = torch.gather(
            v_flat[:, :, None, :].expand(B * H, T, degree, Dh), dim=1, index=indices_for_gather
        )

        # Reshape back: [B, H, T, degree, Dh]
        k_sparse = k_sparse.reshape(B, H, T, degree, Dh)
        v_sparse = v_sparse.reshape(B, H, T, degree, Dh)

        # Compute attention scores: [B, H, T, degree]
        scale = Dh**-0.5
        scores = torch.einsum("bhtd,bhtpd->bhtp", q, k_sparse) * scale

        # Create validity mask for invalid indices (-1)
        validity_mask = (indices_flat >= 0).reshape(B, H, T, degree)

        # Apply validity mask to scores
        scores = scores.masked_fill(~validity_mask, float("-inf"))

        # Apply padding mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:  # [B, T]
                attn_mask = attn_mask.unsqueeze(1).expand(B, H, T)  # [B, H, T]
            else:
                attn_mask = attn_mask  # [B, H, T, T]

            attn_mask_flat = attn_mask.reshape(B * H, T)  # [B*H, T]

            # Gather mask for sparse positions
            mask_sparse = torch.gather(
                attn_mask_flat[:, :, None].expand(B * H, T, degree), dim=1, index=indices_clamped
            )  # [B*H, T, degree]

            mask_sparse = mask_sparse.reshape(B, H, T, degree)
            scores = scores.masked_fill(~mask_sparse, float("-inf"))

        # Softmax over neighbors
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)

        # Aggregate values: [B, H, T, Dh]
        output = torch.einsum("bhtp,bhtpd->bhtd", attn_weights, v_sparse)

        return output
