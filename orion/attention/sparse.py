from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionConfig


def build_orion_sparse_indices(n: int, p: int, device: str):
    """Build Ramanujan-style sparse attention indices.

    Each query q attends to: [q, q-1², q-2², ..., q-p²] (clipped to [0, q])

    Args:
        n: Sequence length
        p: Number of expander neighbors per query
        device: Device to place indices on

    Returns:
        indices: [n, p+1] tensor with attention indices
    """
    idx = torch.zeros(n, p + 1, dtype=torch.long, device=device)
    for q in range(n):
        row = [q]  # Self-attention
        for s in range(1, p + 1):
            k = q - (s * s)  # Quadratic residue
            row.append(k if k >= 0 else 0)  # Clamp to 0
        idx[q] = torch.tensor(row, dtype=torch.long, device=device)
    return idx


class OrionSparseAttention:
    """True-sparse attention using Ramanujan graph structure.

    Combines local context with long-range connections via quadratic residues.
    Complexity: O(T*p*Dh) instead of O(T²*Dh).
    """

    def __init__(self, cfg: AttentionConfig):
        self.cfg = cfg
        self.p = cfg.expander_degree or 8  # Number of neighbors
        self.indices_cache: dict = {}

    def _get_indices(self, n: int, device: str) -> torch.Tensor:
        """Get or build sparse attention indices."""
        cache_key = (n, self.p, str(device))
        if cache_key not in self.indices_cache:
            self.indices_cache[cache_key] = build_orion_sparse_indices(n, self.p, device)
        return self.indices_cache[cache_key]

    def forward(
        self,
        q: torch.Tensor,  # [B, H, T, Dh]
        k: torch.Tensor,  # [B, H, T, Dh]
        v: torch.Tensor,  # [B, H, T, Dh]
        *,
        attn_mask=None,
    ) -> torch.Tensor:
        """Compute sparse attention using gather-based computation.

        Args:
            q, k, v: [B, H, T, Dh] query, key, value tensors
            attn_mask: Ignored (uses Ramanujan structure)

        Returns:
            [B, H, T, Dh] attention output
        """
        B, H, T, Dh = q.shape
        device = q.device

        # Get sparse indices [T, p+1]
        idx = self._get_indices(T, device)
        p_eff = idx.shape[1]  # p+1 (self + p neighbors)

        # Expand indices for gather: [B, H, T, p+1, Dh]
        idx_expanded = idx[None, None, :, :, None].expand(B, H, T, p_eff, Dh)

        # Gather keys and values for sparse positions
        k_expanded = k[:, :, :, None, :].expand(B, H, T, p_eff, Dh)
        v_expanded = v[:, :, :, None, :].expand(B, H, T, p_eff, Dh)

        k_sparse = torch.gather(k_expanded, dim=2, index=idx_expanded)  # [B, H, T, p+1, Dh]
        v_sparse = torch.gather(v_expanded, dim=2, index=idx_expanded)  # [B, H, T, p+1, Dh]

        # Compute sparse attention scores: [B, H, T, p+1]
        scale = Dh**-0.5
        scores = torch.einsum("bhtd,bhtpd->bhtp", q, k_sparse) * scale

        # Softmax over sparse neighbors
        attn_weights = F.softmax(scores, dim=-1)

        # Aggregate values: [B, H, T, Dh]
        output = torch.einsum("bhtp,bhtpd->bhtd", attn_weights, v_sparse)

        return output
