from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionConfig


def build_sparse_indices(
    n: int,
    window_size: int,
    expander_degree: int,
    head_idx: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build sparse attention indices combining local window + expander edges.

    Combines:
    1. Local window: [q-window_size, ..., q-1, q] (dense local context)
    2. Expander edges: d long-range neighbors using modular arithmetic

    Args:
        n: Sequence length (must be >= 0)
        window_size: Size of local sliding window (must be >= 1)
        expander_degree: Number of long-range expander neighbors (must be >= 0)
        head_idx: Head index for per-head variation (deterministic seed)
        device: Device to place indices on

    Returns:
        indices: [n, window_size + expander_degree] tensor with attention indices

    Raises:
        ValueError: If window_size < 1 or expander_degree < 0
    """
    # Input validation
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if expander_degree < 0:
        raise ValueError(f"expander_degree must be >= 0, got {expander_degree}")
    if n < 0:
        raise ValueError(f"sequence length n must be >= 0, got {n}")

    target_degree = window_size + expander_degree
    indices_list = []

    for q in range(n):
        # 1. Local window: [q-window_size, ..., q-1, q]
        window_neighbors = []
        window_start = max(0, q - window_size + 1)
        for i in range(window_start, q + 1):
            window_neighbors.append(i)

        # 2. Expander edges using modular arithmetic with per-head offset
        expander_neighbors = []
        if n > 1:
            mod = n  # Full-range modulus for better coverage
            head_offset = (head_idx * 7) % mod  # Prime-like multiplier for variation

            for s in range(1, expander_degree + 1):
                offset = ((s * s) + head_offset) % mod
                k = q - offset
                # Note: When offset == 0, k == q (self-attention), which is deduped by set
                if k >= 0 and k not in window_neighbors:
                    expander_neighbors.append(k)

        # Combine: window first, then expander
        neighbors_list = window_neighbors + expander_neighbors

        # Refill strategy: pad with additional window positions if needed
        if len(neighbors_list) < target_degree:
            neighbors_set = set(neighbors_list)
            refill_start = max(0, q - window_size - expander_degree)
            refill_end = max(0, q - window_size + 1)

            for i in range(refill_start, refill_end):
                if i not in neighbors_set and len(neighbors_list) < target_degree:
                    neighbors_list.append(i)
                    neighbors_set.add(i)

        # Truncate to exact degree and pad with -1 (invalid)
        neighbors_list = neighbors_list[:target_degree]
        while len(neighbors_list) < target_degree:
            neighbors_list.append(-1)

        indices_list.append(neighbors_list)

    # Handle T==0 edge case: ensure shape is (0, target_degree) not (0,)
    if n == 0:
        indices = torch.empty((0, target_degree), dtype=torch.long, device=device)
    else:
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
        self.window_size = max(1, cfg.window_size or 64)  # Ensure window_size >= 1
        self.expander_degree = cfg.expander_degree or 8
        self.indices_cache: dict[tuple, torch.Tensor] = {}
        self.last_attn_weights: torch.Tensor | None = None  # Store for metrics
        self.last_attn_entropy: float = 0.0
        self.last_attn_entropy_normalized: float = 0.0
        self.last_valid_neighbor_fraction: float = 0.0
        self.last_attention_mass_window_pct: float = 0.0
        self.last_attention_mass_expander_pct: float = 0.0

    def _get_indices(self, n: int, h: int, device: torch.device | str) -> torch.Tensor:
        """Get or build sparse indices for a given sequence length and head.

        Args:
            n: Sequence length
            h: Head index (for per-head variation)
            device: Device to place indices on

        Returns:
            indices: [n, window_size + expander_degree] tensor

        Note:
            Cache key uses str(device) which is stable for q.device but may create
            extra entries if device is passed as different string representations
            (e.g., "cuda" vs "cuda:0"). In practice, q.device is consistent.
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
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute sparse attention.

        Args:
            q, k, v: [B, H, T, Dh] query, key, value tensors
            attn_mask: Optional [B, T] (key padding) or [B, H, T, T] (full mask)
                       True=keep, False=mask out

        Returns:
            [B, H, T, Dh] attention output
        """
        B, H, T, Dh = q.shape
        device = q.device

        # Build indices for each head (per-head variation)
        indices_per_head = torch.stack([self._get_indices(T, h, device) for h in range(H)], dim=0)
        degree = indices_per_head.shape[-1]

        # Expand indices for batch: [H, T, degree] -> [B, H, T, degree]
        indices_expanded = indices_per_head[None, :, :, :].expand(B, H, T, degree)

        # Prepare for gather: reshape to [B*H, T, degree]
        k_flat = k.reshape(B * H, T, Dh)
        v_flat = v.reshape(B * H, T, Dh)
        indices_flat = indices_expanded.reshape(B * H, T, degree)
        indices_clamped = indices_flat.clamp(min=0)

        # Gather K and V: [B*H, T, degree, Dh]
        # Note: expand().gather() pattern is correct but memory-bandwidth heavy at long T.
        # Future optimization: consider block-sparse kernels for very long sequences.
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

        # Apply validity mask for invalid indices (-1)
        validity_mask = (indices_flat >= 0).reshape(B, H, T, degree)
        scores = scores.masked_fill(~validity_mask, float("-inf"))

        # Apply padding/segment mask if provided
        # Note: Masking strategy is safe because:
        # 1. Invalid indices (-1) are clamped to 0 for gather (no out-of-bounds)
        # 2. Validity mask applied first to -inf (invalid slots stay dead)
        # 3. Padding/full masks applied per gathered neighbor (correct for sparse)
        # 4. Softmax produces 0 weight for -inf slots (no NaN propagation)
        if attn_mask is not None:
            scores = self._apply_attention_mask(
                scores, attn_mask, indices_clamped, indices_expanded, B, H, T, degree
            )

        # Softmax over neighbors
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)

        # Store for metrics (detached to avoid graph retention)
        self.last_attn_weights = attn_weights.detach()

        # Compute and store sparse attention metrics
        self._compute_attention_metrics(attn_weights.detach())

        # Aggregate values: [B, H, T, Dh]
        output = torch.einsum("bhtp,bhtpd->bhtd", attn_weights, v_sparse)

        return output

    def _apply_attention_mask(
        self,
        scores: torch.Tensor,
        attn_mask: torch.Tensor,
        indices_clamped: torch.Tensor,
        indices_expanded: torch.Tensor,
        B: int,
        H: int,
        T: int,
        degree: int,
    ) -> torch.Tensor:
        """Apply attention mask to scores.

        Args:
            scores: [B, H, T, degree] attention scores
            attn_mask: [B, T] or [B, H, T, T] attention mask (True=keep)
            indices_clamped: [B*H, T, degree] clamped indices
            indices_expanded: [B, H, T, degree] expanded indices
            B, H, T, degree: Batch, head, time, degree dimensions

        Returns:
            scores: [B, H, T, degree] masked scores
        """
        if attn_mask.dim() == 2:  # Key padding mask: [B, T]
            return self._apply_key_padding_mask(scores, attn_mask, indices_clamped, B, H, T, degree)
        elif attn_mask.dim() == 4:  # Full mask: [B, H, T, T] or [B, 1, T, T]
            return self._apply_full_mask(scores, attn_mask, indices_expanded, B, H, T, degree)
        else:
            raise ValueError(f"attn_mask must be 2D or 4D, got {attn_mask.dim()}D")

    def _apply_key_padding_mask(
        self,
        scores: torch.Tensor,
        attn_mask: torch.Tensor,
        indices_clamped: torch.Tensor,
        B: int,
        H: int,
        T: int,
        degree: int,
    ) -> torch.Tensor:
        """Apply key padding mask to scores.

        Args:
            scores: [B, H, T, degree] attention scores
            attn_mask: [B, T] key padding mask (True=keep)
            indices_clamped: [B*H, T, degree] clamped indices
            B, H, T, degree: Batch, head, time, degree dimensions

        Returns:
            scores: [B, H, T, degree] masked scores
        """
        key_ok = attn_mask.to(torch.bool)  # [B, T]
        key_ok = key_ok[:, None, :].expand(B, H, T).reshape(B * H, T)  # [B*H, T]

        # Gather mask for sparse neighbors
        key_ok_neighbors = torch.gather(
            key_ok[:, :, None].expand(B * H, T, degree), dim=1, index=indices_clamped
        )  # [B*H, T, degree]
        key_ok_neighbors = key_ok_neighbors.reshape(B, H, T, degree)

        return scores.masked_fill(~key_ok_neighbors, float("-inf"))

    def _apply_full_mask(
        self,
        scores: torch.Tensor,
        attn_mask: torch.Tensor,
        indices_expanded: torch.Tensor,
        B: int,
        H: int,
        T: int,
        degree: int,
    ) -> torch.Tensor:
        """Apply full attention mask to scores.

        Args:
            scores: [B, H, T, degree] attention scores
            attn_mask: [B, H, T, T] or [B, 1, T, T] full mask (True=keep)
            indices_expanded: [B, H, T, degree] expanded indices
            B, H, T, degree: Batch, head, time, degree dimensions

        Returns:
            scores: [B, H, T, degree] masked scores

        Raises:
            ValueError: If mask head dimension is neither 1 nor H
        """
        mask4 = attn_mask.to(torch.bool)

        # Validate and handle head dimension
        mask_h = mask4.shape[1]
        if mask_h == 1 and H > 1:
            # Expand [B, 1, T, T] to [B, H, T, T]
            mask4 = mask4.expand(B, H, T, T)
        elif mask_h != H:
            # Explicit check: mask head dim must be 1 or H
            raise ValueError(
                f"attn_mask head dimension must be 1 or {H}, got {mask_h}. Shape: {attn_mask.shape}"
            )

        # Gather mask for sparse neighbors
        idx4 = indices_expanded.clamp(min=0)  # [B, H, T, degree]
        key_ok_neighbors = torch.gather(mask4, 3, idx4)  # [B, H, T, degree]

        return scores.masked_fill(~key_ok_neighbors, float("-inf"))

    def _compute_attention_metrics(self, attn_weights: torch.Tensor) -> None:
        """Compute and store attention metrics from weights.

        Args:
            attn_weights: [B, H, T, degree] attention weights (detached)
        """
        import math

        if attn_weights.numel() == 0:
            self.last_attn_entropy = 0.0
            self.last_attn_entropy_normalized = 0.0
            self.last_valid_neighbor_fraction = 0.0
            self.last_attention_mass_window_pct = 0.0
            self.last_attention_mass_expander_pct = 0.0
            return

        degree = attn_weights.shape[-1]

        # Entropy: -sum(p * log(p))
        attn_weights_safe = torch.clamp(attn_weights, min=1e-10)
        entropy = -(attn_weights * torch.log(attn_weights_safe)).sum(dim=-1).mean().item()
        max_entropy = math.log(degree) if degree > 1 else 1.0
        entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        # Valid neighbor fraction: effective_degree / degree
        eff_degree = (attn_weights > 0).sum(dim=-1).float().mean().item()
        valid_neighbor_fraction = eff_degree / degree if degree > 0 else 0.0

        # Attention mass split: window vs expander
        window_mass = attn_weights[..., : self.window_size].sum(dim=-1).mean().item()
        expander_mass = attn_weights[..., self.window_size :].sum(dim=-1).mean().item()
        total = window_mass + expander_mass
        if total > 0:
            window_pct = 100.0 * window_mass / total
            expander_pct = 100.0 * expander_mass / total
        else:
            window_pct = expander_pct = 0.0

        self.last_attn_entropy = entropy
        self.last_attn_entropy_normalized = entropy_normalized
        self.last_valid_neighbor_fraction = valid_neighbor_fraction
        self.last_attention_mass_window_pct = window_pct
        self.last_attention_mass_expander_pct = expander_pct
