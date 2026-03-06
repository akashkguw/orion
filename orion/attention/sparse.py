from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from .base import AttentionConfig

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    _FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    create_block_mask = None
    flex_attention = None
    _FLEX_ATTENTION_AVAILABLE = False


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
        # 1) Local window is always present and causal.
        window_start = max(0, q - window_size + 1)
        window_neighbors = list(range(window_start, q + 1))
        neighbors_list = list(window_neighbors)
        neighbors_set = set(window_neighbors)

        # 2) Expander edges:
        #    - strictly causal (k <= q-1)
        #    - never self (offset >= 1)
        #    - deterministic head-specific variation
        if expander_degree > 0 and q > 0:
            m = q  # number of strictly-past positions available
            head_offset = ((head_idx * 7) + (head_idx * head_idx * 13)) % m
            for s in range(1, expander_degree + 1):
                offset = ((s * s) + head_offset + (3 * s * head_idx)) % m
                offset += 1  # force offset in [1, q]
                k = q - offset  # now guaranteed in [0, q-1]
                if k not in neighbors_set:
                    neighbors_list.append(k)
                    neighbors_set.add(k)
                if len(neighbors_list) >= target_degree:
                    break

        # 3) Refill from older causal positions if expander produced collisions.
        #    Walk from just before the local window backwards to keep recency bias.
        if len(neighbors_list) < target_degree:
            for i in range(window_start - 1, -1, -1):
                if i not in neighbors_set:
                    neighbors_list.append(i)
                    neighbors_set.add(i)
                if len(neighbors_list) >= target_degree:
                    break

        # 4) Pad with -1 when causal history has fewer than target_degree entries.
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
        self.sparse_impl = (cfg.sparse_impl or "auto").lower()
        self.sparse_block_size = max(16, int(cfg.sparse_block_size or 128))
        self.sparse_probe_every = max(0, int(getattr(cfg, "sparse_probe_every", 0) or 0))
        self.sparse_probe_tokens = max(16, int(getattr(cfg, "sparse_probe_tokens", 256) or 256))
        self.indices_cache: dict[tuple, torch.Tensor] = {}
        self.indices_per_head_cache: dict[tuple, torch.Tensor] = {}
        self.block_mask_cache: dict[tuple, torch.Tensor] = {}
        self._fused_attention_fn = None
        self._warned_fallback = False
        self._warned_compile_fallback = False
        self._warned_probe_failure = False
        self._forward_calls = 0
        self.last_attn_weights: torch.Tensor | None = None  # Store for metrics
        self.last_attn_entropy: float = 0.0
        self.last_attn_entropy_normalized: float = 0.0
        self.last_valid_neighbor_fraction: float = 0.0
        self.last_valid_neighbor_fraction_causal_cap: float = 0.0
        self.last_valid_neighbor_fraction_vs_causal_cap: float = 0.0
        self.last_attention_mass_window_pct: float = 0.0
        self.last_attention_mass_expander_pct: float = 0.0
        self.last_total_neighbor_slots: int = 0
        self.last_valid_neighbor_slots: int = 0
        self.last_invalid_neighbor_slots: int = 0
        self.last_future_neighbor_slots: int = 0
        self.last_duplicate_neighbor_slots: int = 0

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

    def _get_indices_per_head(self, *, T: int, H: int, device: torch.device) -> torch.Tensor:
        """Get stacked sparse indices [H, T, degree], cached by shape/head/device."""
        cache_key = (T, H, self.window_size, self.expander_degree, str(device))
        if cache_key not in self.indices_per_head_cache:
            self.indices_per_head_cache[cache_key] = torch.stack(
                [self._get_indices(T, h, device) for h in range(H)], dim=0
            )
        return self.indices_per_head_cache[cache_key]

    def _compute_index_diagnostics(self, indices_per_head: torch.Tensor) -> None:
        """Compute structural diagnostics for sparse index quality.

        Args:
            indices_per_head: [H, T, degree] sparse neighbor indices
        """
        if indices_per_head.numel() == 0:
            self.last_total_neighbor_slots = 0
            self.last_valid_neighbor_slots = 0
            self.last_invalid_neighbor_slots = 0
            self.last_future_neighbor_slots = 0
            self.last_duplicate_neighbor_slots = 0
            self.last_valid_neighbor_fraction_causal_cap = 0.0
            self.last_valid_neighbor_fraction_vs_causal_cap = 0.0
            return

        H, T, degree = indices_per_head.shape
        valid = indices_per_head >= 0

        total_slots = H * T * degree
        valid_slots = int(valid.sum().item())
        invalid_slots = total_slots - valid_slots

        query_positions = torch.arange(
            T, device=indices_per_head.device, dtype=indices_per_head.dtype
        )[None, :, None]
        future_slots = int(((indices_per_head > query_positions) & valid).sum().item())

        # Count duplicate valid neighbors per [head, query] row.
        slot_idx = torch.arange(
            degree, device=indices_per_head.device, dtype=indices_per_head.dtype
        )[None, None, :]
        safe = torch.where(valid, indices_per_head, -(slot_idx + 1))
        sorted_safe, _ = torch.sort(safe, dim=-1)
        duplicate_slots = int(
            ((sorted_safe[..., 1:] == sorted_safe[..., :-1]) & (sorted_safe[..., 1:] >= 0))
            .sum()
            .item()
        )

        q_plus_one = torch.arange(1, T + 1, device=indices_per_head.device)
        cap = torch.minimum(q_plus_one, torch.tensor(degree, device=indices_per_head.device))
        valid_fraction_causal_cap = (
            float((cap.float().mean() / degree).item()) if degree > 0 else 0.0
        )
        valid_fraction = valid_slots / total_slots if total_slots > 0 else 0.0
        valid_fraction_vs_cap = (
            valid_fraction / valid_fraction_causal_cap if valid_fraction_causal_cap > 0 else 0.0
        )

        self.last_total_neighbor_slots = total_slots
        self.last_valid_neighbor_slots = valid_slots
        self.last_invalid_neighbor_slots = invalid_slots
        self.last_future_neighbor_slots = future_slots
        self.last_duplicate_neighbor_slots = duplicate_slots
        self.last_valid_neighbor_fraction_causal_cap = valid_fraction_causal_cap
        self.last_valid_neighbor_fraction_vs_causal_cap = valid_fraction_vs_cap

    def _supports_fused_sparse(self, q: torch.Tensor, attn_mask: torch.Tensor | None) -> bool:
        """Return whether we can run the fused sparse kernel for this call."""
        if self.sparse_impl == "gather":
            return False
        if not _FLEX_ATTENTION_AVAILABLE:
            return False
        if attn_mask is not None:
            # Fused path currently supports intrinsic sparse causal mask only.
            return False
        if q.device.type != "cuda":
            return False
        return True

    def _get_fused_block_mask(
        self, *, H: int, T: int, device: torch.device, indices_per_head: torch.Tensor
    ) -> torch.Tensor:
        """Get (or build) a cached BlockMask for fused sparse attention."""
        cache_key = (
            H,
            T,
            self.window_size,
            self.expander_degree,
            self.sparse_block_size,
            str(device),
        )
        if cache_key in self.block_mask_cache:
            return self.block_mask_cache[cache_key]

        # Build block-level adjacency from sparse indices:
        # [H, T, degree] -> [H, n_blocks, n_blocks].
        # This avoids allocating a dense [H, T, T] tensor in the mask builder.
        degree = indices_per_head.shape[-1]
        block_count = (T + self.sparse_block_size - 1) // self.sparse_block_size
        adjacency = torch.zeros((H, block_count, block_count), dtype=torch.bool, device=device)
        valid = indices_per_head >= 0
        if valid.any():
            h_idx = torch.arange(H, device=device, dtype=torch.long)[:, None, None].expand(
                H, T, degree
            )
            q_idx = torch.arange(T, device=device, dtype=torch.long)[None, :, None].expand(
                H, T, degree
            )
            q_block = torch.div(q_idx, self.sparse_block_size, rounding_mode="floor")
            k_block = torch.div(
                indices_per_head.clamp(min=0), self.sparse_block_size, rounding_mode="floor"
            )
            adjacency[h_idx[valid], q_block[valid], k_block[valid]] = True

        def mask_mod(b, h, q_idx, kv_idx):
            q_block = torch.div(q_idx, self.sparse_block_size, rounding_mode="floor")
            kv_block = torch.div(kv_idx, self.sparse_block_size, rounding_mode="floor")
            # Keep intrinsic causality even with block-level sparsity.
            return adjacency[h, q_block, kv_block] & (kv_idx <= q_idx)

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=H,
            Q_LEN=T,
            KV_LEN=T,
            device=device,
            BLOCK_SIZE=self.sparse_block_size,
        )
        self.block_mask_cache[cache_key] = block_mask
        return block_mask

    def _get_fused_attention_fn(self):
        """Get a compiled fused sparse attention function (cached)."""
        if self._fused_attention_fn is not None:
            return self._fused_attention_fn
        assert flex_attention is not None
        try:
            self._fused_attention_fn = torch.compile(flex_attention, dynamic=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to compile fused sparse kernel with torch.compile: {type(e).__name__}: {e}"
            ) from e
        return self._fused_attention_fn

    def _set_unavailable_weight_metrics(self) -> None:
        """Set weight-derived metrics when fused path does not expose attention weights."""
        self.last_attn_weights = None
        # Fused flex_attention does not expose per-edge attention weights, so these
        # metrics are unavailable (not zero).
        self.last_attn_entropy = float("nan")
        self.last_attn_entropy_normalized = float("nan")
        self.last_attention_mass_window_pct = float("nan")
        self.last_attention_mass_expander_pct = float("nan")
        # Preserve structural valid-neighbor diagnostics for logging.
        self.last_valid_neighbor_fraction = (
            self.last_valid_neighbor_slots / self.last_total_neighbor_slots
            if self.last_total_neighbor_slots > 0
            else 0.0
        )
        if self.last_valid_neighbor_fraction_causal_cap > 0:
            self.last_valid_neighbor_fraction_vs_causal_cap = (
                self.last_valid_neighbor_fraction / self.last_valid_neighbor_fraction_causal_cap
            )
        else:
            self.last_valid_neighbor_fraction_vs_causal_cap = 0.0

    def _compute_weight_only_metrics(
        self, attn_weights: torch.Tensor, *, window_slot_mask: torch.Tensor | None = None
    ) -> None:
        """Compute entropy and mass split from attention weights without altering index diagnostics."""
        import math

        if attn_weights.numel() == 0:
            self.last_attn_entropy = float("nan")
            self.last_attn_entropy_normalized = float("nan")
            self.last_attention_mass_window_pct = float("nan")
            self.last_attention_mass_expander_pct = float("nan")
            return

        degree = attn_weights.shape[-1]
        attn_weights_safe = torch.clamp(attn_weights, min=1e-10)
        entropy = -(attn_weights * torch.log(attn_weights_safe)).sum(dim=-1).mean().item()
        max_entropy = math.log(degree) if degree > 1 else 1.0
        entropy_normalized = entropy / max_entropy if max_entropy > 0 else float("nan")

        if window_slot_mask is None:
            window_mass = attn_weights[..., : self.window_size].sum(dim=-1).mean().item()
            expander_mass = attn_weights[..., self.window_size :].sum(dim=-1).mean().item()
        else:
            window_mask_f = window_slot_mask.to(dtype=attn_weights.dtype)
            expander_mask_f = (~window_slot_mask).to(dtype=attn_weights.dtype)
            window_mass = (attn_weights * window_mask_f).sum(dim=-1).mean().item()
            expander_mass = (attn_weights * expander_mask_f).sum(dim=-1).mean().item()

        total = window_mass + expander_mass
        if total > 0:
            window_pct = 100.0 * window_mass / total
            expander_pct = 100.0 * expander_mass / total
        else:
            window_pct = float("nan")
            expander_pct = float("nan")

        self.last_attn_entropy = entropy
        self.last_attn_entropy_normalized = entropy_normalized
        self.last_attention_mass_window_pct = window_pct
        self.last_attention_mass_expander_pct = expander_pct

    def _maybe_probe_weight_metrics(
        self, q: torch.Tensor, k: torch.Tensor, *, indices_per_head: torch.Tensor
    ) -> None:
        """Periodically estimate weight-derived metrics on a small gather probe."""
        self._forward_calls += 1
        if self.sparse_probe_every <= 0:
            return
        if self._forward_calls % self.sparse_probe_every != 0:
            return

        B, H, T, Dh = q.shape
        probe_t = min(T, self.sparse_probe_tokens)
        if probe_t <= 0:
            return

        degree = indices_per_head.shape[-1]
        try:
            with torch.no_grad():
                q_probe = q[:, :, :probe_t, :].detach()
                k_probe = k[:, :, :probe_t, :].detach()
                indices_probe = indices_per_head[:, :probe_t, :]

                indices_expanded = indices_probe[None, :, :, :].expand(B, H, probe_t, degree)
                indices_flat = indices_expanded.reshape(B * H, probe_t, degree)
                indices_clamped = indices_flat.clamp(min=0)

                k_flat = k_probe.reshape(B * H, probe_t, Dh)
                indices_for_gather = indices_clamped[:, :, :, None].expand(
                    B * H, probe_t, degree, Dh
                )
                k_sparse = torch.gather(
                    k_flat[:, :, None, :].expand(B * H, probe_t, degree, Dh),
                    dim=1,
                    index=indices_for_gather,
                ).reshape(B, H, probe_t, degree, Dh)

                scale = Dh**-0.5
                scores = torch.einsum("bhtd,bhtpd->bhtp", q_probe, k_sparse) * scale
                validity_mask = (indices_flat >= 0).reshape(B, H, probe_t, degree)
                scores = scores.masked_fill(~validity_mask, float("-inf"))

                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, 0.0)
                self._compute_weight_only_metrics(
                    attn_weights,
                    window_slot_mask=self._build_window_slot_mask(indices_expanded, probe_t),
                )
                self.last_attn_weights = None
        except Exception as e:
            if not self._warned_probe_failure:
                warnings.warn(
                    "Sparse probe metrics failed; keeping attention entropy/mass as NA. "
                    f"Error: {type(e).__name__}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_probe_failure = True

    def _forward_fused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        indices_per_head: torch.Tensor,
    ) -> torch.Tensor:
        """Run fused sparse attention via torch flex_attention."""
        _, H, T, _ = q.shape
        block_mask = self._get_fused_block_mask(
            H=H, T=T, device=q.device, indices_per_head=indices_per_head
        )
        fused_attention_fn = self._get_fused_attention_fn()
        try:
            output = fused_attention_fn(q, k, v, block_mask=block_mask)
        except Exception:
            # Retry once in eager flex mode if compiled flex lowering fails.
            if flex_attention is None or fused_attention_fn is flex_attention:
                raise
            self._fused_attention_fn = flex_attention
            if not self._warned_compile_fallback:
                warnings.warn(
                    "Compiled flex_attention failed at runtime; retrying with eager flex_attention.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_compile_fallback = True
            output = self._fused_attention_fn(q, k, v, block_mask=block_mask)
        self._set_unavailable_weight_metrics()
        self._maybe_probe_weight_metrics(q, k, indices_per_head=indices_per_head)
        return output

    def _forward_gather(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None,
        indices_per_head: torch.Tensor,
    ) -> torch.Tensor:
        """Run gather/scatter sparse attention (reference implementation)."""
        B, H, T, Dh = q.shape
        degree = indices_per_head.shape[-1]

        # Expand indices for batch: [H, T, degree] -> [B, H, T, degree]
        indices_expanded = indices_per_head[None, :, :, :].expand(B, H, T, degree)

        # Prepare for gather: reshape to [B*H, T, degree]
        k_flat = k.reshape(B * H, T, Dh)
        v_flat = v.reshape(B * H, T, Dh)
        indices_flat = indices_expanded.reshape(B * H, T, degree)
        indices_clamped = indices_flat.clamp(min=0)

        # Gather K and V: [B*H, T, degree, Dh]
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

        if attn_mask is not None:
            scores = self._apply_attention_mask(
                scores, attn_mask, indices_clamped, indices_expanded, B, H, T, degree
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)

        self.last_attn_weights = attn_weights.detach()
        self._compute_attention_metrics(
            attn_weights.detach(),
            window_slot_mask=self._build_window_slot_mask(indices_expanded, T),
        )
        return torch.einsum("bhtp,bhtpd->bhtd", attn_weights, v_sparse)

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
        _, H, T, _ = q.shape
        device = q.device

        # Build indices for each head (per-head variation), cached.
        indices_per_head = self._get_indices_per_head(T=T, H=H, device=device)
        self._compute_index_diagnostics(indices_per_head)

        if self._supports_fused_sparse(q, attn_mask):
            try:
                return self._forward_fused(q, k, v, indices_per_head=indices_per_head)
            except Exception as e:
                if self.sparse_impl == "flex":
                    raise RuntimeError(
                        f"Fused sparse attention failed with sparse_impl='flex': {type(e).__name__}: {e}"
                    ) from e
                if not self._warned_fallback:
                    warnings.warn(
                        "Fused sparse attention failed; falling back to gather path. "
                        f"Error: {type(e).__name__}: {e}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_fallback = True
        elif self.sparse_impl == "flex":
            raise RuntimeError(
                "sparse_impl='flex' requested, but fused path is unavailable for this call "
                "(requires CUDA, torch.nn.attention.flex_attention, and attn_mask=None)."
            )

        return self._forward_gather(q, k, v, attn_mask=attn_mask, indices_per_head=indices_per_head)

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

    def _build_window_slot_mask(self, indices_expanded: torch.Tensor, T: int) -> torch.Tensor:
        """Return [B, H, T, degree] mask for neighbors that are inside local window."""
        # indices_expanded: [B, H, T, degree]
        device = indices_expanded.device
        query_positions = torch.arange(T, device=device, dtype=indices_expanded.dtype)[
            None, None, :, None
        ]
        window_start = query_positions - (self.window_size - 1)
        in_window = (indices_expanded >= window_start) & (indices_expanded <= query_positions)
        valid_indices = indices_expanded >= 0
        return in_window & valid_indices

    def _compute_attention_metrics(
        self, attn_weights: torch.Tensor, *, window_slot_mask: torch.Tensor | None = None
    ) -> None:
        """Compute and store attention metrics from weights.

        Args:
            attn_weights: [B, H, T, degree] attention weights (detached)
            window_slot_mask: Optional [B, H, T, degree] boolean mask for local-window slots
        """
        import math

        if attn_weights.numel() == 0:
            self.last_attn_entropy = 0.0
            self.last_attn_entropy_normalized = 0.0
            self.last_valid_neighbor_fraction = 0.0
            self.last_attention_mass_window_pct = 0.0
            self.last_attention_mass_expander_pct = 0.0
            self.last_valid_neighbor_fraction_vs_causal_cap = 0.0
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
        if window_slot_mask is None:
            window_mass = attn_weights[..., : self.window_size].sum(dim=-1).mean().item()
            expander_mass = attn_weights[..., self.window_size :].sum(dim=-1).mean().item()
        else:
            window_mask_f = window_slot_mask.to(dtype=attn_weights.dtype)
            expander_mask_f = (~window_slot_mask).to(dtype=attn_weights.dtype)
            window_mass = (attn_weights * window_mask_f).sum(dim=-1).mean().item()
            expander_mass = (attn_weights * expander_mask_f).sum(dim=-1).mean().item()
        total = window_mass + expander_mass
        if total > 0:
            window_pct = 100.0 * window_mass / total
            expander_pct = 100.0 * expander_mass / total
        else:
            window_pct = expander_pct = 0.0

        self.last_attn_entropy = entropy
        self.last_attn_entropy_normalized = entropy_normalized
        self.last_valid_neighbor_fraction = valid_neighbor_fraction
        if self.last_valid_neighbor_fraction_causal_cap > 0:
            self.last_valid_neighbor_fraction_vs_causal_cap = (
                valid_neighbor_fraction / self.last_valid_neighbor_fraction_causal_cap
            )
        else:
            self.last_valid_neighbor_fraction_vs_causal_cap = 0.0
        self.last_attention_mass_window_pct = window_pct
        self.last_attention_mass_expander_pct = expander_pct
