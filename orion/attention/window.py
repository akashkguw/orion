from __future__ import annotations

import warnings

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
        self.window_probe_every = max(0, int(getattr(cfg, "window_probe_every", 50) or 50))
        self.window_probe_tokens = max(16, int(getattr(cfg, "window_probe_tokens", 256) or 256))

        # Cache the mask so it's only built once per (T, device, dtype) combination
        # instead of being reallocated on every forward pass.
        self._mask_cache: dict[tuple, torch.Tensor] = {}
        self._probe_allow_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._forward_calls = 0
        self._warned_probe_failure = False
        self.last_attn_weights: torch.Tensor | None = None
        self.last_attn_entropy: float = float("nan")
        self.last_attn_entropy_normalized: float = float("nan")
        self.last_attn_score_mean: float = float("nan")

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

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=combined_mask)
        self._maybe_probe_weight_metrics(q, k, attn_mask=attn_mask)
        return output

    def _set_unavailable_weight_metrics(self) -> None:
        self.last_attn_weights = None
        self.last_attn_entropy = float("nan")
        self.last_attn_entropy_normalized = float("nan")
        self.last_attn_score_mean = float("nan")

    def _get_probe_allow_mask(self, T: int, *, device: torch.device) -> torch.Tensor:
        """Return cached [T, T] bool mask for causal sliding-window allow pattern."""
        cache_key = (T, str(device))
        if cache_key not in self._probe_allow_cache:
            rows = torch.arange(T, device=device).unsqueeze(1)
            cols = torch.arange(T, device=device).unsqueeze(0)
            causal = cols <= rows
            in_window = (rows - cols) < self.W
            self._probe_allow_cache[cache_key] = causal & in_window
        return self._probe_allow_cache[cache_key]

    def _maybe_probe_weight_metrics(
        self, q: torch.Tensor, k: torch.Tensor, *, attn_mask: torch.Tensor | None
    ) -> None:
        """Periodically estimate score/entropy on a bounded probe without full T^2 weights."""
        self._forward_calls += 1
        if self.window_probe_every <= 0:
            self._set_unavailable_weight_metrics()
            return
        if self._forward_calls % self.window_probe_every != 0:
            return

        B, H, T, Dh = q.shape
        probe_t = min(T, self.window_probe_tokens)
        if probe_t <= 0:
            self._set_unavailable_weight_metrics()
            return

        try:
            with torch.no_grad():
                q_probe = q[:, :, :probe_t, :].detach()
                k_probe = k[:, :, :probe_t, :].detach()
                scale = Dh**-0.5
                scores = torch.einsum("bhtd,bhsd->bhts", q_probe, k_probe) * scale

                allow = self._get_probe_allow_mask(probe_t, device=q.device)[None, None, :, :]
                allow = allow.expand(B, H, probe_t, probe_t)

                if attn_mask is not None:
                    probe_mask = attn_mask
                    if probe_mask.dim() == 2:
                        probe_mask = probe_mask[:, :probe_t]
                    elif probe_mask.dim() == 4:
                        probe_mask = probe_mask[:, :, :probe_t, :probe_t]
                    extra = _build_external_additive_mask(
                        attn_mask=probe_mask,
                        B=B,
                        H=H,
                        T=probe_t,
                        device=q.device,
                        dtype=q.dtype,
                    )
                    allow = allow & torch.isfinite(extra.expand(B, H, probe_t, probe_t))

                valid_scores = scores.masked_select(allow)
                if valid_scores.numel() == 0:
                    self._set_unavailable_weight_metrics()
                    return
                self.last_attn_score_mean = float(valid_scores.abs().mean().item())

                masked_scores = scores.masked_fill(~allow, float("-inf"))
                attn_weights = F.softmax(masked_scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, 0.0)
                self.last_attn_weights = None

                attn_weights_safe = torch.clamp(attn_weights, min=1e-10)
                entropy_per_row = -(attn_weights * torch.log(attn_weights_safe)).sum(dim=-1)
                degree = allow.sum(dim=-1).to(dtype=entropy_per_row.dtype)
                max_entropy = torch.where(degree > 1, torch.log(degree), torch.ones_like(degree))
                entropy_norm_per_row = torch.where(
                    degree > 1, entropy_per_row / max_entropy, torch.zeros_like(entropy_per_row)
                )
                valid_rows = degree > 0

                if valid_rows.any():
                    self.last_attn_entropy = float(
                        entropy_per_row.masked_select(valid_rows).mean().item()
                    )
                    self.last_attn_entropy_normalized = float(
                        entropy_norm_per_row.masked_select(valid_rows).mean().item()
                    )
                else:
                    self._set_unavailable_weight_metrics()
        except Exception as e:
            self._set_unavailable_weight_metrics()
            if not self._warned_probe_failure:
                warnings.warn(
                    "Window probe metrics failed; keeping attention entropy/score as NA. "
                    f"Error: {type(e).__name__}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_probe_failure = True


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
