"""Tests for sliding-window attention backend."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from orion.attention.base import AttentionConfig
from orion.attention.window import (
    WindowAttention,
    _build_external_additive_mask,
    _build_window_mask,
)


def _manual_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window_size: int,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation for comparison in tests."""
    B, H, T, Dh = q.shape
    scale = Dh**-0.5
    scores = torch.einsum("bhtd,bhsd->bhts", q, k) * scale

    window = _build_window_mask(T, window_size, device=q.device, dtype=q.dtype)
    scores = scores + window

    if attn_mask is not None:
        extra = _build_external_additive_mask(
            attn_mask=attn_mask, B=B, H=H, T=T, device=q.device, dtype=q.dtype
        )
        scores = scores + extra

    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, 0.0)
    return torch.einsum("bhts,bhsd->bhtd", weights, v)


class TestWindowAttentionBasic:
    def test_window_size_validation(self):
        with torch.no_grad():
            try:
                WindowAttention(AttentionConfig(backend="window", window_size=0))
                raise AssertionError("Expected ValueError for window_size=0")
            except ValueError:
                pass

    def test_forward_shape_and_dtype(self):
        cfg = AttentionConfig(backend="window", window_size=8)
        attn = WindowAttention(cfg)

        B, H, T, Dh = 2, 4, 16, 32
        q = torch.randn(B, H, T, Dh, dtype=torch.float32)
        k = torch.randn(B, H, T, Dh, dtype=torch.float32)
        v = torch.randn(B, H, T, Dh, dtype=torch.float32)

        out = attn.forward(q, k, v)

        assert out.shape == (B, H, T, Dh)
        assert out.dtype == q.dtype

    def test_mask_cache_reuse(self):
        cfg = AttentionConfig(backend="window", window_size=8)
        attn = WindowAttention(cfg)

        B, H, T, Dh = 2, 2, 16, 8
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        _ = attn.forward(q, k, v)
        cache_size_1 = len(attn._mask_cache)

        _ = attn.forward(q, k, v)
        cache_size_2 = len(attn._mask_cache)

        assert cache_size_1 == cache_size_2


class TestWindowMaskSemantics:
    def test_build_window_mask_structure(self):
        T, W = 6, 3
        mask = _build_window_mask(T, W, device=torch.device("cpu"), dtype=torch.float32)[0, 0]

        # Query 0: only key 0 allowed
        assert mask[0, 0].item() == 0.0
        assert torch.isneginf(mask[0, 1]).item()

        # Query 4 with W=3: keys 2,3,4 allowed; 0,1,5 blocked
        assert mask[4, 2].item() == 0.0
        assert mask[4, 3].item() == 0.0
        assert mask[4, 4].item() == 0.0
        assert torch.isneginf(mask[4, 1]).item()
        assert torch.isneginf(mask[4, 5]).item()

    def test_forward_matches_manual_without_external_mask(self):
        cfg = AttentionConfig(backend="window", window_size=4)
        attn = WindowAttention(cfg)

        B, H, T, Dh = 2, 3, 8, 16
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        out = attn.forward(q, k, v)
        ref = _manual_window_attention(q, k, v, window_size=4)

        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


class TestWindowExternalMasking:
    def test_key_padding_mask_2d_matches_manual(self):
        cfg = AttentionConfig(backend="window", window_size=8)
        attn = WindowAttention(cfg)

        B, H, T, Dh = 2, 2, 8, 16
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)
        key_mask = torch.ones(B, T, dtype=torch.bool)
        key_mask[0, -2:] = False
        key_mask[1, -1:] = False

        out = attn.forward(q, k, v, attn_mask=key_mask)
        ref = _manual_window_attention(q, k, v, window_size=8, attn_mask=key_mask)

        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_full_mask_4d_broadcast_matches_manual(self):
        cfg = AttentionConfig(backend="window", window_size=8)
        attn = WindowAttention(cfg)

        B, H, T, Dh = 1, 4, 8, 16
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)
        full_mask = torch.ones(B, 1, T, T, dtype=torch.bool)
        full_mask[0, 0, 0, 1:] = False

        out = attn.forward(q, k, v, attn_mask=full_mask)
        ref = _manual_window_attention(q, k, v, window_size=8, attn_mask=full_mask)

        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_invalid_mask_rank_raises(self):
        cfg = AttentionConfig(backend="window", window_size=8)
        attn = WindowAttention(cfg)

        B, H, T, Dh = 1, 2, 8, 16
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)
        bad_mask = torch.ones(B, 1, T, dtype=torch.bool)  # 3D is unsupported

        try:
            _ = attn.forward(q, k, v, attn_mask=bad_mask)
            raise AssertionError("Expected ValueError for unsupported attn_mask rank")
        except ValueError:
            pass

    def test_invalid_full_mask_head_dim_raises(self):
        cfg = AttentionConfig(backend="window", window_size=8)
        attn = WindowAttention(cfg)

        B, H, T, Dh = 1, 2, 8, 16
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)
        bad_mask = torch.ones(B, 3, T, T, dtype=torch.bool)

        try:
            _ = attn.forward(q, k, v, attn_mask=bad_mask)
            raise AssertionError("Expected ValueError for invalid mask head dimension")
        except ValueError:
            pass


class TestWindowMetricsProbe:
    def test_probe_populates_entropy_and_score_metrics(self):
        cfg = AttentionConfig(
            backend="window",
            window_size=8,
            window_probe_every=1,
            window_probe_tokens=16,
        )
        attn = WindowAttention(cfg)

        B, H, T, Dh = 2, 2, 16, 16
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        _ = attn.forward(q, k, v)

        assert math.isfinite(attn.last_attn_score_mean)
        assert math.isfinite(attn.last_attn_entropy)
        assert math.isfinite(attn.last_attn_entropy_normalized)
        assert 0.0 <= attn.last_attn_entropy_normalized <= 1.0

    def test_probe_disabled_marks_weight_metrics_unavailable(self):
        cfg = AttentionConfig(
            backend="window",
            window_size=8,
            window_probe_every=0,
            window_probe_tokens=16,
        )
        attn = WindowAttention(cfg)

        B, H, T, Dh = 1, 2, 8, 16
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        _ = attn.forward(q, k, v)

        assert math.isnan(attn.last_attn_score_mean)
        assert math.isnan(attn.last_attn_entropy)
        assert math.isnan(attn.last_attn_entropy_normalized)
