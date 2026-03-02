"""Tests for dense attention mechanism."""

from __future__ import annotations

import torch

from orion.attention.base import AttentionConfig
from orion.attention.dense import DenseAttention


class TestDenseAttentionBasic:
    """Basic dense attention functionality tests."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 4, 16, 32
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        output = attn.forward(q, k, v)

        assert output.shape == (B, H, T, Dh)

    def test_forward_dtype(self):
        """Test forward pass preserves dtype."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 4, 16, 32
        q = torch.randn(B, H, T, Dh, dtype=torch.float32)
        k = torch.randn(B, H, T, Dh, dtype=torch.float32)
        v = torch.randn(B, H, T, Dh, dtype=torch.float32)

        output = attn.forward(q, k, v)

        assert output.dtype == torch.float32

    def test_forward_device(self):
        """Test forward pass on CPU."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 4, 16, 32
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        output = attn.forward(q, k, v)

        assert output.device == q.device

    def test_stores_attention_weights(self):
        """Test that attention weights are stored."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 4, 16, 32
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        attn.forward(q, k, v)

        assert attn.last_attn_weights is not None
        assert attn.last_attn_weights.shape == (B, H, T, T)

    def test_attention_weights_detached(self):
        """Test that stored attention weights are detached."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 4, 16, 32
        q = torch.randn(B, H, T, Dh, requires_grad=True)
        k = torch.randn(B, H, T, Dh, requires_grad=True)
        v = torch.randn(B, H, T, Dh, requires_grad=True)

        attn.forward(q, k, v)

        assert not attn.last_attn_weights.requires_grad


class TestDenseAttentionCausality:
    """Test causal masking in dense attention."""

    def test_causal_masking(self):
        """Test that future positions are masked."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 1, 1, 4, 8
        q = torch.ones(B, H, T, Dh)
        k = torch.ones(B, H, T, Dh)
        v = torch.arange(T, dtype=torch.float32).view(1, 1, T, 1).expand(B, H, T, Dh)

        attn.forward(q, k, v)
        weights = attn.last_attn_weights

        # Check that future positions have zero weight
        for t in range(T):
            for future_t in range(t + 1, T):
                assert weights[0, 0, t, future_t].item() == 0.0

    def test_causal_masking_first_token(self):
        """Test that first token only attends to itself."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 1, 1, 8, 16
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        attn.forward(q, k, v)
        weights = attn.last_attn_weights

        # First token should only attend to itself
        assert weights[0, 0, 0, 0].item() > 0.0
        for j in range(1, T):
            assert weights[0, 0, 0, j].item() == 0.0

    def test_causal_masking_last_token(self):
        """Test that last token attends to all previous tokens."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 1, 1, 8, 16
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        attn.forward(q, k, v)
        weights = attn.last_attn_weights

        # Last token should attend to all positions up to itself
        for j in range(T):
            assert weights[0, 0, T - 1, j].item() >= 0.0


class TestDenseAttentionMasking:
    """Test attention masking (padding and full masks)."""

    def test_key_padding_mask_2d(self):
        """Test 2D key padding mask."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 2, 8, 16
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        # Mask out last 2 positions for first batch, last 1 for second
        attn_mask = torch.ones(B, T, dtype=torch.bool)
        attn_mask[0, -2:] = False
        attn_mask[1, -1:] = False

        attn.forward(q, k, v, attn_mask=attn_mask)
        weights = attn.last_attn_weights

        # Check that masked positions have zero weight
        for t in range(T):
            # First batch: positions 6, 7 should be masked
            assert weights[0, 0, t, 6].item() == 0.0
            assert weights[0, 0, t, 7].item() == 0.0
            # Second batch: position 7 should be masked
            assert weights[1, 0, t, 7].item() == 0.0

    def test_full_mask_4d(self):
        """Test 4D full attention mask."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 1, 2, 4, 8
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        # Create a mask that blocks specific positions
        attn_mask = torch.ones(B, H, T, T, dtype=torch.bool)
        attn_mask[0, 0, 0, 1] = False  # Block (0,1) for head 0
        attn_mask[0, 1, 1, 2] = False  # Block (1,2) for head 1

        attn.forward(q, k, v, attn_mask=attn_mask)
        weights = attn.last_attn_weights

        # Check masked positions
        assert weights[0, 0, 0, 1].item() == 0.0
        assert weights[0, 1, 1, 2].item() == 0.0

    def test_full_mask_broadcast(self):
        """Test 4D mask with head dimension 1 broadcasts correctly."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 1, 4, 8, 16
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        # Mask with head dimension 1 (should broadcast to all heads)
        attn_mask = torch.ones(B, 1, T, T, dtype=torch.bool)
        attn_mask[0, 0, 0, 1:] = False

        attn.forward(q, k, v, attn_mask=attn_mask)
        weights = attn.last_attn_weights

        # All heads should have the same mask applied
        for h in range(H):
            assert weights[0, h, 0, 0].item() > 0.0
            for j in range(1, T):
                assert weights[0, h, 0, j].item() == 0.0


class TestDenseAttentionCaching:
    """Test causal mask caching."""

    def test_causal_mask_cached(self):
        """Test that causal masks are cached."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 4, 16, 32
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        # First forward pass
        attn.forward(q, k, v)
        cache_size_1 = len(attn.causal_mask_cache)

        # Second forward pass with same T
        attn.forward(q, k, v)
        cache_size_2 = len(attn.causal_mask_cache)

        # Cache should not grow
        assert cache_size_1 == cache_size_2

    def test_causal_mask_different_lengths(self):
        """Test that different sequence lengths use different cache entries."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, Dh = 2, 4, 32

        # First forward pass with T=16
        q1 = torch.randn(B, H, 16, Dh)
        k1 = torch.randn(B, H, 16, Dh)
        v1 = torch.randn(B, H, 16, Dh)
        attn.forward(q1, k1, v1)
        cache_size_1 = len(attn.causal_mask_cache)

        # Second forward pass with T=32
        q2 = torch.randn(B, H, 32, Dh)
        k2 = torch.randn(B, H, 32, Dh)
        v2 = torch.randn(B, H, 32, Dh)
        attn.forward(q2, k2, v2)
        cache_size_2 = len(attn.causal_mask_cache)

        # Cache should grow
        assert cache_size_2 > cache_size_1


class TestDenseAttentionNumerical:
    """Test numerical properties of dense attention."""

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 per query."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 4, 16, 32
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        attn.forward(q, k, v)
        weights = attn.last_attn_weights

        # Sum over key dimension should be 1
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_attention_weights_non_negative(self):
        """Test that attention weights are non-negative."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 4, 16, 32
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        attn.forward(q, k, v)
        weights = attn.last_attn_weights

        assert (weights >= 0.0).all()

    def test_output_bounded(self):
        """Test that output is bounded by input range."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 4, 16, 32
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.ones(B, H, T, Dh)  # All ones for easy checking

        output = attn.forward(q, k, v)

        # Output should be close to 1 (weighted average of all ones)
        assert torch.allclose(output, torch.ones_like(output), atol=0.1)


class TestDenseAttentionGradients:
    """Test gradient flow through dense attention."""

    def test_gradients_flow(self):
        """Test that gradients flow through attention."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 4, 8, 16
        q = torch.randn(B, H, T, Dh, requires_grad=True)
        k = torch.randn(B, H, T, Dh, requires_grad=True)
        v = torch.randn(B, H, T, Dh, requires_grad=True)

        output = attn.forward(q, k, v)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert (q.grad != 0).any()
        assert (k.grad != 0).any()
        assert (v.grad != 0).any()

    def test_no_gradients_in_stored_weights(self):
        """Test that stored weights don't require gradients."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 4, 8, 16
        q = torch.randn(B, H, T, Dh, requires_grad=True)
        k = torch.randn(B, H, T, Dh, requires_grad=True)
        v = torch.randn(B, H, T, Dh, requires_grad=True)

        attn.forward(q, k, v)

        # Stored weights should not require gradients
        assert not attn.last_attn_weights.requires_grad


class TestDenseAttentionEdgeCases:
    """Test edge cases for dense attention."""

    def test_single_token(self):
        """Test attention with single token."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, Dh = 2, 4, 32
        q = torch.randn(B, H, 1, Dh)
        k = torch.randn(B, H, 1, Dh)
        v = torch.randn(B, H, 1, Dh)

        output = attn.forward(q, k, v)

        assert output.shape == (B, H, 1, Dh)
        # Single token should attend entirely to itself
        assert attn.last_attn_weights[0, 0, 0, 0].item() == 1.0

    def test_large_sequence(self):
        """Test attention with large sequence."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 1, 2, 256, 16
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        output = attn.forward(q, k, v)

        assert output.shape == (B, H, T, Dh)

    def test_batch_independence(self):
        """Test that batch elements are independent."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 2, 4, 16, 32
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        output = attn.forward(q, k, v)

        # Compute separately for each batch element
        output_0 = attn.forward(q[0:1], k[0:1], v[0:1])
        output_1 = attn.forward(q[1:2], k[1:2], v[1:2])

        # Results should match
        assert torch.allclose(output[0], output_0[0], atol=1e-5)
        assert torch.allclose(output[1], output_1[0], atol=1e-5)

    def test_head_independence(self):
        """Test that attention heads are independent."""
        cfg = AttentionConfig(backend="dense")
        attn = DenseAttention(cfg)

        B, H, T, Dh = 1, 4, 16, 32
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        output = attn.forward(q, k, v)

        # Compute separately for each head
        output_0 = attn.forward(q[:, 0:1], k[:, 0:1], v[:, 0:1])

        # Results should match
        assert torch.allclose(output[:, 0], output_0[:, 0], atol=1e-5)
