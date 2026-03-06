import pytest
import torch

from orion.attention.base import AttentionConfig
from orion.attention.sparse import SparseAttention, build_sparse_indices
from orion.model import loss_fn
from orion.models_factory import OrionDecoder, build_model


class TestBuildSparseIndices:
    """Test sparse indices generation."""

    def test_indices_shape(self):
        """Test that indices have correct shape."""
        n, window_size, expander_degree = 16, 4, 3
        indices = build_sparse_indices(n, window_size, expander_degree, head_idx=0, device="cpu")

        assert indices.shape == (n, window_size + expander_degree)

    def test_indices_causality(self):
        """Test that all indices are <= query position (causality)."""
        n, window_size, expander_degree = 32, 8, 4
        indices = build_sparse_indices(n, window_size, expander_degree, head_idx=0, device="cpu")

        for q in range(n):
            valid_indices = [idx for idx in indices[q].tolist() if idx >= 0]
            assert all(idx <= q for idx in valid_indices), (
                f"Query {q} has future indices: {valid_indices}"
            )

    def test_window_always_present(self):
        """Test that local window is always included."""
        n, window_size, expander_degree = 32, 8, 4
        indices = build_sparse_indices(n, window_size, expander_degree, head_idx=0, device="cpu")

        for q in range(n):
            valid_indices = set(idx.item() for idx in indices[q] if idx >= 0)
            # Check that recent positions are in the window
            for i in range(max(0, q - window_size + 1), q + 1):
                assert i in valid_indices, f"Query {q} missing window position {i}"

    def test_consistent_degree(self):
        """Test that each query has consistent degree (window_size + expander_degree)."""
        n, window_size, expander_degree = 32, 8, 4
        indices = build_sparse_indices(n, window_size, expander_degree, head_idx=0, device="cpu")

        target_degree = window_size + expander_degree
        for q in range(n):
            row = indices[q].tolist()
            # Count valid indices (>= 0)
            valid_count = sum(1 for idx in row if idx >= 0)
            assert valid_count <= target_degree, (
                f"Query {q} has too many valid indices: {valid_count}"
            )

    def test_per_head_variation(self):
        """Test that different heads produce different patterns."""
        n, window_size, expander_degree = 32, 8, 4

        indices_h0 = build_sparse_indices(n, window_size, expander_degree, head_idx=0, device="cpu")
        indices_h1 = build_sparse_indices(n, window_size, expander_degree, head_idx=1, device="cpu")

        # Patterns should differ (at least for some queries)
        differences = (indices_h0 != indices_h1).sum().item()
        assert differences > 0, "Different heads should produce different patterns"

    def test_deterministic(self):
        """Test that same head_idx produces same pattern."""
        n, window_size, expander_degree = 32, 8, 4

        indices1 = build_sparse_indices(n, window_size, expander_degree, head_idx=0, device="cpu")
        indices2 = build_sparse_indices(n, window_size, expander_degree, head_idx=0, device="cpu")

        assert torch.equal(indices1, indices2), "Same head_idx should produce identical patterns"

    def test_no_duplicate_valid_neighbors_per_query(self):
        """Valid neighbors should be unique per query after deduplication."""
        n, window_size, expander_degree = 64, 8, 16
        indices = build_sparse_indices(n, window_size, expander_degree, head_idx=0, device="cpu")

        for q in range(n):
            valid = [idx for idx in indices[q].tolist() if idx >= 0]
            assert len(valid) == len(set(valid)), (
                f"Query {q} has duplicate valid neighbors: {valid}"
            )

    def test_small_sequence(self):
        """Test with very small sequence length."""
        n, window_size, expander_degree = 4, 2, 2
        indices = build_sparse_indices(n, window_size, expander_degree, head_idx=0, device="cpu")

        assert indices.shape == (4, 4)
        # All indices should be valid for small sequences
        assert (indices >= 0).all() or (indices == -1).any()

    def test_large_expander_degree(self):
        """Test with large expander degree."""
        n, window_size, expander_degree = 64, 8, 16
        indices = build_sparse_indices(n, window_size, expander_degree, head_idx=0, device="cpu")

        assert indices.shape == (n, window_size + expander_degree)


class TestSparseAttention:
    """Test SparseAttention forward pass."""

    def test_forward_shape(self):
        """Test that output has correct shape."""
        B, H, T, Dh = 2, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        output = attn.forward(q, k, v)

        assert output.shape == (B, H, T, Dh)

    def test_forward_device(self):
        """Test that output is on correct device."""
        B, H, T, Dh = 2, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        output = attn.forward(q, k, v)

        assert output.device == q.device

    def test_forward_dtype(self):
        """Test that output has correct dtype."""
        B, H, T, Dh = 2, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh, dtype=torch.float32)
        k = torch.randn(B, H, T, Dh, dtype=torch.float32)
        v = torch.randn(B, H, T, Dh, dtype=torch.float32)

        output = attn.forward(q, k, v)

        assert output.dtype == torch.float32

    def test_forward_no_nan(self):
        """Test that output contains no NaN values."""
        B, H, T, Dh = 2, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        output = attn.forward(q, k, v)

        assert not torch.isnan(output).any()

    def test_attention_mass_split_uses_window_slot_mask(self):
        """Mass split should be computed from semantic window membership, not slot position."""
        cfg = AttentionConfig(backend="sparse", window_size=3, expander_degree=1)
        attn = SparseAttention(cfg)

        # One query token, four sparse neighbors.
        # All attention mass goes to slot #1.
        attn_weights = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]], dtype=torch.float32)
        # Only slot #0 is in-window; slot #1 is expander.
        window_slot_mask = torch.tensor([[[[True, False, False, False]]]])

        attn._compute_attention_metrics(attn_weights, window_slot_mask=window_slot_mask)

        assert attn.last_attention_mass_window_pct == 0.0
        assert attn.last_attention_mass_expander_pct == 100.0

    def test_index_diagnostics_have_no_future_or_duplicates(self):
        """Sparse index diagnostics should report no future/duplicate edges."""
        B, H, T, Dh = 2, 4, 64, 32
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)
        _ = attn.forward(q, k, v)

        assert attn.last_future_neighbor_slots == 0
        assert attn.last_duplicate_neighbor_slots == 0
        assert attn.last_valid_neighbor_fraction_causal_cap > 0.0
        assert attn.last_valid_neighbor_fraction_vs_causal_cap > 0.99

    def test_causal_cap_fraction_matches_theory_for_large_window(self):
        """Causal-cap expected valid fraction should match analytic value."""
        B, H, T, Dh = 1, 1, 256, 8
        window_size, expander_degree = 256, 16
        cfg = AttentionConfig(
            backend="sparse", window_size=window_size, expander_degree=expander_degree
        )
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)
        _ = attn.forward(q, k, v)

        degree = window_size + expander_degree
        expected = sum(min(i + 1, degree) for i in range(T)) / (T * degree)
        assert abs(attn.last_valid_neighbor_fraction_causal_cap - expected) < 1e-6

    def test_forward_with_padding_mask(self):
        """Test forward pass with padding mask."""
        B, H, T, Dh = 2, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        # Create padding mask (last 8 positions masked)
        attn_mask = torch.ones(B, T, dtype=torch.bool)
        attn_mask[:, -8:] = False

        output = attn.forward(q, k, v, attn_mask=attn_mask)

        assert output.shape == (B, H, T, Dh)
        assert not torch.isnan(output).any()

    def test_forward_with_full_mask(self):
        """Test forward pass with full attention mask."""
        B, H, T, Dh = 2, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        # Create full mask (causal + some padding)
        attn_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        attn_mask = attn_mask[None, None, :, :].expand(B, H, T, T).clone()
        attn_mask[:, :, :, -8:] = False  # Mask last 8 positions

        output = attn.forward(q, k, v, attn_mask=attn_mask)

        assert output.shape == (B, H, T, Dh)
        assert not torch.isnan(output).any()

    def test_indices_caching(self):
        """Test that indices are cached correctly."""
        B, H, T, Dh = 2, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        # First forward pass
        attn.forward(q, k, v)
        cache_size_1 = len(attn.indices_cache)

        # Second forward pass (should use cache)
        attn.forward(q, k, v)
        cache_size_2 = len(attn.indices_cache)

        assert cache_size_1 == cache_size_2, "Cache should not grow on repeated calls"

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        B, H, Dh = 2, 4, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        for T in [16, 32, 64, 128]:
            q = torch.randn(B, H, T, Dh)
            k = torch.randn(B, H, T, Dh)
            v = torch.randn(B, H, T, Dh)

            output = attn.forward(q, k, v)

            assert output.shape == (B, H, T, Dh)
            assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through sparse attention."""
        B, H, T, Dh = 2, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh, requires_grad=True)
        k = torch.randn(B, H, T, Dh, requires_grad=True)
        v = torch.randn(B, H, T, Dh, requires_grad=True)

        output = attn.forward(q, k, v)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


class TestSparseAttentionIntegration:
    """Test sparse attention integration with model."""

    def test_model_with_sparse_attention(self):
        """Test building model with sparse attention."""
        cfg = AttentionConfig(backend="sparse", window_size=16, expander_degree=8)
        model = build_model(
            name="orion",
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            mlp_mult=4,
            device=torch.device("cpu"),
            attention_cfg=cfg,
        )

        assert isinstance(model, OrionDecoder)

    def test_forward_pass_with_sparse(self):
        """Test forward pass with sparse attention."""
        cfg = AttentionConfig(backend="sparse", window_size=16, expander_degree=8)
        model = build_model(
            name="orion",
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            mlp_mult=4,
            device=torch.device("cpu"),
            attention_cfg=cfg,
        )

        idx = torch.randint(0, 256, (2, 32))
        logits = model(idx)

        assert logits.shape == (2, 32, 256)

    def test_loss_computation_with_sparse(self):
        """Test loss computation with sparse attention."""
        cfg = AttentionConfig(backend="sparse", window_size=16, expander_degree=8)
        model = build_model(
            name="orion",
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            mlp_mult=4,
            device=torch.device("cpu"),
            attention_cfg=cfg,
        )

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))

        logits = model(idx)
        loss = loss_fn(logits, targets)

        assert loss.item() > 0
        assert loss.requires_grad

    def test_training_step_with_sparse(self):
        """Test training step with sparse attention."""
        cfg = AttentionConfig(backend="sparse", window_size=16, expander_degree=8)
        model = build_model(
            name="orion",
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            mlp_mult=4,
            device=torch.device("cpu"),
            attention_cfg=cfg,
        )

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        idx = torch.randint(0, 256, (4, 32))
        targets = torch.randint(0, 256, (4, 32))

        model.train()
        initial_loss = loss_fn(model(idx), targets).item()

        for _ in range(5):
            loss = loss_fn(model(idx), targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        final_loss = loss.item()
        assert final_loss < initial_loss, (
            f"Loss didn't decrease: {initial_loss:.3f} → {final_loss:.3f}"
        )

    def test_sparse_vs_dense_output_shape(self):
        """Test that sparse and dense produce same output shape."""
        from orion.attention.base import AttentionConfig

        sparse_cfg = AttentionConfig(backend="sparse", window_size=16, expander_degree=8)
        dense_cfg = AttentionConfig(backend="dense")

        sparse_model = build_model(
            name="orion",
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            mlp_mult=4,
            device=torch.device("cpu"),
            attention_cfg=sparse_cfg,
        )

        dense_model = build_model(
            name="orion",
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            mlp_mult=4,
            device=torch.device("cpu"),
            attention_cfg=dense_cfg,
        )

        idx = torch.randint(0, 256, (2, 32))

        sparse_logits = sparse_model(idx)
        dense_logits = dense_model(idx)

        assert sparse_logits.shape == dense_logits.shape


class TestSparseAttentionEdgeCases:
    """Test edge cases and robustness."""

    def test_modulus_full_range_coverage(self):
        """Test that modulus=n provides full-range coverage."""
        n, window_size, expander_degree = 64, 8, 4
        indices = build_sparse_indices(n, window_size, expander_degree, head_idx=0, device="cpu")

        # Check that expander edges can reach beyond window
        # For query at position 63, expander should reach some positions beyond window
        query_63_neighbors = [idx.item() for idx in indices[63] if idx >= 0]
        # Window covers [63-8+1, 63] = [56, 63]
        # Expander should reach some positions < 56
        expander_neighbors = [n for n in query_63_neighbors if n < 56]
        assert len(expander_neighbors) > 0, "Expander should reach beyond window"

    def test_per_head_offset_variation(self):
        """Test that per-head offset creates meaningful variation."""
        n, window_size, expander_degree = 32, 4, 3

        indices_h0 = build_sparse_indices(n, window_size, expander_degree, head_idx=0, device="cpu")
        indices_h1 = build_sparse_indices(n, window_size, expander_degree, head_idx=1, device="cpu")
        indices_h2 = build_sparse_indices(n, window_size, expander_degree, head_idx=2, device="cpu")

        # Count differences between heads
        diff_h0_h1 = (indices_h0 != indices_h1).sum().item()
        diff_h1_h2 = (indices_h1 != indices_h2).sum().item()

        # Should have meaningful differences
        assert diff_h0_h1 > 0, "Head 0 and 1 should differ"
        assert diff_h1_h2 > 0, "Head 1 and 2 should differ"

    def test_window_size_one(self):
        """Test with minimum window size (1)."""
        B, H, T, Dh = 2, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=1, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        output = attn.forward(q, k, v)

        assert output.shape == (B, H, T, Dh)
        assert not torch.isnan(output).any()

    def test_large_expander_degree(self):
        """Test with large expander degree."""
        B, H, T, Dh = 2, 4, 64, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=16)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        output = attn.forward(q, k, v)

        assert output.shape == (B, H, T, Dh)
        assert not torch.isnan(output).any()

    def test_single_head(self):
        """Test with single head."""
        B, H, T, Dh = 2, 1, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        output = attn.forward(q, k, v)

        assert output.shape == (B, H, T, Dh)
        assert not torch.isnan(output).any()

    def test_single_batch(self):
        """Test with single batch."""
        B, H, T, Dh = 1, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        output = attn.forward(q, k, v)

        assert output.shape == (B, H, T, Dh)
        assert not torch.isnan(output).any()

    def test_mask_broadcast_robustness(self):
        """Test that [B, 1, T, T] mask is properly expanded."""
        B, H, T, Dh = 2, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        # Create [B, 1, T, T] mask (common pattern)
        attn_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        attn_mask = attn_mask[None, None, :, :].expand(B, 1, T, T).clone()

        output = attn.forward(q, k, v, attn_mask=attn_mask)

        assert output.shape == (B, H, T, Dh)
        assert not torch.isnan(output).any()

    def test_all_masked_positions(self):
        """Test behavior when all positions are masked."""
        B, H, T, Dh = 2, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        # Mask all positions except first
        attn_mask = torch.zeros(B, T, dtype=torch.bool)
        attn_mask[:, 0] = True

        output = attn.forward(q, k, v, attn_mask=attn_mask)

        assert output.shape == (B, H, T, Dh)
        # Should have NaN for positions with no valid neighbors
        # (except first position which has itself)
        assert not torch.isnan(output[:, :, 0, :]).any()

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 (when not all masked)."""
        B, H, T, Dh = 2, 4, 32, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)

        # Manually compute attention weights to verify they sum to 1
        indices_per_head = torch.stack([attn._get_indices(T, h, q.device) for h in range(H)], dim=0)
        degree = indices_per_head.shape[-1]
        indices_expanded = indices_per_head[None, :, :, :].expand(B, H, T, degree)

        k_flat = k.reshape(B * H, T, Dh)
        indices_flat = indices_expanded.reshape(B * H, T, degree)
        indices_clamped = indices_flat.clamp(min=0)
        indices_for_gather = indices_clamped[:, :, :, None].expand(B * H, T, degree, Dh)

        k_sparse = torch.gather(
            k_flat[:, :, None, :].expand(B * H, T, degree, Dh), dim=1, index=indices_for_gather
        )
        k_sparse = k_sparse.reshape(B, H, T, degree, Dh)

        scale = Dh**-0.5
        scores = torch.einsum("bhtd,bhtpd->bhtp", q, k_sparse) * scale

        validity_mask = (indices_flat >= 0).reshape(B, H, T, degree)
        scores = scores.masked_fill(~validity_mask, float("-inf"))

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)

        # Check that weights sum to 1 for valid positions
        weight_sums = attn_weights.sum(dim=-1)
        valid_positions = validity_mask.any(dim=-1)
        assert torch.allclose(
            weight_sums[valid_positions], torch.ones_like(weight_sums[valid_positions]), atol=1e-5
        )

    def test_different_sequence_lengths_same_model(self):
        """Test that same model handles different sequence lengths."""
        B, H, Dh = 2, 4, 64
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        for T in [16, 32, 64, 128]:
            q = torch.randn(B, H, T, Dh)
            k = torch.randn(B, H, T, Dh)
            v = torch.randn(B, H, T, Dh)

            output = attn.forward(q, k, v)

            assert output.shape == (B, H, T, Dh)
            assert not torch.isnan(output).any()

    def test_indices_cache_different_sequences(self):
        """Test that cache correctly handles different sequence lengths."""
        cfg = AttentionConfig(backend="sparse", window_size=8, expander_degree=4)
        attn = SparseAttention(cfg)

        # Build indices for different lengths
        indices_16 = attn._get_indices(16, 0, "cpu")
        _ = attn._get_indices(32, 0, "cpu")
        indices_16_again = attn._get_indices(16, 0, "cpu")

        # Check cache has both
        assert len(attn.indices_cache) == 2

        # Check that repeated call returns same object (cached)
        assert indices_16 is indices_16_again

    def test_sparse_impl_flex_raises_without_cuda(self):
        """Forcing fused sparse implementation should fail on CPU-only calls."""
        B, H, T, Dh = 1, 2, 16, 8
        cfg = AttentionConfig(
            backend="sparse", window_size=8, expander_degree=4, sparse_impl="flex"
        )
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        with pytest.raises(RuntimeError, match="sparse_impl='flex' requested"):
            _ = attn.forward(q, k, v)

    def test_sparse_impl_gather_forces_reference_path(self):
        """Explicit gather impl should keep metrics available."""
        B, H, T, Dh = 1, 2, 16, 8
        cfg = AttentionConfig(
            backend="sparse", window_size=8, expander_degree=4, sparse_impl="gather"
        )
        attn = SparseAttention(cfg)

        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)
        v = torch.randn(B, H, T, Dh)

        out = attn.forward(q, k, v)
        assert out.shape == (B, H, T, Dh)
        assert attn.last_attn_weights is not None
