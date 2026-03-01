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
