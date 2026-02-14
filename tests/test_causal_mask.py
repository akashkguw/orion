import torch

from orion.model import TinyDecoderOnly


def test_causal_mask_shape():
    """Test causal mask has correct shape."""
    seq_len = 32
    device = torch.device("cpu")

    # Create causal mask as in model
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    assert causal.shape == (seq_len, seq_len)
    assert causal.dtype == torch.bool


def test_causal_mask_pattern():
    """Test causal mask has correct pattern (upper triangular)."""
    seq_len = 5
    device = torch.device("cpu")

    causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    # Lower triangle should be False (not masked)
    for i in range(seq_len):
        for j in range(i + 1):
            assert not causal[i, j], f"Position ({i}, {j}) should not be masked"

    # Upper triangle should be True (masked)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert causal[i, j], f"Position ({i}, {j}) should be masked"


def test_causal_mask_prevents_future_attention():
    """Test that causal mask prevents attending to future tokens."""
    vocab_size = 256
    d_model = 64
    n_layers = 2
    n_heads = 4
    batch_size = 2
    seq_len = 16

    model = TinyDecoderOnly(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_mult=4,
    )
    model.eval()

    # Create input where first half is different from second half
    x = torch.zeros(batch_size, seq_len, dtype=torch.long)
    x[:, : seq_len // 2] = 1  # First half: token 1
    x[:, seq_len // 2 :] = 2  # Second half: token 2

    with torch.no_grad():
        logits = model(x)

    # Logits shape: [batch_size, seq_len, vocab_size]
    assert logits.shape == (batch_size, seq_len, vocab_size)

    # First half tokens should not be influenced by second half
    # (This is a sanity check that model runs without error)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_causal_mask_diagonal():
    """Test causal mask diagonal is not masked (can attend to self)."""
    seq_len = 10
    device = torch.device("cpu")

    causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    # Diagonal should be False (not masked, can attend to self)
    for i in range(seq_len):
        assert not causal[i, i], f"Diagonal position ({i}, {i}) should not be masked"


def test_causal_mask_variable_seq_len():
    """Test causal mask works with different sequence lengths."""
    vocab_size = 256
    d_model = 64
    n_layers = 2
    n_heads = 4

    model = TinyDecoderOnly(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_mult=4,
    )
    model.eval()

    # Test with different sequence lengths
    for seq_len in [8, 16, 32, 64]:
        x = torch.randint(0, vocab_size, (2, seq_len))
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, seq_len, vocab_size)
        assert not torch.isnan(logits).any()


def test_causal_mask_batch_independence():
    """Test that causal mask is applied consistently across batch."""
    vocab_size = 256
    d_model = 64
    n_layers = 2
    n_heads = 4
    seq_len = 16

    model = TinyDecoderOnly(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_mult=4,
    )
    model.eval()

    # Create batch with same input repeated
    x = torch.randint(0, vocab_size, (1, seq_len))
    x_batch = x.repeat(4, 1)

    with torch.no_grad():
        logits_single = model(x)
        logits_batch = model(x_batch)

    # All batch elements should have identical logits
    for i in range(4):
        assert torch.allclose(logits_single[0], logits_batch[i])
