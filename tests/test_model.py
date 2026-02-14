import torch

from orion.model import TinyDecoderOnly, loss_fn


def test_tiny_decoder_only_forward():
    """Test model forward pass."""
    vocab_size = 256
    d_model = 64
    n_layers = 2
    n_heads = 4
    batch_size = 2
    seq_len = 32

    model = TinyDecoderOnly(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_mult=4,
    )

    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(idx)

    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_loss_fn():
    """Test loss computation."""
    batch_size = 2
    seq_len = 32
    vocab_size = 256

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = loss_fn(logits, targets)

    assert loss.item() > 0
    assert loss.requires_grad


def test_model_device_handling():
    """Test model on different devices."""
    model = TinyDecoderOnly(vocab_size=256, d_model=64, n_layers=2, n_heads=4, mlp_mult=4)
    idx = torch.randint(0, 256, (2, 32))

    # CPU
    logits_cpu = model(idx)
    assert logits_cpu.device.type == "cpu"

    # GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        idx_gpu = idx.cuda()
        logits_gpu = model(idx_gpu)
        assert logits_gpu.device.type == "cuda"
