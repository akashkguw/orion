import tempfile
from pathlib import Path

import torch

from orion.model import TinyDecoderOnly, loss_fn


def test_checkpoint_save_and_load():
    """Test saving and loading model checkpoint."""
    vocab_size = 256
    d_model = 64
    n_layers = 2
    n_heads = 4

    # Create and train model
    model = TinyDecoderOnly(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_mult=4,
    )

    # Get initial state
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Train for a few steps
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        x = torch.randint(0, vocab_size, (2, 32))
        y = torch.roll(x, shifts=-1, dims=1)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Get trained state
    trained_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Verify model changed
    for key in initial_state:
        assert not torch.allclose(initial_state[key], trained_state[key])

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "checkpoint.pt"
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 3,
        }
        torch.save(ckpt, ckpt_path)

        # Create new model and load checkpoint
        model2 = TinyDecoderOnly(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_mult=4,
        )
        loaded_ckpt = torch.load(ckpt_path)
        model2.load_state_dict(loaded_ckpt["model"])

        # Verify states match
        for key in trained_state:
            assert torch.allclose(trained_state[key], model2.state_dict()[key])

        # Verify both models produce same output
        x = torch.randint(0, vocab_size, (2, 32))
        with torch.no_grad():
            logits1 = model(x)
            logits2 = model2(x)
        assert torch.allclose(logits1, logits2)


def test_checkpoint_metadata():
    """Test checkpoint contains required metadata."""
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

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "checkpoint.pt"
        ckpt = {
            "model": model.state_dict(),
            "step": 100,
            "seed": 42,
            "config": {"model": {"d_model": d_model}},
        }
        torch.save(ckpt, ckpt_path)

        loaded = torch.load(ckpt_path)
        assert loaded["step"] == 100
        assert loaded["seed"] == 42
        assert loaded["config"]["model"]["d_model"] == d_model
        assert "model" in loaded


def test_checkpoint_device_transfer():
    """Test checkpoint can be loaded on different device."""
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

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "checkpoint.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)

        # Load on CPU
        loaded = torch.load(ckpt_path, map_location="cpu")
        model_cpu = TinyDecoderOnly(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_mult=4,
        )
        model_cpu.load_state_dict(loaded["model"])
        assert next(model_cpu.parameters()).device.type == "cpu"

        # Verify inference works
        x = torch.randint(0, vocab_size, (2, 32))
        logits = model_cpu(x)
        assert logits.shape == (2, 32, vocab_size)
