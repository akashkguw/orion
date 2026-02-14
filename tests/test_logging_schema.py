import json
import tempfile
from pathlib import Path

import torch

from orion.logging_utils import JsonlLogger
from orion.model import TinyDecoderOnly, loss_fn


def test_metrics_jsonl_format():
    """Test metrics are logged in correct JSONL format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = Path(tmpdir) / "metrics.jsonl"
        logger = JsonlLogger(metrics_path)

        # Log a few rows
        logger.log({"step": 1, "loss": 5.5, "ppl": 245.0})
        logger.log({"step": 2, "loss": 5.4, "ppl": 221.0})

        # Read and verify format
        lines = metrics_path.read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)


def test_metrics_standard_fields():
    """Test metrics contain required standard fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = Path(tmpdir) / "metrics.jsonl"
        logger = JsonlLogger(metrics_path)

        logger.log({"step": 1, "loss": 5.5, "ppl": 245.0})

        line = metrics_path.read_text().strip()
        obj = json.loads(line)

        # Check required fields
        assert "step" in obj
        assert "loss" in obj
        assert "ppl" in obj
        assert "wall_time_s" in obj

        # Check types
        assert isinstance(obj["step"], int)
        assert isinstance(obj["loss"], float)
        assert isinstance(obj["ppl"], float)
        assert isinstance(obj["wall_time_s"], float)


def test_metrics_wall_time_monotonic():
    """Test wall_time_s increases monotonically."""
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = Path(tmpdir) / "metrics.jsonl"
        logger = JsonlLogger(metrics_path)

        logger.log({"step": 1, "loss": 5.5, "ppl": 245.0})
        time.sleep(0.01)
        logger.log({"step": 2, "loss": 5.4, "ppl": 221.0})

        lines = metrics_path.read_text().strip().split("\n")
        obj1 = json.loads(lines[0])
        obj2 = json.loads(lines[1])

        assert obj2["wall_time_s"] > obj1["wall_time_s"]


def test_metrics_optional_vram_field():
    """Test optional VRAM field is included when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = Path(tmpdir) / "metrics.jsonl"
        logger = JsonlLogger(metrics_path)

        logger.log({"step": 1, "loss": 5.5, "ppl": 245.0, "vram_max_mb": 2048})

        line = metrics_path.read_text().strip()
        obj = json.loads(line)

        assert "vram_max_mb" in obj
        assert isinstance(obj["vram_max_mb"], int)
        assert obj["vram_max_mb"] == 2048


def test_checkpoint_structure():
    """Test checkpoint contains required fields."""
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
    optimizer = torch.optim.Adam(model.parameters())

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "checkpoint.pt"

        ckpt = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "step": 100,
            "seed": 42,
            "config": {"model": {"d_model": d_model}},
        }
        torch.save(ckpt, ckpt_path)

        # Load and verify
        loaded = torch.load(ckpt_path)

        assert "model" in loaded
        assert "opt" in loaded
        assert "step" in loaded
        assert "seed" in loaded
        assert "config" in loaded

        assert isinstance(loaded["step"], int)
        assert isinstance(loaded["seed"], int)
        assert isinstance(loaded["config"], dict)


def test_checkpoint_model_state_dict():
    """Test checkpoint model state dict is valid."""
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

        ckpt = {"model": model.state_dict()}
        torch.save(ckpt, ckpt_path)

        loaded = torch.load(ckpt_path)
        model2 = TinyDecoderOnly(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_mult=4,
        )

        # Should load without error
        model2.load_state_dict(loaded["model"])

        # Verify weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


def test_run_directory_structure():
    """Test run directory follows convention."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "runs" / "exp_test"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create expected files
        metrics_file = run_dir / "metrics.jsonl"
        checkpoint_file = run_dir / "checkpoint.pt"

        # Log metrics
        logger = JsonlLogger(metrics_file)
        logger.log({"step": 1, "loss": 5.5, "ppl": 245.0})

        # Save checkpoint
        model = TinyDecoderOnly(256, 64, 2, 4, 4)
        torch.save({"model": model.state_dict()}, checkpoint_file)

        # Verify structure
        assert metrics_file.exists()
        assert checkpoint_file.exists()
        assert (run_dir / "metrics.jsonl").is_file()
        assert (run_dir / "checkpoint.pt").is_file()


def test_metrics_step_field_1indexed():
    """Test step field is 1-indexed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = Path(tmpdir) / "metrics.jsonl"
        logger = JsonlLogger(metrics_path)

        # Log steps starting from 1
        for step in range(1, 6):
            logger.log({"step": step, "loss": 5.5, "ppl": 245.0})

        lines = metrics_path.read_text().strip().split("\n")
        for i, line in enumerate(lines):
            obj = json.loads(line)
            assert obj["step"] == i + 1
