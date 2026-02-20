import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from orion.config import load_config
from orion.model import TinyDecoderOnly, loss_fn
from orion.train import train as run_train


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
        loaded_ckpt = torch.load(ckpt_path, weights_only=False)
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
            "epoch": 100,
            "seed": 42,
            "config": {"model": {"d_model": d_model}},
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
            },
        }
        torch.save(ckpt, ckpt_path)

        loaded = torch.load(ckpt_path, weights_only=False)
        assert loaded["step"] == 100
        assert loaded["epoch"] == 100
        assert loaded["seed"] == 42
        assert loaded["config"]["model"]["d_model"] == d_model
        assert "model" in loaded
        assert "rng_state" in loaded


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
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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


def test_training_resume_deterministic():
    """Test resume produces same final weights and metrics."""
    os.environ.pop("SMOKE_TEST", None)
    device = torch.device("cpu")

    def write_config(path: Path, out_dir: Path, steps: int) -> None:
        cfg = {
            "run": {
                "out_dir": str(out_dir),
                "seed": 123,
                "steps": steps,
                "log_every": 1,
                "save_every": 3,
            },
            "data": {
                "dataset": "toy",
                "vocab_size": 32,
                "seq_len": 16,
                "batch_size": 2,
            },
            "model": {
                "name": "tiny",
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 2,
                "mlp_mult": 2,
            },
            "optim": {"lr": 1e-3},
        }
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        full_dir = tmp_path / "full"
        resume_dir = tmp_path / "resume"

        full_cfg = tmp_path / "full.yaml"
        partial_cfg = tmp_path / "partial.yaml"
        resume_cfg = tmp_path / "resume.yaml"

        write_config(full_cfg, full_dir, steps=6)
        write_config(partial_cfg, resume_dir, steps=3)
        write_config(resume_cfg, resume_dir, steps=6)

        run_train(load_config(str(full_cfg)), device=device)

        run_train(load_config(str(partial_cfg)), device=device)
        run_train(
            load_config(str(resume_cfg)),
            device=device,
            resume_path=resume_dir / "checkpoint.pt",
        )

        full_ckpt = torch.load(full_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        resume_ckpt = torch.load(
            resume_dir / "checkpoint.pt", map_location="cpu", weights_only=False
        )

        assert full_ckpt["step"] == 6
        assert resume_ckpt["step"] == 6
        assert "rng_state" in resume_ckpt

        for key, value in full_ckpt["model"].items():
            assert torch.allclose(value, resume_ckpt["model"][key])

        metrics_lines = (resume_dir / "metrics.jsonl").read_text().splitlines()
        steps = [json.loads(line)["step"] for line in metrics_lines]
        assert steps == list(range(1, 7))


def test_resume_missing_checkpoint_raises():
    """Resume should fail clearly when checkpoint path does not exist."""
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg_path = tmp_path / "cfg.yaml"
        cfg = {
            "run": {"out_dir": str(tmp_path / "runs"), "seed": 123, "steps": 2, "log_every": 1},
            "data": {"dataset": "toy", "vocab_size": 16, "seq_len": 8, "batch_size": 2},
            "model": {"name": "tiny", "d_model": 16, "n_layers": 1, "n_heads": 2, "mlp_mult": 2},
            "optim": {"lr": 1e-3},
        }
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        with pytest.raises(FileNotFoundError):
            run_train(
                load_config(str(cfg_path)),
                device=device,
                resume_path=tmp_path / "does-not-exist.pt",
            )


def test_unknown_scheduler_name_raises():
    """Unsupported scheduler names should raise a helpful error."""
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg_path = tmp_path / "cfg.yaml"
        cfg = {
            "run": {"out_dir": str(tmp_path / "runs"), "seed": 123, "steps": 1, "log_every": 1},
            "data": {"dataset": "toy", "vocab_size": 16, "seq_len": 8, "batch_size": 2},
            "model": {"name": "tiny", "d_model": 16, "n_layers": 1, "n_heads": 2, "mlp_mult": 2},
            "optim": {"lr": 1e-3, "scheduler": {"name": "not-a-real-scheduler"}},
        }
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        with pytest.raises(ValueError, match="Unknown scheduler name"):
            run_train(load_config(str(cfg_path)), device=device)


def test_resume_noop_if_checkpoint_is_ahead_of_target_steps():
    """If checkpoint step >= requested steps, resume should no-op and preserve metrics."""
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        run_dir = tmp_path / "runs"
        cfg6 = tmp_path / "cfg6.yaml"
        cfg3 = tmp_path / "cfg3.yaml"

        base = {
            "run": {"out_dir": str(run_dir), "seed": 123, "log_every": 1, "save_every": 2},
            "data": {"dataset": "toy", "vocab_size": 16, "seq_len": 8, "batch_size": 2},
            "model": {"name": "tiny", "d_model": 16, "n_layers": 1, "n_heads": 2, "mlp_mult": 2},
            "optim": {"lr": 1e-3},
        }

        cfg = dict(base)
        cfg["run"] = dict(base["run"], steps=6)
        cfg6.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        cfg = dict(base)
        cfg["run"] = dict(base["run"], steps=3)
        cfg3.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        run_train(load_config(str(cfg6)), device=device)
        before_lines = (run_dir / "metrics.jsonl").read_text().splitlines()

        run_train(
            load_config(str(cfg3)),
            device=device,
            resume_path=run_dir / "checkpoint.pt",
        )

        after_lines = (run_dir / "metrics.jsonl").read_text().splitlines()
        assert after_lines == before_lines


def test_steps_per_epoch_affects_saved_epoch_field():
    """Checkpoint epoch should be derived from steps_per_epoch when provided."""
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        run_dir = tmp_path / "runs"
        cfg_path = tmp_path / "cfg.yaml"
        cfg = {
            "run": {
                "out_dir": str(run_dir),
                "seed": 123,
                "steps": 5,
                "log_every": 1,
                "save_every": 5,
                "steps_per_epoch": 2,
            },
            "data": {"dataset": "toy", "vocab_size": 16, "seq_len": 8, "batch_size": 2},
            "model": {"name": "tiny", "d_model": 16, "n_layers": 1, "n_heads": 2, "mlp_mult": 2},
            "optim": {"lr": 1e-3},
        }
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        run_train(load_config(str(cfg_path)), device=device)
        ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert ckpt["step"] == 5
        assert ckpt["epoch"] == 3
