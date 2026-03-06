"""Test all model and attention combinations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from orion.config import load_config
from orion.train import train


@pytest.mark.parametrize(
    "model_name,attention_backend",
    [
        ("tiny", None),
        ("orion", "dense"),
        ("orion", "sparse"),
        ("orion", "window"),
    ],
)
def test_model_attention_combinations(model_name: str, attention_backend: str | None) -> None:
    """Test all model and attention backend combinations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg_path = tmp_path / "config.yaml"
        out_dir = tmp_path / "run"

        cfg = {
            "run": {
                "out_dir": str(out_dir),
                "seed": 42,
                "steps": 5,
                "log_every": 1,
                "save_every": 5,
            },
            "data": {
                "dataset": "toy",
                "vocab_size": 32,
                "seq_len": 16,
                "batch_size": 2,
            },
            "model": {
                "name": model_name,
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 2,
                "mlp_mult": 2,
            },
            "optim": {"lr": 1e-3},
        }

        if attention_backend:
            cfg["attention"] = {
                "backend": attention_backend,
                "window_size": 8,
                "expander_degree": 2,
            }

        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        device = torch.device("cpu")
        train(load_config(str(cfg_path)), device=device)

        # Verify checkpoint was created
        assert (out_dir / "checkpoint.pt").exists()

        # Verify metrics were logged
        metrics_file = out_dir / "metrics.jsonl"
        assert metrics_file.exists()
        lines = metrics_file.read_text().splitlines()
        assert len(lines) > 0

        # First line is run metrics; verify attention degree uses configured values.
        run_obj = json.loads(lines[0])
        if attention_backend:
            assert run_obj["attention_degree"] == 10  # window_size + expander_degree
