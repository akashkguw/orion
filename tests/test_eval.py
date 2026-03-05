from __future__ import annotations

import math

import torch

from orion.config import OrionConfig
from orion.eval import evaluate, evaluate_long_context
from orion.model import TinyDecoderOnly


def _save_tiny_checkpoint(path: str, *, vocab_size: int, d_model: int) -> None:
    model = TinyDecoderOnly(
        vocab_size=vocab_size, d_model=d_model, n_layers=1, n_heads=1, mlp_mult=2
    )
    torch.save({"model": model.state_dict()}, path)


def test_evaluate_uses_checkpoint_vocab_when_config_vocab_mismatches(tmp_path):
    ckpt_path = tmp_path / "checkpoint.pt"
    _save_tiny_checkpoint(str(ckpt_path), vocab_size=17, d_model=16)

    cfg = OrionConfig(
        {
            "run": {"out_dir": str(tmp_path / "run")},
            "data": {"seq_len": 8, "batch_size": 2, "vocab_size": 256},
            "model": {"name": "tiny", "d_model": 16, "n_layers": 1, "n_heads": 1, "mlp_mult": 2},
        }
    )

    metrics = evaluate(cfg, checkpoint=str(ckpt_path), device=torch.device("cpu"))
    assert set(metrics.keys()) == {"loss", "ppl"}
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["ppl"])


def test_evaluate_long_context_uses_checkpoint_vocab_when_config_vocab_mismatches(tmp_path):
    ckpt_path = tmp_path / "checkpoint.pt"
    _save_tiny_checkpoint(str(ckpt_path), vocab_size=19, d_model=12)

    cfg = OrionConfig(
        {
            "run": {"out_dir": str(tmp_path / "run")},
            "data": {"batch_size": 1, "vocab_size": 256},
            "model": {"name": "tiny", "d_model": 12, "n_layers": 1, "n_heads": 1, "mlp_mult": 2},
        }
    )

    metrics = evaluate_long_context(cfg, checkpoint=str(ckpt_path), device=torch.device("cpu"))
    assert set(metrics.keys()) == {
        "eval_ppl_512",
        "eval_ppl_1024",
        "eval_ppl_2048",
        "eval_ppl_4096",
    }
    for key in metrics:
        assert math.isfinite(metrics[key])
