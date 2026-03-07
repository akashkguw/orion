from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from orion.attention.base import AttentionConfig
from orion.config import OrionConfig
from orion.models_factory import build_model
from orion.stability import StabilityConfig


def _tiny_attention_cfg():
    return AttentionConfig(backend="dense")


def _tiny_sparse_cfg():
    return AttentionConfig(backend="sparse", window_size=4, expander_degree=2, sparse_impl="gather")


def _build_orion(stability_cfg=None, attention_cfg=None, d_model=32, n_heads=4, n_layers=2):
    if attention_cfg is None:
        attention_cfg = _tiny_attention_cfg()
    return build_model(
        name="orion",
        vocab_size=64,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_mult=2,
        device=torch.device("cpu"),
        attention_cfg=attention_cfg,
        stability_cfg=stability_cfg,
    )


# ---- Config parsing ----


def test_stability_config_defaults():
    cfg = OrionConfig({"run": {"out_dir": "runs/test"}, "model": {"name": "orion"}})
    sc = cfg.stability_config()
    assert sc.qk_norm is False
    assert sc.ortho_init is False
    assert sc.spectral_norm is False


def test_stability_config_parsing():
    cfg = OrionConfig(
        {
            "run": {"out_dir": "runs/test"},
            "stability": {"qk_norm": True, "ortho_init": True, "spectral_norm": False},
        }
    )
    sc = cfg.stability_config()
    assert sc.qk_norm is True
    assert sc.ortho_init is True
    assert sc.spectral_norm is False


def test_stability_config_no_stability_section():
    cfg = OrionConfig({"run": {"out_dir": "runs/test"}})
    sc = cfg.stability_config()
    assert sc == StabilityConfig()


def test_stability_config_string_bool_parsing():
    cfg = OrionConfig(
        {
            "run": {"out_dir": "runs/test"},
            "stability": {"qk_norm": "true", "ortho_init": "0", "spectral_norm": "no"},
        }
    )
    sc = cfg.stability_config()
    assert sc.qk_norm is True
    assert sc.ortho_init is False
    assert sc.spectral_norm is False


def test_stability_config_invalid_section_type_raises():
    cfg = OrionConfig({"run": {"out_dir": "runs/test"}, "stability": True})
    with pytest.raises(ValueError, match="'stability' must be a mapping"):
        _ = cfg.stability_config()


def test_stability_config_invalid_bool_value_raises():
    cfg = OrionConfig({"run": {"out_dir": "runs/test"}, "stability": {"qk_norm": "definitely"}})
    with pytest.raises(ValueError, match="Invalid boolean value for stability.qk_norm"):
        _ = cfg.stability_config()


# ---- QK-norm ----


def test_qk_norm_output_shape_unchanged():
    sparse_cfg = _tiny_sparse_cfg()
    model_norm = _build_orion(StabilityConfig(qk_norm=True), attention_cfg=sparse_cfg)
    model_base = _build_orion(StabilityConfig(qk_norm=False), attention_cfg=sparse_cfg)
    idx = torch.randint(0, 64, (2, 16))
    out_norm = model_norm(idx)
    out_base = model_base(idx)
    assert out_norm.shape == out_base.shape


def test_qk_norm_bounds_score_magnitude():
    """Model with QK-norm should have lower mean absolute attn scores than without.

    We scale q/k projection weights up by 10x in both models. Without QK-norm, this
    inflates dot products proportionally. With QK-norm, RMSNorm bounds q and k before
    the dot product, so scores stay small regardless of weight magnitude.
    """
    from orion.models.blocks import DecoderBlock

    torch.manual_seed(42)
    sparse_cfg = _tiny_sparse_cfg()
    model_base = _build_orion(StabilityConfig(qk_norm=False), attention_cfg=sparse_cfg)
    torch.manual_seed(42)
    model_norm = _build_orion(StabilityConfig(qk_norm=True), attention_cfg=sparse_cfg)

    # Scale up q/k projections in both models — norm model will still bound scores
    for model in (model_base, model_norm):
        for block in model.modules():
            if isinstance(block, DecoderBlock):
                with torch.no_grad():
                    block.q_proj.weight.mul_(10.0)
                    block.k_proj.weight.mul_(10.0)

    idx = torch.randint(0, 64, (2, 32))
    model_base.eval()
    model_norm.eval()

    with torch.no_grad():
        model_base(idx)
        model_norm(idx)

    from orion.train import _find_attention_backends

    sparse_base, _, _ = _find_attention_backends(model_base)
    sparse_norm, _, _ = _find_attention_backends(model_norm)

    assert sparse_base is not None and sparse_norm is not None
    assert sparse_norm.last_attn_score_mean < sparse_base.last_attn_score_mean


def test_qk_norm_blocks_have_norm_layers():
    from orion.models.blocks import DecoderBlock

    model = _build_orion(StabilityConfig(qk_norm=True), attention_cfg=_tiny_sparse_cfg())
    for block in model.modules():
        if isinstance(block, DecoderBlock):
            assert block.q_norm is not None
            assert block.k_norm is not None


def test_no_qk_norm_blocks_have_none():
    from orion.models.blocks import DecoderBlock

    model = _build_orion(StabilityConfig(qk_norm=False), attention_cfg=_tiny_sparse_cfg())
    for block in model.modules():
        if isinstance(block, DecoderBlock):
            assert block.q_norm is None
            assert block.k_norm is None


# ---- Ortho init ----


def test_ortho_init_orthogonality():
    """After ortho init, q_proj.weight @ q_proj.weight.T ≈ I (for square weights)."""
    model = _build_orion(
        StabilityConfig(ortho_init=True),
        attention_cfg=_tiny_sparse_cfg(),
        d_model=32,
        n_heads=4,
    )
    from orion.models.blocks import DecoderBlock

    for block in model.modules():
        if isinstance(block, DecoderBlock):
            W = block.q_proj.weight  # [d_model, d_model]
            WWT = W @ W.T
            eye = torch.eye(W.shape[0])
            # Orthogonal: WW^T should be close to I
            assert torch.allclose(WWT, eye, atol=1e-4), (
                f"q_proj not orthogonal: max_err={((WWT - eye).abs().max()):.4f}"
            )
            break  # Check first block


# ---- Spectral norm ----


def test_spectral_norm_wraps_projections():
    model = _build_orion(StabilityConfig(spectral_norm=True), attention_cfg=_tiny_sparse_cfg())
    from orion.models.blocks import DecoderBlock

    for block in model.modules():
        if isinstance(block, DecoderBlock):
            assert hasattr(block.q_proj, "weight_orig"), (
                "q_proj missing weight_orig (spectral norm not applied)"
            )
            assert hasattr(block.k_proj, "weight_orig"), (
                "k_proj missing weight_orig (spectral norm not applied)"
            )
            break


def test_spectral_norm_no_wrap_without_flag():
    model = _build_orion(StabilityConfig(spectral_norm=False), attention_cfg=_tiny_sparse_cfg())
    from orion.models.blocks import DecoderBlock

    for block in model.modules():
        if isinstance(block, DecoderBlock):
            assert not hasattr(block.q_proj, "weight_orig")
            break


# ---- Combined toggles ----


def test_all_toggles_train_step():
    """All three toggles enabled — run 3 training steps without crash."""
    from orion.config import OrionConfig
    from orion.train import train

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dict = {
            "run": {"out_dir": tmpdir, "seed": 0, "steps": 3, "log_every": 1, "save_every": 3},
            "data": {"seq_len": 16, "batch_size": 2, "vocab_size": 64},
            "model": {"name": "orion", "d_model": 32, "n_layers": 2, "n_heads": 4, "mlp_mult": 2},
            "attention": {"backend": "sparse", "window_size": 4, "expander_degree": 2},
            "stability": {"qk_norm": True, "ortho_init": True, "spectral_norm": True},
        }
        cfg = OrionConfig(cfg_dict)
        train(cfg, device=torch.device("cpu"))
        assert (Path(tmpdir) / "checkpoint.pt").exists()
        assert (Path(tmpdir) / "metrics.jsonl").exists()


def test_stability_flags_in_run_metrics():
    """type:run JSON entry contains correct stability boolean fields."""
    from orion.config import OrionConfig
    from orion.train import train

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dict = {
            "run": {"out_dir": tmpdir, "seed": 0, "steps": 3, "log_every": 1, "save_every": 3},
            "data": {"seq_len": 16, "batch_size": 2, "vocab_size": 64},
            "model": {"name": "orion", "d_model": 32, "n_layers": 2, "n_heads": 4, "mlp_mult": 2},
            "attention": {"backend": "sparse", "window_size": 4, "expander_degree": 2},
            "stability": {"qk_norm": True, "ortho_init": False, "spectral_norm": True},
        }
        cfg = OrionConfig(cfg_dict)
        train(cfg, device=torch.device("cpu"))

        metrics_path = Path(tmpdir) / "metrics.jsonl"
        run_entry = None
        with open(metrics_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("type") == "run":
                    run_entry = row
                    break

        assert run_entry is not None, "No 'type: run' entry found in metrics.jsonl"
        assert run_entry["qk_norm"] is True
        assert run_entry["ortho_init"] is False
        assert run_entry["spectral_norm"] is True


def test_dense_backend_disables_stability_flags_in_run_metrics():
    """Dense backend should ignore stability controls and log all flags false."""
    from orion.config import OrionConfig
    from orion.train import train

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dict = {
            "run": {"out_dir": tmpdir, "seed": 0, "steps": 3, "log_every": 1, "save_every": 3},
            "data": {"seq_len": 16, "batch_size": 2, "vocab_size": 64},
            "model": {"name": "orion", "d_model": 32, "n_layers": 2, "n_heads": 4, "mlp_mult": 2},
            "attention": {"backend": "dense"},
            "stability": {"qk_norm": True, "ortho_init": True, "spectral_norm": True},
        }
        cfg = OrionConfig(cfg_dict)
        train(cfg, device=torch.device("cpu"))

        metrics_path = Path(tmpdir) / "metrics.jsonl"
        run_entry = None
        with open(metrics_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("type") == "run":
                    run_entry = row
                    break

        assert run_entry is not None, "No 'type: run' entry found in metrics.jsonl"
        assert run_entry["qk_norm"] is False
        assert run_entry["ortho_init"] is False
        assert run_entry["spectral_norm"] is False


def test_attn_score_mean_in_window_metrics():
    """attn_score_mean is present in type:window JSONL entries."""
    from orion.config import OrionConfig
    from orion.train import train

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dict = {
            "run": {"out_dir": tmpdir, "seed": 0, "steps": 50, "log_every": 50, "save_every": 50},
            "data": {"seq_len": 16, "batch_size": 2, "vocab_size": 64},
            "model": {"name": "orion", "d_model": 32, "n_layers": 2, "n_heads": 4, "mlp_mult": 2},
            "attention": {"backend": "dense"},
            "stability": {"qk_norm": True},
        }
        cfg = OrionConfig(cfg_dict)
        train(cfg, device=torch.device("cpu"))

        metrics_path = Path(tmpdir) / "metrics.jsonl"
        window_entry = None
        with open(metrics_path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("type") == "window":
                    window_entry = row
                    break

        assert window_entry is not None, "No 'type: window' entry found"
        assert "attn_score_mean" in window_entry
        assert isinstance(window_entry["attn_score_mean"], float)
        assert "attention_entropy_collapse" in window_entry
        assert isinstance(window_entry["attention_entropy_collapse"], bool)
