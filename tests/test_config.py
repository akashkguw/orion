import tempfile
from pathlib import Path

from orion.config import OrionConfig, load_config


def test_orion_config_get():
    """Test hierarchical config access."""
    cfg = OrionConfig({"model": {"d_model": 128, "n_layers": 2}})
    assert cfg.get("model", "d_model") == 128
    assert cfg.get("model", "n_layers") == 2
    assert cfg.get("model", "missing", default=999) == 999


def test_orion_config_out_dir():
    """Test output directory path generation."""
    cfg = OrionConfig({"run": {"out_dir": "runs/test_run"}})
    out_dir = cfg.out_dir
    assert isinstance(out_dir, Path)
    assert "test_run" in str(out_dir)


def test_load_config_from_yaml():
    """Test loading config from YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("model:\n  d_model: 256\n  n_layers: 4\n")
        f.flush()

        cfg = load_config(f.name)
        assert cfg.get("model", "d_model") == 256
        assert cfg.get("model", "n_layers") == 4

        Path(f.name).unlink()


def test_load_config_nested_access():
    """Test nested config access with defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            "data:\n  seq_len: 512\nmodel:\n  d_model: 512\n  n_heads: 8\nrun:\n  out_dir: runs/exp\n"
        )
        f.flush()

        cfg = load_config(f.name)
        assert cfg.get("data", "seq_len") == 512
        assert cfg.get("data", "batch_size", default=32) == 32
        assert cfg.get("model", "d_model") == 512
        assert cfg.get("model", "n_heads") == 8

        Path(f.name).unlink()


def test_load_config_invalid_yaml():
    """Test loading invalid YAML raises error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [")
        f.flush()

        try:
            load_config(f.name)
            raise AssertionError("Should have raised an error")
        except Exception:
            pass

        Path(f.name).unlink()


def test_config_raw_access():
    """Test direct access to raw config dict."""
    cfg = OrionConfig({"key": "value", "nested": {"inner": 42}})
    assert cfg.raw["key"] == "value"
    assert cfg.raw["nested"]["inner"] == 42


def test_attention_config_defaults():
    """Missing attention section defaults to dense with no window/expander."""
    cfg = OrionConfig({"run": {"out_dir": "/tmp"}})
    acfg = cfg.attention_config()
    assert acfg.backend == "dense"
    assert acfg.window_size is None
    assert acfg.expander_degree is None


def test_attention_config_from_yaml():
    """Attention section is parsed into AttentionConfig with correct types."""
    cfg = OrionConfig(
        {
            "run": {"out_dir": "/tmp"},
            "attention": {"backend": "sparse", "window_size": 64, "expander_degree": 4},
        }
    )
    acfg = cfg.attention_config()
    assert acfg.backend == "sparse"
    assert acfg.window_size == 64
    assert acfg.expander_degree == 4
