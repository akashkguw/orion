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
