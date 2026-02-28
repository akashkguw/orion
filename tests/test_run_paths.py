from pathlib import Path

from orion.run_paths import resolve_run_paths


def test_resolve_run_paths_uses_explicit_out_dir():
    cfg = {"run": {"out_dir": "/tmp/custom-out", "run_id": "abc"}}
    paths = resolve_run_paths(cfg)
    assert str(paths.out_dir) == "/tmp/custom-out"
    assert paths.run_id == "abc"


def test_resolve_run_paths_builds_from_base_dir_and_run_id():
    cfg = {"run": {"base_dir": "/tmp/orion", "run_id": "abc123"}}
    paths = resolve_run_paths(cfg)
    assert paths.out_dir == Path("/tmp/orion") / "runs" / "abc123"
    assert paths.run_id == "abc123"


def test_resolve_run_paths_applies_run_id_override():
    cfg = {"run": {"base_dir": "/tmp/orion", "run_id": "old"}}
    paths = resolve_run_paths(cfg, run_id_override="new-id")
    assert paths.run_id == "new-id"
    assert paths.out_dir == Path("/tmp/orion") / "runs" / "new-id"


def test_overrides_take_precedence_over_out_dir():
    cfg = {"run": {"out_dir": "/tmp/from-config", "run_id": "cfg-id"}}
    paths = resolve_run_paths(cfg, base_dir="/tmp/override-base", run_id_override="override-id")
    assert paths.run_id == "override-id"
    assert paths.out_dir == Path("/tmp/override-base") / "runs" / "override-id"
