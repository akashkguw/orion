from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    out_dir: Path


def is_colab() -> bool:
    return Path("/content").exists()


def default_base_dir() -> Path:
    drive_root = Path("/content/drive/MyDrive")
    if is_colab() and drive_root.exists():
        return drive_root / "orion"
    return Path(".")


def make_run_id() -> str:
    return datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")


def resolve_run_paths(
    cfg_raw: dict[str, Any],
    *,
    base_dir: str | None = None,
    run_id_override: str | None = None,
) -> RunPaths:
    run_cfg = cfg_raw.setdefault("run", {})

    # If caller provides overrides, they take precedence over config out_dir.
    if base_dir is not None or run_id_override is not None:
        rid = str(run_id_override or run_cfg.get("run_id") or make_run_id())
        run_cfg["run_id"] = rid

        base = (
            Path(base_dir).expanduser()
            if base_dir is not None
            else Path(str(run_cfg.get("base_dir"))).expanduser()
            if run_cfg.get("base_dir")
            else default_base_dir()
        )
        out_dir = base / "runs" / rid
        return RunPaths(run_id=rid, out_dir=out_dir)

    # No overrides supplied: explicit out_dir in config wins.
    out_dir_cfg = run_cfg.get("out_dir")
    if out_dir_cfg:
        out_dir = Path(str(out_dir_cfg)).expanduser()
        run_id = str(run_cfg.get("run_id") or out_dir.name)
        return RunPaths(run_id=run_id, out_dir=out_dir)

    rid = str(run_cfg.get("run_id") or make_run_id())
    run_cfg["run_id"] = rid
    base = (
        Path(str(run_cfg.get("base_dir"))).expanduser()
        if run_cfg.get("base_dir")
        else default_base_dir()
    )

    out_dir = base / "runs" / rid
    return RunPaths(run_id=rid, out_dir=out_dir)
