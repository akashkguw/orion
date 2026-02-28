from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

from .config import load_config
from .eval import evaluate
from .run_paths import resolve_run_paths
from .train import train


def _git_meta() -> dict[str, str]:
    def _cmd(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return "unknown"

    return {
        "commit": _cmd(["git", "rev-parse", "HEAD"]),
        "branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(prog="orion.run")
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--base-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--resume", nargs="?", const="auto", default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg.raw, base_dir=args.base_dir, run_id_override=args.run_id)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    cfg.raw.setdefault("run", {})["out_dir"] = str(paths.out_dir)

    resolved_cfg_path = paths.out_dir / "config.resolved.yaml"
    resolved_cfg_path.write_text(yaml.safe_dump(cfg.raw, sort_keys=False), encoding="utf-8")

    started_at = datetime.now(UTC).isoformat()
    meta = {
        "run_id": paths.run_id,
        "started_at": started_at,
        "config_path": args.config,
        "resolved_config": str(resolved_cfg_path),
        "mode": args.mode,
        "device": args.device,
        **_git_meta(),
        "status": "running",
    }
    _write_json(paths.out_dir / "meta.json", meta)

    device = torch.device(args.device)
    checkpoint_latest = paths.out_dir / "checkpoint.pt"

    try:
        if args.mode in {"train", "both"}:
            resume_path = None
            if args.resume is not None:
                resume_path = checkpoint_latest if args.resume == "auto" else Path(args.resume)
                if args.resume == "auto" and not checkpoint_latest.exists():
                    resume_path = None

            train(
                cfg,
                device=device,
                resume_path=resume_path,
                save_every_override=args.save_every,
            )

        if args.mode in {"eval", "both"}:
            ckpt = args.checkpoint or str(checkpoint_latest)
            metrics = evaluate(cfg, checkpoint=ckpt, device=device)
            _write_json(paths.out_dir / "eval.json", metrics)

        meta["status"] = "completed"
    except Exception as exc:  # pragma: no cover
        meta["status"] = "failed"
        meta["error"] = str(exc)
        raise
    finally:
        meta["finished_at"] = datetime.now(UTC).isoformat()
        _write_json(paths.out_dir / "meta.json", meta)


if __name__ == "__main__":
    main()
