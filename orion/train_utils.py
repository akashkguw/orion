from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import OrionConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    rng_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def _coerce_byte_tensor(state: Any) -> torch.ByteTensor:
    if isinstance(state, torch.Tensor):
        return state.detach().to(dtype=torch.uint8, device="cpu")
    if isinstance(state, (list, tuple)):
        return torch.tensor(state, dtype=torch.uint8)
    raise TypeError(f"Unsupported RNG state type: {type(state)!r}")


def restore_rng_state(rng_state: dict[str, Any] | None) -> None:
    if not rng_state:
        return
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.random.set_rng_state(_coerce_byte_tensor(rng_state["torch"]))
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        cuda_states = rng_state["cuda"]
        if isinstance(cuda_states, (list, tuple)):
            torch.cuda.set_rng_state_all([_coerce_byte_tensor(s) for s in cuda_states])
        else:
            torch.cuda.set_rng_state_all([_coerce_byte_tensor(cuda_states)])


def load_last_wall_time(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    last_line: str | None = None
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line
    if not last_line:
        return None
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError:
        return None
    wall_time = payload.get("wall_time_s")
    if isinstance(wall_time, (int, float)):
        return float(wall_time)
    return None


def build_scheduler(opt: torch.optim.Optimizer, cfg: OrionConfig):
    sched_cfg = cfg.get("optim", "scheduler", default=None)
    if not isinstance(sched_cfg, dict):
        return None
    name = str(sched_cfg.get("name", "")).lower()
    if not name:
        return None
    kwargs = {k: v for k, v in sched_cfg.items() if k != "name"}
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, **kwargs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, **kwargs)
    raise ValueError(f"Unknown scheduler name: {name!r}")


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    step: int,
    epoch: int,
    seed: int,
    cfg: OrionConfig,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "seed": seed,
        "config": cfg.raw,
        "rng_state": capture_rng_state(),
    }
    torch.save(ckpt, path)
