from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.optim import AdamW

from .config import OrionConfig, load_config
from .logging_utils import JsonlLogger
from .model import loss_fn
from .models_factory import build_model
from .train_utils import (
    build_scheduler,
    load_last_wall_time,
    restore_rng_state,
    save_checkpoint,
    set_seed,
)


def train(
    cfg: OrionConfig,
    *,
    device: torch.device,
    resume_path: Path | None = None,
    save_every_override: int | None = None,
) -> None:
    dataset = str(cfg.get("data", "dataset", default="toy")).lower()
    data_root = str(cfg.get("data", "root", default="data"))

    out_dir: Path = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("run", "seed", default=123))
    set_seed(seed)

    seq_len = int(cfg.get("data", "seq_len", default=128))
    batch_size = int(cfg.get("data", "batch_size", default=8))

    d_model = int(cfg.get("model", "d_model", default=128))
    n_layers = int(cfg.get("model", "n_layers", default=2))
    n_heads = int(cfg.get("model", "n_heads", default=4))
    mlp_mult = int(cfg.get("model", "mlp_mult", default=4))

    steps = int(cfg.get("run", "steps", default=50))
    log_every = int(cfg.get("run", "log_every", default=1))
    save_every = int(cfg.get("run", "save_every", default=steps))
    if save_every_override is not None:
        save_every = int(save_every_override)
    steps_per_epoch = cfg.get("run", "steps_per_epoch", default=None)
    if steps_per_epoch is not None:
        steps_per_epoch = int(steps_per_epoch)
        if steps_per_epoch <= 0:
            steps_per_epoch = None

    # Override steps if running smoke test
    if os.getenv("SMOKE_TEST") == "true":
        steps = int(os.getenv("SMOKE_STEPS", "20"))

    # -------- Dataset selection --------
    if dataset in {"tinyshakespeare", "shakespeare"}:
        from orion.data.shakespeare import load_tiny_shakespeare, sample_batch

        train_ids, val_ids, tok = load_tiny_shakespeare(data_root)
        vocab_size = tok.vocab_size

        def get_batch(split: str):
            ids = train_ids if split == "train" else val_ids
            return sample_batch(ids, batch_size=batch_size, seq_len=seq_len, device=device)

    else:
        vocab_size = int(cfg.get("data", "vocab_size", default=256))

        def get_batch(split: str):
            # toy data: random tokens; shift as next-token prediction
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            y = torch.roll(x, shifts=-1, dims=1)
            return x, y

    # -------- Model / Optim --------
    model_name = str(cfg.get("model", "name", default="tiny"))
    attention_cfg = cfg.attention_config()

    model = build_model(
        name=model_name,
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_mult=mlp_mult,
        device=device,
        attention_cfg=attention_cfg,
    )

    opt = AdamW(model.parameters(), lr=float(cfg.get("optim", "lr", default=3e-4)))
    scheduler = build_scheduler(opt, cfg)

    start_step = 0
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0))
        restore_rng_state(ckpt.get("rng_state"))

    metrics_path = out_dir / "metrics.jsonl"
    wall_time_offset = load_last_wall_time(metrics_path) if resume_path else None
    logger = JsonlLogger(metrics_path, wall_time_offset=wall_time_offset)

    def compute_epoch(step: int) -> int:
        if steps_per_epoch:
            return (step - 1) // steps_per_epoch + 1
        return step

    # -------- Train loop --------
    model.train()
    if start_step >= steps:
        return
    for step in range(start_step + 1, steps + 1):
        x, y = get_batch("train")
        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if step % log_every == 0:
            with torch.no_grad():
                ppl = float(torch.exp(loss).clamp(max=1e6).item())
            row = {"step": step, "loss": float(loss.item()), "ppl": ppl}
            if device.type == "cuda":
                row["vram_max_mb"] = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
            logger.log(row)
            print(row)

        if step % save_every == 0:
            save_checkpoint(
                out_dir / "checkpoint.pt",
                model=model,
                opt=opt,
                scheduler=scheduler,
                step=step,
                epoch=compute_epoch(step),
                seed=seed,
                cfg=cfg,
            )

    # Always save last
    save_checkpoint(
        out_dir / "checkpoint.pt",
        model=model,
        opt=opt,
        scheduler=scheduler,
        step=steps,
        epoch=compute_epoch(steps),
        seed=seed,
        cfg=cfg,
    )


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", nargs="?", const="auto", default=None)
    p.add_argument("--save-every", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    resume_path = None
    if args.resume is not None:
        resume_path = cfg.out_dir / "checkpoint.pt" if args.resume == "auto" else Path(args.resume)

    train(
        cfg,
        device=device,
        resume_path=resume_path,
        save_every_override=args.save_every,
    )


if __name__ == "__main__":
    main()
