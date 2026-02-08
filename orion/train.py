from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from .config import load_config
from .logging_utils import JsonlLogger
from .model import TinyDecoderOnly, loss_fn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    dataset = str(cfg.get("data", "dataset", default="toy")).lower()
    data_root = str(cfg.get("data", "root", default="data"))

    out_dir: Path = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("run", "seed", default=123))
    set_seed(seed)

    device = torch.device(args.device)

    seq_len = int(cfg.get("data", "seq_len", default=128))
    batch_size = int(cfg.get("data", "batch_size", default=8))

    d_model = int(cfg.get("model", "d_model", default=128))
    n_layers = int(cfg.get("model", "n_layers", default=2))
    n_heads = int(cfg.get("model", "n_heads", default=4))
    mlp_mult = int(cfg.get("model", "mlp_mult", default=4))

    steps = int(cfg.get("run", "steps", default=50))
    log_every = int(cfg.get("run", "log_every", default=1))
    save_every = int(cfg.get("run", "save_every", default=steps))

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
    model = TinyDecoderOnly(vocab_size, d_model, n_layers, n_heads, mlp_mult).to(device)
    opt = AdamW(model.parameters(), lr=float(cfg.get("optim", "lr", default=3e-4)))

    logger = JsonlLogger(out_dir / "metrics.jsonl")

    # -------- Train loop --------
    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch("train")
        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % log_every == 0:
            with torch.no_grad():
                ppl = float(torch.exp(loss).clamp(max=1e6).item())
            row = {"step": step, "loss": float(loss.item()), "ppl": ppl}
            if device.type == "cuda":
                row["vram_max_mb"] = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
            logger.log(row)
            print(row)

        if step % save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "seed": seed,
                "config": cfg.raw,
            }
            torch.save(ckpt, out_dir / "checkpoint.pt")

    # Always save last
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": steps, "seed": seed, "config": cfg.raw},
        out_dir / "checkpoint.pt",
    )


if __name__ == "__main__":
    main()
