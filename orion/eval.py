from __future__ import annotations
import torch
from pathlib import Path
from .config import load_config
from .model import TinyDecoderOnly, loss_fn


@torch.no_grad()
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    vocab_size = int(cfg.get("data", "vocab_size", default=256))
    seq_len = int(cfg.get("data", "seq_len", default=128))
    batch_size = int(cfg.get("data", "batch_size", default=8))

    d_model = int(cfg.get("model", "d_model", default=128))
    n_layers = int(cfg.get("model", "n_layers", default=2))
    n_heads = int(cfg.get("model", "n_heads", default=4))
    mlp_mult = int(cfg.get("model", "mlp_mult", default=4))

    model = TinyDecoderOnly(vocab_size, d_model, n_layers, n_heads, mlp_mult).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.roll(x, shifts=-1, dims=1)
    logits = model(x)
    loss = loss_fn(logits, y)
    ppl = float(torch.exp(loss).clamp(max=1e6).item())
    print({"loss": float(loss.item()), "ppl": ppl})


if __name__ == "__main__":
    main()

