from __future__ import annotations

import torch

from .config import OrionConfig, load_config
from .model import loss_fn
from .models_factory import build_model


def _infer_vocab_size_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int | None:
    """Infer vocab size from common embedding/head parameters."""
    for key in ("head.weight", "tok_emb.weight", "tok.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
            return int(tensor.shape[0])
    return None


def _resolve_vocab_size(cfg: OrionConfig, ckpt: dict) -> int:
    """Prefer checkpoint vocab size to avoid eval-time shape mismatches."""
    cfg_vocab = cfg.get("data", "vocab_size", default=None)
    cfg_vocab_size = int(cfg_vocab) if cfg_vocab is not None else None

    model_state = ckpt.get("model")
    if isinstance(model_state, dict):
        ckpt_vocab_size = _infer_vocab_size_from_state_dict(model_state)
        if ckpt_vocab_size is not None:
            return ckpt_vocab_size

    if cfg_vocab_size is not None:
        return cfg_vocab_size
    return 256


@torch.no_grad()
def evaluate(cfg: OrionConfig, *, checkpoint: str, device: torch.device) -> dict[str, float]:
    seq_len = int(cfg.get("data", "seq_len", default=128))
    batch_size = int(cfg.get("data", "batch_size", default=8))

    d_model = int(cfg.get("model", "d_model", default=128))
    n_layers = int(cfg.get("model", "n_layers", default=2))
    n_heads = int(cfg.get("model", "n_heads", default=4))
    mlp_mult = int(cfg.get("model", "mlp_mult", default=4))

    model_name = str(cfg.get("model", "name", default="tiny"))
    attention_cfg = cfg.attention_config()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    vocab_size = _resolve_vocab_size(cfg, ckpt)

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

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.roll(x, shifts=-1, dims=1)
    logits = model(x)
    loss = loss_fn(logits, y)
    ppl = float(torch.exp(loss).clamp(max=1e6).item())
    return {"loss": float(loss.item()), "ppl": ppl}


@torch.no_grad()
def evaluate_long_context(
    cfg: OrionConfig, *, checkpoint: str, device: torch.device
) -> dict[str, float]:
    """Evaluate model at multiple context lengths (512, 1024, 2048, 4096).

    Args:
        cfg: Configuration
        checkpoint: Path to checkpoint
        device: Device to evaluate on

    Returns:
        Dictionary with eval_ppl_512, eval_ppl_1024, eval_ppl_2048, eval_ppl_4096
    """
    batch_size = int(cfg.get("data", "batch_size", default=8))

    d_model = int(cfg.get("model", "d_model", default=128))
    n_layers = int(cfg.get("model", "n_layers", default=2))
    n_heads = int(cfg.get("model", "n_heads", default=4))
    mlp_mult = int(cfg.get("model", "mlp_mult", default=4))

    model_name = str(cfg.get("model", "name", default="tiny"))
    attention_cfg = cfg.attention_config()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    vocab_size = _resolve_vocab_size(cfg, ckpt)

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

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    results = {}
    for context_len in [512, 1024, 2048, 4096]:
        x = torch.randint(0, vocab_size, (batch_size, context_len), device=device)
        y = torch.roll(x, shifts=-1, dims=1)
        logits = model(x)
        loss = loss_fn(logits, y)
        ppl = float(torch.exp(loss).clamp(max=1e6).item())
        results[f"eval_ppl_{context_len}"] = ppl

    return results


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
    print(evaluate(cfg, checkpoint=args.checkpoint, device=device))


if __name__ == "__main__":
    main()
