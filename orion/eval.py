from __future__ import annotations

import torch

from .config import OrionConfig, load_config
from .model import loss_fn
from .models_factory import build_model
from .stability import any_stability_enabled, effective_stability_for_backend


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


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        pattern in text
        for pattern in (
            "cuda out of memory",
            "outofmemoryerror",
            "cudnn_status_alloc_failed",
            "cublas_status_alloc_failed",
            "resource exhausted",
            "out of memory",
        )
    )


def _build_and_load_model_for_eval(
    *,
    model_cfg: OrionConfig,
    ckpt: dict,
    vocab_size: int,
    device: torch.device,
) -> torch.nn.Module:
    """Build model matching checkpoint architecture and load weights."""
    d_model = int(model_cfg.get("model", "d_model", default=128))
    n_layers = int(model_cfg.get("model", "n_layers", default=2))
    n_heads = int(model_cfg.get("model", "n_heads", default=4))
    mlp_mult = int(model_cfg.get("model", "mlp_mult", default=4))
    model_name = str(model_cfg.get("model", "name", default="tiny"))
    max_seq_len = int(model_cfg.get("model", "max_seq_len", default=4096))
    attention_cfg = model_cfg.attention_config()
    raw_stability_cfg = model_cfg.stability_config()
    effective_stability_cfg = effective_stability_for_backend(
        raw_stability_cfg, attention_backend=attention_cfg.backend
    )

    model = build_model(
        name=model_name,
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_mult=mlp_mult,
        device=device,
        attention_cfg=attention_cfg,
        stability_cfg=effective_stability_cfg,
        max_seq_len=max_seq_len,
    )

    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError:
        # Backward compatibility for older checkpoints that applied stability to non-sparse backends.
        if effective_stability_cfg != raw_stability_cfg and any_stability_enabled(
            raw_stability_cfg
        ):
            model = build_model(
                name=model_name,
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                mlp_mult=mlp_mult,
                device=device,
                attention_cfg=attention_cfg,
                stability_cfg=raw_stability_cfg,
                max_seq_len=max_seq_len,
            )
            model.load_state_dict(ckpt["model"], strict=True)
        else:
            raise
    model.eval()
    return model


@torch.no_grad()
def _evaluate_single_context_ppl(
    model: torch.nn.Module,
    *,
    vocab_size: int,
    context_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Evaluate perplexity for one context length with CUDA OOM backoff."""
    local_batch = max(1, int(batch_size))

    while True:
        try:
            x = torch.randint(0, vocab_size, (local_batch, context_len), device=device)
            y = torch.roll(x, shifts=-1, dims=1)
            logits = model(x)
            loss = loss_fn(logits, y)
            return float(torch.exp(loss).clamp(max=1e6).item())
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if device.type != "cuda" or not _is_cuda_oom(exc):
                raise
            if local_batch <= 1:
                torch.cuda.empty_cache()
                return float("nan")
            local_batch = max(1, local_batch // 2)
            torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(cfg: OrionConfig, *, checkpoint: str, device: torch.device) -> dict[str, float]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Use the config saved at training time for model architecture so the
    # model we build exactly matches the checkpoint — not the current yaml,
    # which may differ (e.g. name changed from "tiny" to "orion").
    if "config" in ckpt:
        model_cfg = OrionConfig(raw=ckpt["config"])
    else:
        model_cfg = cfg

    vocab_size = _resolve_vocab_size(model_cfg, ckpt)

    # Eval batch params can still come from the current yaml (they don't affect weights)
    seq_len = int(cfg.get("data", "seq_len", default=128))
    batch_size = int(cfg.get("data", "batch_size", default=8))

    model = _build_and_load_model_for_eval(
        model_cfg=model_cfg,
        ckpt=ckpt,
        vocab_size=vocab_size,
        device=device,
    )

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

    Returns:
        Dictionary with eval_ppl_512, eval_ppl_1024, eval_ppl_2048, eval_ppl_4096
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    if "config" in ckpt:
        model_cfg = OrionConfig(raw=ckpt["config"])
    else:
        model_cfg = cfg

    vocab_size = _resolve_vocab_size(model_cfg, ckpt)
    train_batch_size = int(cfg.get("data", "batch_size", default=8))
    eval_batch_override = cfg.get("eval", "long_context_batch_size", default=None)
    if eval_batch_override is not None:
        base_eval_batch_size = int(eval_batch_override)
    else:
        # Dense long-context eval can OOM if we reuse the training batch size.
        base_eval_batch_size = 1 if device.type == "cuda" else train_batch_size
    base_eval_batch_size = max(1, base_eval_batch_size)

    model = _build_and_load_model_for_eval(
        model_cfg=model_cfg,
        ckpt=ckpt,
        vocab_size=vocab_size,
        device=device,
    )

    results = {}
    for context_len in [512, 1024, 2048, 4096]:
        results[f"eval_ppl_{context_len}"] = _evaluate_single_context_ppl(
            model,
            vocab_size=vocab_size,
            context_len=context_len,
            batch_size=base_eval_batch_size,
            device=device,
        )

    # Ensure temporary CUDA allocations from eval are promptly released.
    if device.type == "cuda":
        del model
        del ckpt
        torch.cuda.empty_cache()

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
