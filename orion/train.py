from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from .config import OrionConfig, load_config
from .logging_utils import JsonlLogger
from .metrics import MetricsTracker, metrics_to_dict
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
    metrics_tracker = MetricsTracker(window_size=50)

    # Reset VRAM stats after model init to avoid counting initialization allocations
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def compute_epoch(step: int) -> int:
        if steps_per_epoch:
            return (step - 1) // steps_per_epoch + 1
        return step

    # Get attention config for metrics
    window_size = int(cfg.get("model", "window_size", default=64))
    expander_degree = int(cfg.get("model", "expander_degree", default=8))

    # -------- Train loop --------
    model.train()
    if start_step >= steps:
        return

    # Log run metrics once at start (only on first training, not on resume)
    if start_step == 0:
        run_metrics = metrics_tracker.record_run_metrics(
            step=0,
            window_size=window_size,
            expander_degree=expander_degree,
            batch_size=batch_size,
            seq_len=seq_len,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        logger.log({"type": "run", **metrics_to_dict(run_metrics)})

    for step in range(start_step + 1, steps + 1):
        step_time_start = time.time()

        x, y = get_batch("train")
        logits, residual = model(x, return_residual=True)
        loss = loss_fn(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Compute gradient norm (pre-clip)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))

        opt.step()
        if scheduler is not None:
            scheduler.step()

        step_time_sec = time.time() - step_time_start
        step_time_ms = step_time_sec * 1000.0
        throughput = metrics_tracker.compute_throughput(batch_size, seq_len, step_time_sec)

        # Compute top-1 accuracy
        accuracy_top1 = metrics_tracker.compute_top1_accuracy(logits, y)

        # Get current learning rate
        learning_rate = opt.param_groups[0]["lr"]

        # Record step metrics (every step)
        step_metrics = metrics_tracker.record_step_metrics(
            step=step,
            loss=float(loss.item()),
            grad_norm=grad_norm,
            throughput=throughput,
            step_time_ms=step_time_ms,
            accuracy_top1=accuracy_top1,
            learning_rate=learning_rate,
        )
        logger.log({"type": "step", **metrics_to_dict(step_metrics)})

        # Log every log_every steps
        if step % log_every == 0:
            print(
                f"Step {step}: loss={step_metrics.loss:.4f}, ppl={step_metrics.ppl:.2f}, "
                f"throughput={throughput:.1f} tok/s, grad_norm={grad_norm:.4f}, "
                f"clipped={step_metrics.grad_clipped}"
            )

        # Record window metrics every 50 steps
        if step % 50 == 0:
            vram_peak_mib = 0
            if device.type == "cuda":
                vram_peak_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
                torch.cuda.reset_peak_memory_stats()

            # Compute activation norm from residual stream (detached to avoid graph retention)
            r = residual.detach()
            activation_norm = float(torch.sqrt((r.float() ** 2).mean()).item())

            # Compute attention entropy if using sparse or dense attention
            # Note: Attention entropy is available for SparseAttention and DenseAttention models.
            # For TinyDecoderOnly (which uses nn.TransformerEncoderLayer with nn.MultiheadAttention),
            # entropy will be 0.0 since nn.MultiheadAttention can return weights, but
            # nn.TransformerEncoderLayer/Encoder doesn't provide a clean way to retrieve them
            # externally without modifying/replacing modules or using hooks.
            attention_entropy = 0.0
            attention_entropy_normalized = 0.0
            valid_neighbor_fraction = 0.0
            attention_mass_window_pct = 0.0
            attention_mass_expander_pct = 0.0

            # Try to get attention weights from the first layer's attention backend
            try:
                from .attention.dense import DenseAttention
                from .attention.sparse import SparseAttention

                # Handle both OrionDecoder (ModuleList) and TinyDecoderOnly (TransformerEncoder)
                first_block = None
                if hasattr(model.blocks, "__getitem__"):  # ModuleList
                    first_block = model.blocks[0]
                elif hasattr(model.blocks, "layers"):  # TransformerEncoder
                    first_block = model.blocks.layers[0]

                if first_block is not None:
                    attn_backend = getattr(first_block, "attn", None) or getattr(
                        first_block, "self_attn", None
                    )
                    if attn_backend is not None:
                        attn_backend = getattr(attn_backend, "attn_backend", attn_backend)
                        if isinstance(attn_backend, (DenseAttention, SparseAttention)) and hasattr(
                            attn_backend, "last_attn_weights"
                        ):
                            weights = attn_backend.last_attn_weights
                            if weights is not None:
                                # For dense attention, use full sequence length as degree
                                if isinstance(attn_backend, DenseAttention):
                                    degree = weights.shape[-1]  # T (full sequence)
                                else:  # SparseAttention
                                    degree = window_size + expander_degree
                                entropy, entropy_norm = metrics_tracker.compute_attention_entropy(
                                    weights, degree
                                )
                                attention_entropy = entropy
                                attention_entropy_normalized = entropy_norm

                                # Compute sparse-specific metrics
                                if isinstance(attn_backend, SparseAttention):
                                    valid_neighbor_fraction = (
                                        metrics_tracker.compute_valid_neighbor_fraction(weights)
                                    )
                                    (
                                        attention_mass_window_pct,
                                        attention_mass_expander_pct,
                                    ) = metrics_tracker.compute_attention_mass_split(
                                        weights, window_size
                                    )
            except (AttributeError, IndexError, TypeError):
                pass  # Silently skip if path doesn't exist or is wrong type

            window_metrics = metrics_tracker.record_window_metrics(
                step=step,
                vram_peak_mib=vram_peak_mib,
                activation_norm=activation_norm,
                attention_entropy=attention_entropy,
                attention_entropy_normalized=attention_entropy_normalized,
                valid_neighbor_fraction=valid_neighbor_fraction,
                attention_mass_window_pct=attention_mass_window_pct,
                attention_mass_expander_pct=attention_mass_expander_pct,
            )
            logger.log({"type": "window", **metrics_to_dict(window_metrics)})

        # Save checkpoint before eval
        if step % save_every == 0 or step % 1000 == 0:
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

        # Evaluate at long context lengths every 1000 steps (after checkpoint save)
        if step % 1000 == 0:
            from .eval import evaluate_long_context

            eval_results = evaluate_long_context(
                cfg, checkpoint=str(out_dir / "checkpoint.pt"), device=device
            )
            eval_metrics = metrics_tracker.record_eval_metrics(
                step=step,
                eval_ppl_512=eval_results.get("eval_ppl_512", 0.0),
                eval_ppl_1024=eval_results.get("eval_ppl_1024", 0.0),
                eval_ppl_2048=eval_results.get("eval_ppl_2048", 0.0),
                eval_ppl_4096=eval_results.get("eval_ppl_4096", 0.0),
            )
            logger.log({"type": "eval", **metrics_to_dict(eval_metrics)})

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
