from __future__ import annotations

import copy
import csv
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .attention.base import AttentionConfig
from .data.shakespeare import load_tiny_shakespeare
from .model import loss_fn
from .models_factory import build_model
from .stability import StabilityConfig, any_stability_enabled, effective_stability_for_backend

SUPPORTED_BACKENDS = {"dense", "window", "sparse"}


@dataclass(frozen=True)
class TrialSpec:
    variant_id: str
    variant_cfg_path: str
    backend: str
    sparse_tag: str
    window_size: int | None
    expander_degree: int | None
    seq_len: int
    seed: int
    lr: float

    @property
    def trial_id(self) -> str:
        lr_tag = str(self.lr).replace(".", "p")
        return f"{self.variant_id}_T{self.seq_len}_seed{self.seed}_lr{lr_tag}"


@dataclass(frozen=True)
class VariantDefinition:
    variant_id: str
    cfg_path: Path
    backend: str
    sparse_tag: str
    window_size: int | None
    expander_degree: int | None


@dataclass(frozen=True)
class ProfileContext:
    profile_name: str
    profile_path: Path
    repo_root: Path
    raw_profile: dict[str, Any]
    strict_apples_to_apples: bool
    train_tokens_target: int
    seeds: list[int]
    seq_lens: list[int]
    lr_grid: list[float]
    fixed_batch_by_seq: dict[int, int]
    batch_candidates_by_seq: dict[int, list[int]]
    variant_entries: list[dict[str, Any]]
    run_val_eval: bool
    val_eval_batches: int
    run_model_only_bench: bool
    bench_warmup: int
    bench_iters: int
    eval_long_context_batch_size: int | None
    disable_probe_metrics_for_apples_to_apples: bool
    auto_retry_oom: bool
    auto_retry_dense_cuda_fail: bool
    overwrite_default: bool
    val_ppl_tolerance: float
    log_every: int
    save_every: int | str


@dataclass(frozen=True)
class RunResult:
    profile: str
    experiment_id: str
    runs_root: Path
    generated_cfg_root: Path
    summary_csv: Path
    trial_count: int


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get_hardware_meta() -> dict[str, Any]:
    gpu_name = "cpu"
    gpu_count = 0
    cuda_version = None
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": cuda_version,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "git_commit": get_git_commit(),
    }


def _cfg_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return default


def _cfg_int_map(raw: dict[str, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    for key, value in raw.items():
        out[int(key)] = int(value)
    return out


def _cfg_int_list_map(raw: dict[str, Any]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for key, value in raw.items():
        out[int(key)] = [int(item) for item in list(value)]
    return out


def load_profile_context(
    profile: str,
    *,
    repo_root: Path | None = None,
    profiles_root: Path | None = None,
) -> ProfileContext:
    repo = Path.cwd() if repo_root is None else Path(repo_root)
    root = (
        repo / "configs" / "experiments" / "profiles"
        if profiles_root is None
        else Path(profiles_root)
    )

    profile_path = root / f"{profile}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing profile config: {profile_path}")

    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Profile config must be a YAML mapping: {profile_path}")

    strict = _cfg_bool(raw.get("strict_apples_to_apples"), True)
    train_tokens_target = int(raw.get("train_tokens_target", 4_000_000))

    seeds = [int(v) for v in raw.get("seeds", [123])]
    seq_lens = [int(v) for v in raw.get("seq_lens", [256, 512, 1024])]
    lr_grid = [float(v) for v in raw.get("lr_grid", [3e-4])]

    fixed_batch_by_seq = _cfg_int_map(dict(raw.get("fixed_batch_by_seq", {})))
    batch_candidates_by_seq = _cfg_int_list_map(dict(raw.get("batch_candidates_by_seq", {})))

    variant_entries = list(raw.get("variants", []))
    if not variant_entries:
        raise ValueError(f"Profile has no variants: {profile_path}")

    runner = raw.get("runner", {}) or {}
    if not isinstance(runner, dict):
        raise ValueError(f"runner must be a mapping in {profile_path}")

    analysis = raw.get("analysis", {}) or {}
    if not isinstance(analysis, dict):
        raise ValueError(f"analysis must be a mapping in {profile_path}")

    save_every_raw = runner.get("save_every", "final")
    if isinstance(save_every_raw, str):
        save_every = save_every_raw.strip().lower()
        if save_every not in {"final", "steps", "default"}:
            raise ValueError(
                f"runner.save_every must be one of final|steps|default or int, got {save_every_raw!r}"
            )
    else:
        save_every = int(save_every_raw)

    auto_retry_oom = _cfg_bool(runner.get("auto_retry_oom"), not strict)
    auto_retry_dense_cuda_fail = _cfg_bool(runner.get("auto_retry_dense_cuda_fail"), not strict)

    eval_batch_raw = runner.get("eval_long_context_batch_size", 1)
    eval_long_context_batch_size = None if eval_batch_raw is None else int(eval_batch_raw)

    return ProfileContext(
        profile_name=profile,
        profile_path=profile_path,
        repo_root=repo,
        raw_profile=raw,
        strict_apples_to_apples=strict,
        train_tokens_target=train_tokens_target,
        seeds=seeds,
        seq_lens=seq_lens,
        lr_grid=lr_grid,
        fixed_batch_by_seq=fixed_batch_by_seq,
        batch_candidates_by_seq=batch_candidates_by_seq,
        variant_entries=variant_entries,
        run_val_eval=_cfg_bool(runner.get("run_val_eval"), True),
        val_eval_batches=int(runner.get("val_eval_batches", 40)),
        run_model_only_bench=_cfg_bool(runner.get("run_model_only_bench"), True),
        bench_warmup=int(runner.get("bench_warmup", 5)),
        bench_iters=int(runner.get("bench_iters", 20)),
        eval_long_context_batch_size=eval_long_context_batch_size,
        disable_probe_metrics_for_apples_to_apples=_cfg_bool(
            runner.get("disable_probe_metrics_for_apples_to_apples"), strict
        ),
        auto_retry_oom=auto_retry_oom,
        auto_retry_dense_cuda_fail=auto_retry_dense_cuda_fail,
        overwrite_default=_cfg_bool(runner.get("overwrite"), False),
        val_ppl_tolerance=float(analysis.get("val_ppl_tolerance", 0.20)),
        log_every=int(runner.get("log_every", 10)),
        save_every=save_every,
    )


def _derive_sparse_tag(backend: str, attention_cfg: dict[str, Any], entry: dict[str, Any]) -> str:
    explicit = entry.get("sparse_tag")
    if explicit:
        return str(explicit)
    if backend == "dense":
        return "dense"
    if backend == "window":
        return f"w{int(attention_cfg.get('window_size', 64))}"
    if backend == "sparse":
        window_size = int(attention_cfg.get("window_size", 64))
        expander_degree = int(attention_cfg.get("expander_degree", 8))
        return f"w{window_size}_d{expander_degree}"
    return backend


def load_variant_definitions(ctx: ProfileContext) -> list[VariantDefinition]:
    defs: list[VariantDefinition] = []

    for entry in ctx.variant_entries:
        variant_id = str(entry.get("id", "")).strip()
        if not variant_id:
            raise ValueError(f"Variant entry missing 'id': {entry}")

        cfg_rel = str(entry.get("config", "")).strip()
        if not cfg_rel:
            raise ValueError(f"Variant '{variant_id}' missing 'config' path")

        cfg_path = Path(cfg_rel)
        if not cfg_path.is_absolute():
            cfg_path = (ctx.repo_root / cfg_path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Variant '{variant_id}' config not found: {cfg_path}")

        cfg_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg_raw, dict):
            raise ValueError(f"Variant config must be mapping: {cfg_path}")

        attention_cfg = cfg_raw.get("attention", {}) or {}
        if not isinstance(attention_cfg, dict):
            attention_cfg = {}

        backend = str(attention_cfg.get("backend", entry.get("backend", ""))).lower().strip()
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Variant '{variant_id}' has unsupported backend '{backend}' in {cfg_path}"
            )

        window_size = int(attention_cfg["window_size"]) if "window_size" in attention_cfg else None
        expander_degree = (
            int(attention_cfg["expander_degree"]) if "expander_degree" in attention_cfg else None
        )

        defs.append(
            VariantDefinition(
                variant_id=variant_id,
                cfg_path=cfg_path,
                backend=backend,
                sparse_tag=_derive_sparse_tag(backend, attention_cfg, entry),
                window_size=window_size,
                expander_degree=expander_degree,
            )
        )

    return defs


def build_trial_specs(ctx: ProfileContext, variants: list[VariantDefinition]) -> list[TrialSpec]:
    specs: list[TrialSpec] = []

    for seed in ctx.seeds:
        for seq_len in ctx.seq_lens:
            for lr in ctx.lr_grid:
                for variant in variants:
                    if (
                        variant.backend in {"window", "sparse"}
                        and variant.window_size is not None
                        and variant.window_size >= seq_len
                    ):
                        continue

                    specs.append(
                        TrialSpec(
                            variant_id=variant.variant_id,
                            variant_cfg_path=str(variant.cfg_path),
                            backend=variant.backend,
                            sparse_tag=variant.sparse_tag,
                            window_size=variant.window_size,
                            expander_degree=variant.expander_degree,
                            seq_len=int(seq_len),
                            seed=int(seed),
                            lr=float(lr),
                        )
                    )

    return specs


def steps_for_token_budget(seq_len: int, batch_size: int, token_budget: int) -> int:
    return max(1, math.ceil(token_budget / (seq_len * batch_size)))


def attention_degree_from_spec(spec: TrialSpec) -> int:
    if spec.backend == "dense":
        return int(spec.seq_len)
    if spec.backend == "window":
        return int(spec.window_size or 0)
    return int((spec.window_size or 0) + (spec.expander_degree or 0))


def _compute_save_every(ctx: ProfileContext, steps: int, run_cfg: dict[str, Any]) -> int:
    save_every = ctx.save_every
    if isinstance(save_every, int):
        return max(1, int(save_every))
    if save_every == "steps" or save_every == "final":
        return steps
    existing = run_cfg.get("save_every", steps)
    return max(1, int(existing))


def trial_to_config(
    spec: TrialSpec,
    out_dir: Path,
    batch_size: int,
    *,
    ctx: ProfileContext,
    variant_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    steps = steps_for_token_budget(spec.seq_len, batch_size, ctx.train_tokens_target)

    base_cfg = variant_cache.get(spec.variant_id)
    if base_cfg is None:
        raise KeyError(f"Missing variant config in cache: {spec.variant_id}")

    cfg = copy.deepcopy(base_cfg)

    run_cfg = cfg.setdefault("run", {})
    run_cfg["out_dir"] = str(out_dir)
    run_cfg["seed"] = int(spec.seed)
    run_cfg["steps"] = int(steps)
    run_cfg["log_every"] = int(ctx.log_every)
    run_cfg["save_every"] = _compute_save_every(ctx, steps, run_cfg)

    data_cfg = cfg.setdefault("data", {})
    data_cfg["dataset"] = str(data_cfg.get("dataset", "tinyshakespeare"))
    data_cfg["root"] = str(data_cfg.get("root", "data"))
    data_cfg["seq_len"] = int(spec.seq_len)
    data_cfg["batch_size"] = int(batch_size)

    optim_cfg = cfg.setdefault("optim", {})
    optim_cfg["lr"] = float(spec.lr)

    if ctx.eval_long_context_batch_size is not None:
        eval_cfg = cfg.setdefault("eval", {})
        eval_cfg["long_context_batch_size"] = int(ctx.eval_long_context_batch_size)

    if ctx.strict_apples_to_apples and ctx.disable_probe_metrics_for_apples_to_apples:
        attention_cfg = cfg.setdefault("attention", {})
        backend = str(attention_cfg.get("backend", "")).lower()
        if backend == "sparse":
            attention_cfg["sparse_probe_every"] = 0
        if backend == "window":
            attention_cfg["window_probe_every"] = 0

    return cfg


def load_metrics_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not metrics_path.exists():
        return rows

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def mean_key(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row]
    return float(np.mean(vals)) if vals else math.nan


def deterministic_sample_batch(
    ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    *,
    generator: torch.Generator,
):
    n = ids.numel()
    max_start = n - (seq_len + 1)
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack([ids[s : s + seq_len] for s in starts], dim=0).to(device)
    y = torch.stack([ids[s + 1 : s + 1 + seq_len] for s in starts], dim=0).to(device)
    return x, y


def load_model_from_cfg_and_ckpt(
    cfg: dict[str, Any],
    ckpt_path: Path,
    device: torch.device,
):
    _, _, tok = load_tiny_shakespeare(cfg["data"].get("root", "data"))
    vocab_size = tok.vocab_size

    att_cfg = cfg.get("attention", {})
    attention_cfg = AttentionConfig(
        backend=str(att_cfg.get("backend", "dense")),
        window_size=int(att_cfg["window_size"]) if "window_size" in att_cfg else None,
        expander_degree=(int(att_cfg["expander_degree"]) if "expander_degree" in att_cfg else None),
        sparse_impl=str(att_cfg.get("sparse_impl", "flex")),
        sparse_block_size=int(att_cfg.get("sparse_block_size", 128)),
        sparse_probe_every=int(att_cfg.get("sparse_probe_every", 0)),
        sparse_probe_tokens=int(att_cfg.get("sparse_probe_tokens", 256)),
        window_probe_every=int(att_cfg.get("window_probe_every", 50)),
        window_probe_tokens=int(att_cfg.get("window_probe_tokens", 256)),
    )

    stability_raw = cfg.get("stability", {}) or {}
    if not isinstance(stability_raw, dict):
        stability_raw = {}
    raw_stability_cfg = StabilityConfig(
        qk_norm=_cfg_bool(stability_raw.get("qk_norm"), False),
        ortho_init=_cfg_bool(stability_raw.get("ortho_init"), False),
        spectral_norm=_cfg_bool(stability_raw.get("spectral_norm"), False),
    )
    effective_stability_cfg = effective_stability_for_backend(
        raw_stability_cfg,
        attention_backend=attention_cfg.backend,
    )

    model = build_model(
        name=cfg["model"]["name"],
        vocab_size=vocab_size,
        d_model=int(cfg["model"]["d_model"]),
        n_layers=int(cfg["model"]["n_layers"]),
        n_heads=int(cfg["model"]["n_heads"]),
        mlp_mult=int(cfg["model"]["mlp_mult"]),
        device=device,
        attention_cfg=attention_cfg,
        stability_cfg=effective_stability_cfg,
    )

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError:
        # Backward compatibility for checkpoints that kept stability wrappers on non-sparse backends.
        if effective_stability_cfg != raw_stability_cfg and any_stability_enabled(
            raw_stability_cfg
        ):
            model = build_model(
                name=cfg["model"]["name"],
                vocab_size=vocab_size,
                d_model=int(cfg["model"]["d_model"]),
                n_layers=int(cfg["model"]["n_layers"]),
                n_heads=int(cfg["model"]["n_heads"]),
                mlp_mult=int(cfg["model"]["mlp_mult"]),
                device=device,
                attention_cfg=attention_cfg,
                stability_cfg=raw_stability_cfg,
            )
            model.load_state_dict(ckpt["model"], strict=True)
        else:
            raise
    model.eval()
    return model, tok


def evaluate_val_ppl(
    cfg: dict[str, Any], ckpt_path: Path, *, batches: int = 40
) -> tuple[float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model_from_cfg_and_ckpt(cfg, ckpt_path, device)

    _, val_ids, _ = load_tiny_shakespeare(cfg["data"].get("root", "data"))
    batch_size = int(cfg["data"]["batch_size"])
    seq_len = int(cfg["data"]["seq_len"])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026)

    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = deterministic_sample_batch(
                val_ids,
                batch_size,
                seq_len,
                device,
                generator=generator,
            )
            logits = model(x)
            loss = loss_fn(logits, y)
            losses.append(float(loss.item()))

    val_loss = float(np.mean(losses))
    val_ppl = float(math.exp(min(val_loss, 100.0)))
    return val_loss, val_ppl


def benchmark_model_only_forward(
    cfg: dict[str, Any],
    ckpt_path: Path,
    *,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model_from_cfg_and_ckpt(cfg, ckpt_path, device)

    batch_size = int(cfg["data"]["batch_size"])
    seq_len = int(cfg["data"]["seq_len"])
    vocab_size = tok.vocab_size

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    for _ in range(warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    tokens = iters * batch_size * seq_len
    return tokens / dt if dt > 0 else math.nan


def summarize_run(
    spec: TrialSpec,
    cfg: dict[str, Any],
    run_dir: Path,
    duration_s: float,
    status: str,
    *,
    ctx: ProfileContext,
    hardware_meta: dict[str, Any],
    error: str = "",
) -> dict[str, Any]:
    out = {
        "trial_id": spec.trial_id,
        "variant_id": spec.variant_id,
        "backend": spec.backend,
        "sparse_tag": spec.sparse_tag,
        "seq_len": spec.seq_len,
        "seed": spec.seed,
        "lr": spec.lr,
        "batch_size": int(cfg["data"]["batch_size"]),
        "steps_target": int(cfg["run"]["steps"]),
        "train_tokens_target": ctx.train_tokens_target,
        "attention_degree_expected": attention_degree_from_spec(spec),
        "duration_s": duration_s,
        "status": status,
        "error": error,
        "run_dir": str(run_dir),
        **hardware_meta,
    }

    metrics_path = run_dir / "metrics.jsonl"
    rows = load_metrics_rows(metrics_path)
    if status != "ok" or not rows:
        return out

    step_rows = [row for row in rows if row.get("type") == "step"]
    window_rows = [row for row in rows if row.get("type") == "window"]
    run_rows = [row for row in rows if row.get("type") == "run"]

    if not step_rows:
        out["status"] = "no_step_rows"
        return out

    final_step = step_rows[-1]
    tail = step_rows[-min(20, len(step_rows)) :]

    out.update(
        {
            "steps_observed": int(final_step.get("step", len(step_rows))),
            "final_loss": float(final_step.get("loss", math.nan)),
            "final_ppl": float(final_step.get("ppl", math.nan)),
            "final_acc_top1": float(final_step.get("accuracy_top1", math.nan)),
            "throughput_tok_s_tail_mean": mean_key(tail, "throughput_tokens_per_sec"),
            "step_time_ms_tail_mean": mean_key(tail, "step_time_ms"),
            "grad_norm_tail_mean": mean_key(tail, "grad_norm"),
        }
    )

    if run_rows:
        run_row = run_rows[-1]
        out.update(
            {
                "qk_norm": bool(run_row.get("qk_norm", False)),
                "ortho_init": bool(run_row.get("ortho_init", False)),
                "spectral_norm": bool(run_row.get("spectral_norm", False)),
            }
        )

    if window_rows:
        window_row = window_rows[-1]
        out.update(
            {
                "vram_peak_mib": float(window_row.get("vram_peak_mib", math.nan)),
                "attn_entropy": float(window_row.get("attention_entropy", math.nan)),
                "attn_entropy_norm": float(
                    window_row.get("attention_entropy_normalized", math.nan)
                ),
                "valid_neighbors": float(window_row.get("valid_neighbor_fraction", math.nan)),
                "window_mass_pct": float(window_row.get("attention_mass_window_pct", math.nan)),
                "expander_mass_pct": float(window_row.get("attention_mass_expander_pct", math.nan)),
                "valid_neighbor_fraction_causal_cap": float(
                    window_row.get("valid_neighbor_fraction_causal_cap", math.nan)
                ),
                "valid_neighbor_fraction_vs_causal_cap": float(
                    window_row.get("valid_neighbor_fraction_vs_causal_cap", math.nan)
                ),
                "future_neighbor_slots": float(window_row.get("future_neighbor_slots", math.nan)),
                "duplicate_neighbor_slots": float(
                    window_row.get("duplicate_neighbor_slots", math.nan)
                ),
            }
        )

    ckpt_path = run_dir / "checkpoint.pt"
    if ctx.run_val_eval and ckpt_path.exists():
        try:
            val_loss, val_ppl = evaluate_val_ppl(cfg, ckpt_path, batches=ctx.val_eval_batches)
            out["val_loss"] = val_loss
            out["val_ppl"] = val_ppl
        except Exception as exc:
            out["val_loss"] = math.nan
            out["val_ppl"] = math.nan
            out["val_eval_error"] = str(exc)

    if ctx.run_model_only_bench and ckpt_path.exists():
        try:
            out["model_only_forward_tok_s"] = benchmark_model_only_forward(
                cfg,
                ckpt_path,
                warmup=ctx.bench_warmup,
                iters=ctx.bench_iters,
            )
        except Exception as exc:
            out["model_only_forward_tok_s"] = math.nan
            out["bench_error"] = str(exc)

    return out


def is_oom_error(text: str) -> bool:
    lowered = (text or "").lower()
    patterns = [
        "cuda out of memory",
        "cuda error: out of memory",
        "torch.cuda.outofmemoryerror",
        "cudnn_status_alloc_failed",
        "cublas_status_alloc_failed",
        "std::bad_alloc",
        "out of memory",
        "resource exhausted",
    ]
    return any(pattern in lowered for pattern in patterns)


def is_retryable_cuda_failure(text: str) -> bool:
    lowered = (text or "").lower()
    patterns = [
        "cuda error",
        "cudnn",
        "cublas",
        "illegal memory access",
        "device-side assert",
        "resource exhausted",
    ]
    return any(pattern in lowered for pattern in patterns)


def _batch_candidates_for_seq(ctx: ProfileContext, seq_len: int) -> list[int]:
    if ctx.strict_apples_to_apples:
        if seq_len not in ctx.fixed_batch_by_seq:
            keys = sorted(ctx.fixed_batch_by_seq.keys())
            raise KeyError(f"Missing fixed_batch_by_seq entry for seq_len={seq_len}; keys={keys}")
        return [int(ctx.fixed_batch_by_seq[seq_len])]

    if seq_len not in ctx.batch_candidates_by_seq:
        keys = sorted(ctx.batch_candidates_by_seq.keys())
        raise KeyError(f"Missing batch_candidates_by_seq entry for seq_len={seq_len}; keys={keys}")

    candidates = [int(v) for v in ctx.batch_candidates_by_seq[seq_len]]
    if 1 not in candidates:
        candidates.append(1)
    return candidates


def run_trial(
    spec: TrialSpec,
    *,
    ctx: ProfileContext,
    runs_root: Path,
    generated_cfg_root: Path,
    variant_cache: dict[str, dict[str, Any]],
    hardware_meta: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    run_dir = runs_root / spec.trial_id
    cfg_dir = generated_cfg_root / spec.trial_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists() and not overwrite:
        cfg_path_existing = cfg_dir / "config.yaml"
        if cfg_path_existing.exists():
            cfg = yaml.safe_load(cfg_path_existing.read_text(encoding="utf-8"))
        else:
            batch_size = _batch_candidates_for_seq(ctx, spec.seq_len)[0]
            cfg = trial_to_config(
                spec,
                run_dir,
                batch_size,
                ctx=ctx,
                variant_cache=variant_cache,
            )
        print(f"[skip] {spec.trial_id}")
        return summarize_run(
            spec,
            cfg,
            run_dir,
            duration_s=0.0,
            status="ok",
            ctx=ctx,
            hardware_meta=hardware_meta,
        )

    try:
        batch_candidates = _batch_candidates_for_seq(ctx, spec.seq_len)
    except KeyError as exc:
        err = str(exc)
        print(f"[fail] {spec.trial_id}: {err}")
        cfg = trial_to_config(spec, run_dir, 1, ctx=ctx, variant_cache=variant_cache)
        return summarize_run(
            spec,
            cfg,
            run_dir,
            duration_s=0.0,
            status="failed",
            error=err,
            ctx=ctx,
            hardware_meta=hardware_meta,
        )

    last_error = ""

    for attempt_idx, batch_size in enumerate(batch_candidates, start=1):
        cfg = trial_to_config(
            spec,
            run_dir,
            batch_size,
            ctx=ctx,
            variant_cache=variant_cache,
        )
        cfg_path = cfg_dir / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        cmd = [sys.executable, "-m", "orion.train", "--config", str(cfg_path)]
        child_env = os.environ.copy()
        child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        t0 = time.time()
        proc = subprocess.run(
            cmd,
            cwd=ctx.repo_root,
            text=True,
            capture_output=True,
            env=child_env,
        )
        dt = time.time() - t0

        (run_dir / f"attempt_{attempt_idx}.stdout.log").write_text(proc.stdout, encoding="utf-8")
        (run_dir / f"attempt_{attempt_idx}.stderr.log").write_text(proc.stderr, encoding="utf-8")

        if proc.returncode == 0:
            print(f"[ok] {spec.trial_id} (bs={batch_size}, {dt:.1f}s)")
            return summarize_run(
                spec,
                cfg,
                run_dir,
                duration_s=dt,
                status="ok",
                ctx=ctx,
                hardware_meta=hardware_meta,
            )

        stderr_tail = ""
        for line in reversed((proc.stderr or "").splitlines()):
            if line.strip():
                stderr_tail = line.strip()
                break

        stdout_tail = ""
        for line in reversed((proc.stdout or "").splitlines()):
            if line.strip():
                stdout_tail = line.strip()
                break

        reason = stderr_tail or stdout_tail
        err_blob = ((proc.stderr or "") + "\n" + (proc.stdout or ""))[-12000:]
        last_error = err_blob

        retryable = False
        if ctx.auto_retry_oom and is_oom_error(err_blob):
            retryable = True
        if (
            not retryable
            and ctx.auto_retry_dense_cuda_fail
            and spec.backend == "dense"
            and is_retryable_cuda_failure(err_blob)
        ):
            retryable = True

        if retryable and (not ctx.strict_apples_to_apples) and attempt_idx < len(batch_candidates):
            suffix = f" | tail: {reason[:180]}" if reason else ""
            print(
                f"[retry-backoff] {spec.trial_id}: bs={batch_size} failed, trying smaller batch"
                f"{suffix}"
            )
            continue

        suffix = f" | tail: {reason[:220]}" if reason else ""
        print(f"[fail] {spec.trial_id} (bs={batch_size}){suffix}")
        return summarize_run(
            spec,
            cfg,
            run_dir,
            duration_s=dt,
            status="failed",
            error=err_blob,
            ctx=ctx,
            hardware_meta=hardware_meta,
        )

    cfg = trial_to_config(
        spec,
        run_dir,
        batch_candidates[0] if batch_candidates else 1,
        ctx=ctx,
        variant_cache=variant_cache,
    )
    return summarize_run(
        spec,
        cfg,
        run_dir,
        duration_s=0.0,
        status="failed",
        error=last_error,
        ctx=ctx,
        hardware_meta=hardware_meta,
    )


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_profile(
    profile: str,
    *,
    repo_root: Path | None = None,
    overwrite: bool | None = None,
) -> RunResult:
    ctx = load_profile_context(profile, repo_root=repo_root)
    variants = load_variant_definitions(ctx)
    trial_specs = build_trial_specs(ctx, variants)

    profile_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", ctx.profile_name)
    experiment_id = time.strftime(f"{profile_slug}_dense_window_sparse_%Y%m%d_%H%M%S")

    runs_root = ctx.repo_root / "runs" / experiment_id
    generated_cfg_root = ctx.repo_root / "configs" / "generated" / experiment_id
    runs_root.mkdir(parents=True, exist_ok=True)
    generated_cfg_root.mkdir(parents=True, exist_ok=True)

    hardware_meta = get_hardware_meta()
    (runs_root / "hardware_meta.json").write_text(
        json.dumps(hardware_meta, indent=2),
        encoding="utf-8",
    )

    variant_cache: dict[str, dict[str, Any]] = {
        variant.variant_id: yaml.safe_load(variant.cfg_path.read_text(encoding="utf-8"))
        for variant in variants
    }

    if overwrite is None:
        overwrite = ctx.overwrite_default

    print("PROFILE:", ctx.profile_name)
    print("PROFILE_CONFIG_PATH:", ctx.profile_path)
    print("STRICT_APPLES_TO_APPLES:", ctx.strict_apples_to_apples)
    print("TRAIN_TOKENS_TARGET:", ctx.train_tokens_target)
    print("VARIANTS_IN_PROFILE:", len(variants))
    print("RUNS_ROOT:", runs_root)
    print("CFG_ROOT:", generated_cfg_root)
    print("PLANNED_TRIALS:", len(trial_specs))

    rows: list[dict[str, Any]] = []
    total = len(trial_specs)
    for idx, spec in enumerate(trial_specs, start=1):
        print(f"\n[{idx}/{total}] {spec.trial_id}")
        row = run_trial(
            spec,
            ctx=ctx,
            runs_root=runs_root,
            generated_cfg_root=generated_cfg_root,
            variant_cache=variant_cache,
            hardware_meta=hardware_meta,
            overwrite=bool(overwrite),
        )
        rows.append(row)

    summary_csv = runs_root / "summary.csv"
    _write_rows_csv(summary_csv, rows)
    print(f"\nSaved: {summary_csv}")

    return RunResult(
        profile=ctx.profile_name,
        experiment_id=experiment_id,
        runs_root=runs_root,
        generated_cfg_root=generated_cfg_root,
        summary_csv=summary_csv,
        trial_count=len(trial_specs),
    )


def load_summary_df(summary_csv: str | Path):
    import pandas as pd

    return pd.read_csv(summary_csv)


def prepare_analysis_columns(df_in):
    out = df_in.copy()

    if "train_tok_per_s" not in out.columns and "throughput_tok_s_tail_mean" in out.columns:
        out["train_tok_per_s"] = out["throughput_tok_s_tail_mean"]

    if "model_tok_per_s" not in out.columns and "model_only_forward_tok_s" in out.columns:
        out["model_tok_per_s"] = out["model_only_forward_tok_s"]

    if "peak_vram_gb" not in out.columns:
        if "vram_peak_mib" in out.columns:
            out["peak_vram_gb"] = out["vram_peak_mib"] / 1024.0
        elif "vram_peak_mb" in out.columns:
            out["peak_vram_gb"] = out["vram_peak_mb"] / 1024.0

    if "val_ppl" not in out.columns and "final_ppl" in out.columns:
        out["val_ppl"] = out["final_ppl"]

    return out


def _empty_paired_df():
    import pandas as pd

    return pd.DataFrame(
        columns=[
            "trial_id",
            "backend",
            "sparse_tag",
            "seq_len",
            "seed",
            "lr",
            "speedup_train_over_dense",
            "speedup_model_over_dense",
            "vram_ratio_over_dense",
            "val_ppl_delta",
        ]
    )


def paired_analysis_tables(df_ok):
    import numpy as np
    import pandas as pd

    dense = df_ok[df_ok["backend"] == "dense"].copy()
    non_dense = df_ok[df_ok["backend"] != "dense"].copy()

    pair_keys = ["seq_len", "seed", "lr"]
    required_cols = pair_keys + [
        "trial_id",
        "train_tok_per_s",
        "model_tok_per_s",
        "val_ppl",
        "peak_vram_gb",
    ]
    missing = [c for c in required_cols if c not in df_ok.columns]

    if missing or dense.empty or non_dense.empty:
        if missing:
            print("Missing required columns for paired analysis:", missing)
        if dense.empty or non_dense.empty:
            print("No dense or non-dense rows available for pairing.")
        empty = _empty_paired_df()
        return empty, empty, pd.DataFrame()

    dense_small = dense[required_cols].rename(
        columns={
            "trial_id": "dense_trial_id",
            "train_tok_per_s": "train_tok_per_s_dense",
            "model_tok_per_s": "model_tok_per_s_dense",
            "val_ppl": "val_ppl_dense",
            "peak_vram_gb": "peak_vram_gb_dense",
        }
    )

    paired_all = non_dense.merge(dense_small, on=pair_keys, how="inner")
    paired_all["speedup_train_over_dense"] = paired_all["train_tok_per_s"] / paired_all[
        "train_tok_per_s_dense"
    ].replace(0, np.nan)
    paired_all["speedup_model_over_dense"] = paired_all["model_tok_per_s"] / paired_all[
        "model_tok_per_s_dense"
    ].replace(0, np.nan)
    paired_all["vram_ratio_over_dense"] = paired_all["peak_vram_gb"] / paired_all[
        "peak_vram_gb_dense"
    ].replace(0, np.nan)
    paired_all["val_ppl_delta"] = paired_all["val_ppl"] - paired_all["val_ppl_dense"]

    window_rows = non_dense[non_dense["backend"] == "window"].copy()
    sparse_rows = non_dense[non_dense["backend"] == "sparse"].copy()

    sparse_best = sparse_rows.copy()
    if len(sparse_rows) > 0:
        score_col = "val_ppl" if sparse_rows["val_ppl"].notna().any() else "final_ppl"
        sparse_best = (
            sparse_rows.sort_values(
                by=pair_keys + [score_col, "model_tok_per_s"],
                ascending=[True, True, True, True, False],
                na_position="last",
            )
            .drop_duplicates(subset=pair_keys, keep="first")
            .copy()
        )

    focused = pd.concat([window_rows, sparse_best], ignore_index=True)

    paired = focused.merge(dense_small, on=pair_keys, how="inner")
    paired["speedup_train_over_dense"] = paired["train_tok_per_s"] / paired[
        "train_tok_per_s_dense"
    ].replace(0, np.nan)
    paired["speedup_model_over_dense"] = paired["model_tok_per_s"] / paired[
        "model_tok_per_s_dense"
    ].replace(0, np.nan)
    paired["vram_ratio_over_dense"] = paired["peak_vram_gb"] / paired["peak_vram_gb_dense"].replace(
        0, np.nan
    )
    paired["val_ppl_delta"] = paired["val_ppl"] - paired["val_ppl_dense"]

    agg = paired.groupby(["backend", "seq_len", "sparse_tag", "lr"], as_index=False).agg(
        runs=("trial_id", "count"),
        train_speedup_mean=("speedup_train_over_dense", "mean"),
        train_speedup_std=("speedup_train_over_dense", "std"),
        model_speedup_mean=("speedup_model_over_dense", "mean"),
        vram_ratio_mean=("vram_ratio_over_dense", "mean"),
        val_ppl_delta_mean=("val_ppl_delta", "mean"),
        val_ppl_delta_std=("val_ppl_delta", "std"),
    )

    return paired_all, paired, agg


def select_winners(agg, *, val_ppl_tolerance: float = 0.20):
    if len(agg) == 0:
        return agg.copy()

    return agg[
        (agg["model_speedup_mean"] > 1.0)
        & (agg["vram_ratio_mean"] < 1.0)
        & (agg["val_ppl_delta_mean"] <= float(val_ppl_tolerance))
    ].copy()


def plot_speedup_ratios(paired):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(paired) == 0:
        print("No paired rows; skipping speedup plots.")
        return

    plot_df = paired.copy()
    plot_df["variant"] = (
        plot_df["backend"] + "_" + plot_df["sparse_tag"] + "_lr" + plot_df["lr"].astype(str)
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=plot_df,
        x="seq_len",
        y="speedup_train_over_dense",
        hue="variant",
        marker="o",
        estimator="mean",
        errorbar="sd",
    )
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title("Train Throughput Ratio vs Dense")
    plt.ylabel("ratio (>1 means faster than dense)")
    plt.xlabel("seq_len")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=plot_df,
        x="seq_len",
        y="speedup_model_over_dense",
        hue="variant",
        marker="o",
        estimator="mean",
        errorbar="sd",
    )
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title("Model-Only Throughput Ratio vs Dense")
    plt.ylabel("ratio (>1 means faster than dense)")
    plt.xlabel("seq_len")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_vram_and_quality(paired):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(paired) == 0:
        print("No paired rows; skipping VRAM/quality plots.")
        return

    plot_df = paired.copy()
    plot_df["variant"] = (
        plot_df["backend"] + "_" + plot_df["sparse_tag"] + "_lr" + plot_df["lr"].astype(str)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(
        data=plot_df,
        x="seq_len",
        y="vram_ratio_over_dense",
        hue="variant",
        marker="o",
        estimator="mean",
        errorbar="sd",
        ax=axes[0],
    )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("VRAM Ratio vs Dense")
    axes[0].set_ylabel("ratio (<1 means less VRAM than dense)")
    axes[0].set_xlabel("seq_len")

    sns.lineplot(
        data=plot_df,
        x="seq_len",
        y="val_ppl_delta",
        hue="variant",
        marker="o",
        estimator="mean",
        errorbar="sd",
        ax=axes[1],
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Validation PPL Delta vs Dense")
    axes[1].set_ylabel("delta (<0 means better than dense)")
    axes[1].set_xlabel("seq_len")

    handles, labels = axes[1].get_legend_handles_labels()
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    fig.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_all_numeric_metrics(
    df_ok,
    *,
    out_dir: str | Path,
    show_inline: bool = True,
    max_inline: int | None = None,
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df_plot = df_ok.copy()

    def variant_label(row):
        lr = row.get("lr", np.nan)
        backend = row.get("backend", "unknown")
        if backend == "dense":
            return f"dense_lr{lr}"
        return f"{backend}_{row.get('sparse_tag', 'na')}_lr{lr}"

    df_plot["variant"] = df_plot.apply(variant_label, axis=1)

    control_numeric_cols = {
        "seq_len",
        "seed",
        "lr",
        "batch_size",
        "steps_target",
        "steps_observed",
        "train_tokens_target",
        "attention_degree_expected",
        "gpu_count",
        "cuda_available",
    }

    numeric_cols = [c for c in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[c])]

    metric_cols = []
    for col in numeric_cols:
        if col in control_numeric_cols:
            continue
        s = df_plot[col]
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        metric_cols.append(col)

    metric_cols = sorted(metric_cols)

    plots_dir = Path(out_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered numeric metrics: {len(metric_cols)}")
    rendered = 0

    for idx, metric in enumerate(metric_cols, start=1):
        sub = df_plot[["seq_len", "variant", metric]].dropna()
        if sub.empty:
            continue

        plt.figure(figsize=(10, 5))
        if sub["seq_len"].nunique() > 1:
            sns.lineplot(
                data=sub,
                x="seq_len",
                y=metric,
                hue="variant",
                marker="o",
                estimator="mean",
                errorbar="sd",
            )
            plt.xlabel("seq_len")
        else:
            sns.boxplot(data=sub, x="variant", y=metric)
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("variant")

        plt.title(f"{metric} (all variants)")
        plt.ylabel(metric)
        plt.tight_layout()

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric)
        out_file = plots_dir / f"{idx:03d}_{safe_name}.png"
        plt.savefig(out_file, dpi=170)

        rendered += 1
        if show_inline and (max_inline is None or rendered <= max_inline):
            plt.show()
        else:
            plt.close()

    print(f"Rendered and saved {rendered} metric plots to: {plots_dir}")
    return metric_cols


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run config-driven experiment profile")
    p.add_argument(
        "--profile", required=True, help="Profile name under configs/experiments/profiles"
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing run dirs")
    args = p.parse_args()

    run_result = run_profile(args.profile, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "profile": run_result.profile,
                "experiment_id": run_result.experiment_id,
                "runs_root": str(run_result.runs_root),
                "summary_csv": str(run_result.summary_csv),
                "trial_count": run_result.trial_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
