from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import OrionConfig
from .train import train


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    backend: str
    window_size: int | None = None
    expander_degree: int | None = None
    sparse_impl: str | None = None


BENCHMARK_SPECS = [
    BenchmarkSpec(name="dense", backend="dense"),
    BenchmarkSpec(name="window_w8", backend="window", window_size=8),
    BenchmarkSpec(
        name="sparse_w8_d8",
        backend="sparse",
        window_size=8,
        expander_degree=8,
        sparse_impl="gather",
    ),
]


def _metric_or_nan(row: dict, key: str) -> float:
    value = row.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_run_metrics(metrics_path: Path) -> dict[str, float]:
    step_rows: list[dict] = []
    window_rows: list[dict] = []
    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            t = row.get("type")
            if t == "step":
                step_rows.append(row)
            elif t == "window":
                window_rows.append(row)

    last5 = step_rows[-5:] if step_rows else []
    throughput = (
        sum(_metric_or_nan(r, "throughput_tokens_per_sec") for r in last5) / len(last5)
        if last5
        else float("nan")
    )
    last_window = window_rows[-1] if window_rows else {}
    return {
        "throughput_tokens_per_sec": throughput,
        "vram_peak_mib": _metric_or_nan(last_window, "vram_peak_mib"),
        "attention_entropy_normalized": _metric_or_nan(last_window, "attention_entropy_normalized"),
        "attn_score_mean": _metric_or_nan(last_window, "attn_score_mean"),
    }


def _fmt(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _sanitize_for_json(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    return value


def _build_cfg(
    *,
    out_dir: Path,
    spec: BenchmarkSpec,
    steps: int,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> OrionConfig:
    attention: dict[str, object] = {"backend": spec.backend}
    if spec.window_size is not None:
        attention["window_size"] = spec.window_size
    if spec.expander_degree is not None:
        attention["expander_degree"] = spec.expander_degree
    if spec.sparse_impl is not None:
        attention["sparse_impl"] = spec.sparse_impl

    raw = {
        "run": {
            "out_dir": str(out_dir),
            "seed": seed,
            "steps": steps,
            "log_every": steps,
            "save_every": steps,
        },
        "data": {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "vocab_size": 256,
        },
        "model": {
            "name": "orion",
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "mlp_mult": 4,
        },
        "attention": attention,
        "optim": {"lr": 3e-4},
    }
    return OrionConfig(raw)


def run_benchmarks(
    *,
    out_dir: Path,
    device: torch.device,
    steps: int,
    seq_len: int,
    batch_size: int,
    seed: int,
    clean: bool,
) -> dict:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for spec in BENCHMARK_SPECS:
        run_dir = out_dir / spec.name
        cfg = _build_cfg(
            out_dir=run_dir,
            spec=spec,
            steps=steps,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=seed,
        )
        started = time.time()
        train(cfg, device=device)
        elapsed = time.time() - started
        summary = _read_run_metrics(run_dir / "metrics.jsonl")
        summary.update(
            {
                "name": spec.name,
                "backend": spec.backend,
                "window_size": spec.window_size,
                "expander_degree": spec.expander_degree,
                "elapsed_sec": elapsed,
                "run_dir": str(run_dir),
            }
        )
        rows.append(summary)

    payload = {
        "device": str(device),
        "steps": steps,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "seed": seed,
        "results": rows,
    }
    payload_json = _sanitize_for_json(payload)
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(payload_json, indent=2), encoding="utf-8"
    )

    lines = [
        "# CI Benchmark Summary",
        "",
        f"- device: `{device}`",
        f"- steps: `{steps}`",
        f"- seq_len: `{seq_len}`",
        f"- batch_size: `{batch_size}`",
        "",
        "| run | backend | throughput tok/s | vram MiB | attn_ent_norm | attn_score_mean | elapsed s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['name']} | {row['backend']} | "
            f"{_fmt(row['throughput_tokens_per_sec'], 1)} | "
            f"{_fmt(row['vram_peak_mib'], 1)} | "
            f"{_fmt(row['attention_entropy_normalized'], 4)} | "
            f"{_fmt(row['attn_score_mean'], 4)} | "
            f"{_fmt(row['elapsed_sec'], 1)} |"
        )
    (out_dir / "benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload_json


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Run minimal dense/window/sparse benchmarks and emit VRAM+throughput summary for CI."
        )
    )
    p.add_argument("--out-dir", default="runs/ci_benchmark")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--no-clean", action="store_true")
    args = p.parse_args()

    payload = run_benchmarks(
        out_dir=Path(args.out_dir),
        device=torch.device(args.device),
        steps=max(1, int(args.steps)),
        seq_len=max(8, int(args.seq_len)),
        batch_size=max(1, int(args.batch_size)),
        seed=int(args.seed),
        clean=not args.no_clean,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
