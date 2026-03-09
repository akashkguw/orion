"""Ablation runner for stability mechanisms over sparse attention configs.

Usage:
    python -m orion.ablation --config configs/ablation_sparse_a.yaml
    python -m orion.ablation --config configs/ablation_sparse_b.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from orion.config import OrionConfig
from orion.train import train

# 8 stability combinations: (qk_norm, ortho_init, spectral_norm)
COMBOS = [
    ("baseline", False, False, False),
    ("qk_norm", True, False, False),
    ("ortho", False, True, False),
    ("spectral", False, False, True),
    ("qk+ortho", True, True, False),
    ("qk+spectral", True, False, True),
    ("ortho+spectral", False, True, True),
    ("all", True, True, True),
]


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_finite(value: object) -> bool:
    return math.isfinite(_to_float(value))


def _read_metrics(metrics_path: Path) -> dict:
    """Parse metrics.jsonl and return summary dict."""
    step_entries = []
    window_entries = []
    run_entry = None

    with open(metrics_path) as f:
        for line in f:
            row = json.loads(line.strip())
            t = row.get("type")
            if t == "step":
                step_entries.append(row)
            elif t == "window":
                window_entries.append(row)
            elif t == "run":
                run_entry = row

    result = {}

    if step_entries:
        last = step_entries[-1]
        result["final_loss"] = last.get("loss", float("nan"))
        result["final_ppl"] = last.get("ppl", float("nan"))
        # Avg throughput from last 5 step entries
        last5 = step_entries[-5:]
        toks = [e.get("throughput_tokens_per_sec", 0.0) for e in last5]
        result["avg_throughput"] = sum(toks) / len(toks) if toks else 0.0

    if window_entries:
        last_w = window_entries[-1]
        result["divergence_rate"] = last_w.get("divergence_rate", 0.0)
        result["attn_score_mean"] = last_w.get("attn_score_mean", 0.0)
        result["activation_norm_rms"] = last_w.get("activation_norm_rms", 0.0)

    if run_entry:
        result["qk_norm"] = run_entry.get("qk_norm", False)
        result["ortho_init"] = run_entry.get("ortho_init", False)
        result["spectral_norm"] = run_entry.get("spectral_norm", False)

    return result


def _annotate_overhead_vs_baseline(results: list[dict]) -> float:
    baseline = next((r for r in results if r.get("combo") == "baseline"), None)
    baseline_thr = _to_float((baseline or {}).get("avg_throughput"))

    for r in results:
        thr = _to_float(r.get("avg_throughput"))
        if _is_finite(baseline_thr) and baseline_thr > 0 and _is_finite(thr):
            overhead = 100.0 * (baseline_thr - thr) / baseline_thr
        else:
            overhead = float("nan")
        r["throughput_overhead_pct_vs_baseline"] = overhead
        r["spectral_overhead_pct_vs_baseline"] = (
            overhead if bool(r.get("spectral_norm", False)) else float("nan")
        )

    return baseline_thr


def _select_winners(results: list[dict]) -> dict:
    stable = [
        r
        for r in results
        if r.get("status") == "ok"
        and _is_finite(r.get("final_ppl"))
        and _to_float(r.get("divergence_rate")) <= 0.0
    ]
    if not stable:
        return {"quality_winner": None, "efficiency_winner": None}

    quality = min(stable, key=lambda r: _to_float(r.get("final_ppl")))
    best_ppl = _to_float(quality.get("final_ppl"))
    near_best = [r for r in stable if _to_float(r.get("final_ppl")) <= best_ppl * 1.03] or stable
    efficiency = max(near_best, key=lambda r: _to_float(r.get("avg_throughput")))

    return {
        "quality_winner": {
            "combo": quality.get("combo"),
            "final_ppl": _to_float(quality.get("final_ppl")),
            "avg_throughput": _to_float(quality.get("avg_throughput")),
            "divergence_rate": _to_float(quality.get("divergence_rate")),
        },
        "efficiency_winner": {
            "combo": efficiency.get("combo"),
            "final_ppl": _to_float(efficiency.get("final_ppl")),
            "avg_throughput": _to_float(efficiency.get("avg_throughput")),
            "divergence_rate": _to_float(efficiency.get("divergence_rate")),
        },
    }


def run_ablation(base_config_path: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(base_config_path) as f:
        base_raw = yaml.safe_load(f)

    base_name = Path(base_config_path).stem  # e.g. "ablation_sparse_a"
    summary_dir = Path("runs")
    summary_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for combo_name, qk_norm, ortho_init, spectral_norm in COMBOS:
        run_name = f"{base_name}_{combo_name}"
        out_dir = f"runs/{run_name}"
        print(f"\n{'=' * 60}")
        print(f"Running: {run_name}")
        print(f"  qk_norm={qk_norm}, ortho_init={ortho_init}, spectral_norm={spectral_norm}")
        print(f"{'=' * 60}")

        raw = deepcopy(base_raw)
        raw["run"]["out_dir"] = out_dir
        raw["stability"] = {
            "qk_norm": qk_norm,
            "ortho_init": ortho_init,
            "spectral_norm": spectral_norm,
        }

        cfg = OrionConfig(raw)
        try:
            train(cfg, device=device)
            metrics_path = Path(out_dir) / "metrics.jsonl"
            summary = _read_metrics(metrics_path)
            summary["combo"] = combo_name
            summary["out_dir"] = out_dir
            summary["status"] = "ok"
        except Exception as e:
            print(f"  ERROR: {e}")
            summary = {
                "combo": combo_name,
                "out_dir": out_dir,
                "status": f"error: {e}",
                "final_loss": float("nan"),
                "final_ppl": float("nan"),
            }

        results.append(summary)

    baseline_thr = _annotate_overhead_vs_baseline(results)
    winners = _select_winners(results)

    # Print comparison table
    print(f"\n{'=' * 60}")
    print(f"ABLATION SUMMARY: {base_name}")
    print(f"{'=' * 60}")
    header = (
        f"{'Combo':<18} {'Loss':>8} {'PPL':>8} {'Div%':>6} "
        f"{'ScoreMean':>10} {'tok/s':>8} {'Overhd%':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        overhead = r.get("throughput_overhead_pct_vs_baseline", float("nan"))
        overhead_str = f"{overhead:>8.2f}" if _is_finite(overhead) else f"{'NA':>8}"
        print(
            f"{r['combo']:<18} "
            f"{r.get('final_loss', float('nan')):>8.4f} "
            f"{r.get('final_ppl', float('nan')):>8.2f} "
            f"{r.get('divergence_rate', 0.0) * 100:>6.1f} "
            f"{r.get('attn_score_mean', 0.0):>10.4f} "
            f"{r.get('avg_throughput', 0.0):>8.1f} "
            f"{overhead_str}"
        )
    if winners.get("quality_winner"):
        qw = winners["quality_winner"]
        ew = winners["efficiency_winner"]
        print(
            "\nWinners:"
            f" quality={qw['combo']} (ppl={qw['final_ppl']:.2f}, tok/s={qw['avg_throughput']:.1f}),"
            f" efficiency={ew['combo']} (ppl={ew['final_ppl']:.2f}, tok/s={ew['avg_throughput']:.1f})"
        )

    # Save JSON summary
    summary_payload = {
        "base_config": base_config_path,
        "baseline_throughput": baseline_thr,
        "winners": winners,
        "results": results,
    }
    summary_path = summary_dir / f"ablation_summary_{base_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Also write/update the combined summary file
    combined_path = summary_dir / "ablation_summary.json"
    combined = {}
    if combined_path.exists():
        with open(combined_path) as f:
            try:
                combined = json.load(f)
            except json.JSONDecodeError:
                combined = {}
    combined[base_name] = summary_payload
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined summary updated at {combined_path}")
    return summary_payload


def run_top2_ablation(
    config_paths: list[str] | None = None,
) -> dict:
    """Run ablations for top-2 sparse configs and produce a combined summary."""
    if config_paths is None:
        config_paths = [
            "configs/ablation_sparse_a.yaml",
            "configs/ablation_sparse_b.yaml",
        ]
    summaries = [run_ablation(path) for path in config_paths]

    all_stable: list[dict] = []
    for summary in summaries:
        base = Path(summary["base_config"]).stem
        for result in summary.get("results", []):
            if result.get("status") == "ok" and _to_float(result.get("divergence_rate")) <= 0.0:
                all_stable.append(
                    {
                        "base": base,
                        "combo": result.get("combo"),
                        "final_ppl": _to_float(result.get("final_ppl")),
                        "avg_throughput": _to_float(result.get("avg_throughput")),
                    }
                )
    global_quality = min(all_stable, key=lambda r: r["final_ppl"]) if all_stable else None
    global_efficiency = max(all_stable, key=lambda r: r["avg_throughput"]) if all_stable else None
    payload = {
        "configs": config_paths,
        "summaries": summaries,
        "global_winners": {
            "quality": global_quality,
            "efficiency": global_efficiency,
        },
    }
    out_path = Path("runs") / "ablation_top2_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nTop-2 summary saved to {out_path}")
    return payload


def main():
    p = argparse.ArgumentParser(description="Run stability ablation matrix over sparse configs.")
    p.add_argument("--config", required=False, help="Base ablation config YAML path")
    p.add_argument(
        "--top2",
        action="store_true",
        help="Run ablations for both top sparse configs (A and B) and summarize winners.",
    )
    args = p.parse_args()
    if args.top2:
        run_top2_ablation()
        return
    if not args.config:
        raise SystemExit("Either --config or --top2 is required.")
    run_ablation(args.config)


if __name__ == "__main__":
    main()
