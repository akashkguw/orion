"""Ablation runner for stability mechanisms over sparse attention configs.

Usage:
    python -m orion.ablation --config configs/ablation_sparse_a.yaml
    python -m orion.ablation --config configs/ablation_sparse_b.yaml
"""

from __future__ import annotations

import argparse
import json
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


def run_ablation(base_config_path: str) -> None:
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

    # Print comparison table
    print(f"\n{'=' * 60}")
    print(f"ABLATION SUMMARY: {base_name}")
    print(f"{'=' * 60}")
    header = f"{'Combo':<18} {'Loss':>8} {'PPL':>8} {'Div%':>6} {'ScoreMean':>10} {'tok/s':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['combo']:<18} "
            f"{r.get('final_loss', float('nan')):>8.4f} "
            f"{r.get('final_ppl', float('nan')):>8.2f} "
            f"{r.get('divergence_rate', 0.0) * 100:>6.1f} "
            f"{r.get('attn_score_mean', 0.0):>10.4f} "
            f"{r.get('avg_throughput', 0.0):>8.1f}"
        )

    # Save JSON summary
    summary_path = summary_dir / f"ablation_summary_{base_name}.json"
    with open(summary_path, "w") as f:
        json.dump({"base_config": base_config_path, "results": results}, f, indent=2)
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
    combined[base_name] = {"base_config": base_config_path, "results": results}
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined summary updated at {combined_path}")


def main():
    p = argparse.ArgumentParser(description="Run stability ablation matrix over sparse configs.")
    p.add_argument("--config", required=True, help="Base ablation config YAML path")
    args = p.parse_args()
    run_ablation(args.config)


if __name__ == "__main__":
    main()
