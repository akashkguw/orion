# Orion

Orion is a research framework for decoder-only Transformers with three attention regimes:

- `dense` (full causal attention)
- `window` (causal sliding window)
- `sparse` (window + expander graph), with fused `flex_attention` support

It is built for practical long-context experimentation: reproducible configs, robust metrics, ablations, benchmark regression checks, and notebook-driven experiment orchestration.

## Why Orion

Orion is designed for researchers who want to answer questions like:

- Where does sparse attention beat dense in throughput and memory?
- How does window-only compare with structured sparse at matched token budgets?
- Do sparse stability controls (QK-norm, ortho init, spectral norm) improve quality/robustness?

The repo gives you a full pipeline to run those studies end-to-end.

## Core Capabilities

- Multi-backend attention: `dense`, `window`, `sparse`
- Sparse graph attention with deterministic causal index construction
- Fused sparse path via `torch.nn.attention.flex_attention` (CUDA), plus diagnostics
- Stability controls for sparse backends:
  - `qk_norm`
  - `ortho_init`
  - `spectral_norm`
- Structured metrics logging (`metrics.jsonl`) with run/step/window/eval event types
- Long-context evaluation (`512/1024/2048/4096`) with OOM backoff
- Config-first research workflow (`configs/experiments/profiles` + `variants`)
- CI benchmark regression artifact generation (JSON + Markdown)

## Installation

### Requirements

- Python `3.11+`
- PyTorch `2.x`
- CUDA is optional (required for fused sparse `flex` execution)

### Setup

```bash
git clone https://github.com/akashkguw/orion.git
cd orion
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

### Verify

```bash
python -m pytest -q
```

## 60-Second Quick Start

### Smoke train

```bash
SMOKE_TEST=true SMOKE_STEPS=20 python -m orion.train --config configs/golden.yaml
```

### Evaluate

```bash
python -m orion.eval --config configs/golden.yaml --checkpoint runs/latest/checkpoint.pt
```

### Inspect metrics

```bash
tail -n 10 runs/latest/metrics.jsonl
```

## Attention Backends

| Backend | Mechanism | Typical Complexity | Notes |
|---|---|---:|---|
| `dense` | Full causal attention | `O(T^2)` | Strong short-context baseline; can become memory-heavy at long context |
| `window` | Causal sliding window | `O(T * W)` | Local-context-only approximation; fast and simple |
| `sparse` | Window + expander edges | `O(T * (W + d))` | Captures local + structured long-range context |

### Sparse execution modes (`attention.sparse_impl`)

- `flex`: require fused sparse path
- `auto`: try fused sparse, fallback to reference path when unavailable
- `gather`: force reference gather/scatter path

For research comparisons in this repo, sparse experiment configs are set to `flex`.

### Important runtime constraint for `flex`

Current fused sparse path requires:

- CUDA device
- `torch.nn.attention.flex_attention` availability
- intrinsic sparse causal masking path (no external `attn_mask`)

If these conditions are not met and `sparse_impl: flex` is requested, training raises by design.

## Stability Controls (Sparse-Only)

Stability toggles are applied only when `attention.backend: sparse`.

```yaml
stability:
  qk_norm: true
  ortho_init: false
  spectral_norm: false
```

- `qk_norm`: RMSNorm on per-head Q/K before dot product
- `ortho_init`: orthogonal init for attention projections
- `spectral_norm`: spectral norm wrapper on Q/K projections

The run log records which controls were effectively active.

## Metrics and Telemetry

Orion writes JSONL metrics to `{run_dir}/metrics.jsonl`.

### Event types

- `type: run` (once)
- `type: step` (every step)
- `type: window` (every 50 steps)
- `type: eval` (every 1000 steps)

### Key fields

- Step-level:
  - `loss`, `ppl`, `throughput_tokens_per_sec`, `grad_norm`, `step_time_ms`, `accuracy_top1`, `learning_rate`
- Window-level:
  - `vram_peak_mib`, `divergence_rate`, `activation_norm_rms`, `attention_entropy`, `attention_entropy_normalized`, `attn_score_mean`, `clip_rate`, `spike_rate`
- Sparse-only diagnostics (when sparse backend is active):
  - `valid_neighbor_fraction`
  - `attention_mass_window_pct`, `attention_mass_expander_pct`
  - `future_neighbor_slots`, `duplicate_neighbor_slots`
  - `valid_neighbor_fraction_causal_cap`, `valid_neighbor_fraction_vs_causal_cap`

Unavailable metrics are logged/printed as `NA` rather than misleading zeros.

## Config-First Experiment Framework

The experiment workflow is config-driven, with execution handled by `orion.experiments`.

### Experiment structure

- Profiles: `configs/experiments/profiles/*.yaml`
- Variant arms: `configs/experiments/variants/*.yaml`

### Supported profile presets

- `pilot9`: fast 9-run dense/window/sparse sanity
- `pilot`: broader sweep
- `full`: longer-context sweep
- `pilot_norm`: sparse+norm vs window vs dense

### Run in notebook

Open [`experiment.ipynb`](./experiment.ipynb), then set:

```python
PROFILE = "pilot9"  # or pilot, full, pilot_norm
```

The notebook is intentionally thin:
- it selects a profile
- calls `orion.experiments.run_profile(PROFILE)`
- reads `summary.csv`
- runs paired analysis/plots

All sweep knobs live in profile YAML (`runner.*`, `analysis.*`, `variants[]`).

### Run from CLI (no notebook)

```bash
python -m orion.experiments --profile pilot9
```

This writes:
- `runs/<experiment_id>/summary.csv`
- `runs/<experiment_id>/hardware_meta.json`
- `configs/generated/<experiment_id>/<trial_id>/config.yaml`

## Training, Evaluation, and Run Orchestration

### Direct training

```bash
python -m orion.train --config configs/tinyshakespeare_dense.yaml
python -m orion.train --config configs/tinyshakespeare_window.yaml
python -m orion.train --config configs/tinyshakespeare_sparse.yaml
```

### Resume

```bash
python -m orion.train --config configs/tinyshakespeare_sparse.yaml --resume
```

### Environment-aware orchestrator (`orion.run`)

```bash
python -m orion.run --config configs/tinyshakespeare_sparse.yaml --mode both
```

`orion.run` writes:

- resolved config snapshot
- run metadata (`meta.json`)
- optional eval output (`eval.json`)

and supports explicit `--base-dir`, `--run-id`, and `--steps` overrides.

## Ablation Matrix (Sparse Stability)

Run full 8-combo ablations:

```bash
python -m orion.ablation --config configs/ablation_sparse_a.yaml
python -m orion.ablation --config configs/ablation_sparse_b.yaml
```

Or run both and summarize winners:

```bash
python -m orion.ablation --top2
```

Outputs include:

- `runs/ablation_summary_<config>.json`
- `runs/ablation_summary.json`
- `runs/ablation_top2_summary.json`

## CI Benchmark Regression

Run minimal regression benchmark:

```bash
python -m orion.benchmark_regression \
  --out-dir runs/ci_benchmark \
  --device cpu \
  --steps 50 \
  --seq-len 128 \
  --batch-size 4
```

Outputs:

- `runs/ci_benchmark/benchmark_summary.json`
- `runs/ci_benchmark/benchmark_summary.md`

Note: on CPU, sparse with `sparse_impl=flex` is expected to be marked `skipped` in the benchmark summary.

## Colab Workflow

- Notebook: [`orion.ipynb`](./orion.ipynb)
- Experiment harness: [`experiment.ipynb`](./experiment.ipynb)

Recommended for reproducibility:

1. Mount Drive
2. Keep configs in-repo (`configs/experiments/...`)
3. Save outputs to run-specific directories
4. Use fixed token budgets for apples-to-apples backend comparisons

## Developer Workflow

```bash
make dev
make lint
make format-check
make test
make smoke
```

Current CI jobs (`.github/workflows/ci.yml`):

1. Lint + format checks
2. Unit tests
3. Smoke training
4. Benchmark regression artifact upload

## Project Layout

```text
orion/
  attention/
    base.py
    dense.py
    window.py
    sparse.py
  data/
    shakespeare.py
  models/
    blocks.py
  model.py
  models_factory.py
  train.py
  eval.py
  ablation.py
  benchmark_regression.py
  metrics.py
  stability.py
  run.py

configs/
  golden.yaml
  tinyshakespeare_*.yaml
  ablation_sparse_*.yaml
  experiments/
    profiles/
    variants/
```

## Notes and Caveats

- Fused sparse (`flex`) is CUDA-dependent.
- Dense long-context eval may OOM at large context + batch; evaluator includes batch backoff.
- `window` and fused `sparse` attention entropy/mass may use probe-based estimation depending on config.
- Local visualization scripts under `orion/local_attention_viz/` are intended for local runs and are git-ignored as artifacts.

## License

Apache-2.0

## Contact

For research collaboration, questions, or feedback: **akashkg@uw.edu**
