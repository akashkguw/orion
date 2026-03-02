# Orion

A research framework for long-context, decoder-only Transformers with **structured sparse attention** (sliding window + expander edges).

Orion combines efficient sparse attention patterns with comprehensive metrics tracking to enable research on long-context language models. It provides multiple attention backends (dense, sparse, window), real-time metrics for model health monitoring, and reproducible training with deterministic checkpointing.

**Key Features:**
- **Sparse Attention** - O(T·(W+d)) vs O(T²) dense (7x faster on 512 tokens)
- **Multiple Backends** - Dense, sparse, and window attention
- **Real Metrics** - Activation norm, attention entropy, long-context eval
- **Reproducible** - Deterministic training with seed control
- **Well-tested** - 136 tests covering all components
- **Production-ready** - Configs for 256-4K context lengths

**Next Steps:**
- Norm control (QK-norm, orthogonal init, spectral normalization)
- Stability improvements for long-context training

**Quick Links:** [Installation](#installation) | [Quick Start](#quick-start) | [Usage](#usage) | [Development](#development)

---

## Quick Start

**Local Setup (5 minutes):**
```bash
git clone https://github.com/akashkguw/orion.git && cd orion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m orion.train --config configs/golden.yaml
```

**Google Colab (No Setup):**
Click to open: [Orion-Master.ipynb](https://colab.research.google.com/github/akashkguw/orion/blob/main/orion.ipynb)

---

## Installation

**Requirements:** Python 3.10+, PyTorch 2.0+, GPU optional (CUDA 11.8+)

**Setup:**
```bash
make dev                    # Development setup (recommended)
# OR
pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
```

**Verify:**
```bash
python -c "import orion; print('✓ Orion installed')"
make test                   # Run quick test
```

---

## Sparse Attention

Sparse attention reduces complexity from **O(T²)** to **O(T·(W+d))** by attending to:
- **Local Window** - Dense context for nearby tokens
- **Expander Edges** - Structured long-range connections

**Example (T=512, W=64, d=8):**
```
Query at position 100 attends to:
  Window:   [37-100]           (64 positions)
  Expander: [99, 96, 89, ...]  (8 positions)
  Total:    72 positions vs 512 for dense (7.1x speedup)
```

**Configuration:**
```yaml
model:
  attention_type: sparse
  window_size: 64        # Local window
  expander_degree: 8     # Long-range neighbors
```

**When to Use:**
| Scenario | Recommendation |
|----------|---|
| Long sequences (512+) | Use sparse |
| Limited GPU memory | Use sparse |
| Short sequences (<256) | Dense is simpler |

For details, see [SPARSE_ATTENTION_ARCHITECTURE.md](SPARSE_ATTENTION_ARCHITECTURE.md)

## Norm Control (Next Step)

Planned stability improvements for long-context training:
- **QK-Norm** - Query-Key normalization for attention stability
- **Orthogonal Init** - Orthogonal weight initialization
- **Spectral Normalization** - Spectral norm regularization

These techniques help prevent gradient explosion and improve training stability for very long sequences (4K+ tokens).

---

## Usage

**Training:**
```bash
python -m orion.train --config configs/golden.yaml
python -m orion.train --config configs/golden.yaml --resume  # Resume from latest
```

**Evaluation:**
```bash
python -m orion.eval --config configs/golden.yaml --checkpoint runs/latest/checkpoint.pt
```

**View Metrics:**
```bash
tail -n 5 runs/latest/metrics.jsonl                    # Last 5 steps
cat runs/latest/metrics.jsonl | jq .                   # Pretty print
cat runs/latest/metrics.jsonl | jq 'select(.type == "step") | .loss'  # Extract field
```

**Available Configs:**
- `golden.yaml` - Dense attention baseline
- `exp_sparse_smoke.yaml` - Sparse attention (quick test)
- `exp_sparse_window_256.yaml` - Sparse with larger window
- `tinyshakespeare_*.yaml` - Various configurations

**Configuration Example:**
```yaml
run:
  out_dir: runs/exp_shakespeare_dense
  seed: 123
  steps: 1000
  log_every: 10
  save_every: 100

data:
  dataset: tinyshakespeare
  seq_len: 256
  batch_size: 16

model:
  name: orion
  d_model: 256
  n_layers: 4
  n_heads: 4
  attention_type: sparse
  window_size: 64
  expander_degree: 8

optim:
  lr: 3e-4
```

## Metrics

Orion logs detailed metrics at multiple frequencies:

**Every Step:**
- `loss`, `ppl`, `throughput_tokens_per_sec`, `grad_norm`, `diverged`

**Every 50 Steps:**
- `vram_peak_mib`, `divergence_rate`, `activation_norm_rms`, `attention_entropy`, `attention_entropy_normalized`

**Once Per Run:**
- `attention_degree`, `compute_proxy_per_token`, `compute_proxy_per_seq`, `compute_proxy_per_step`

**Every 1000 Steps:**
- `eval_ppl_512`, `eval_ppl_1024`, `eval_ppl_2048`, `eval_ppl_4096`

**Format:** All metrics logged to `{run_dir}/metrics.jsonl` (JSONL = JSON Lines):
```json
{"type": "run_metrics", "step": 1, "attention_degree": 72, ...}
{"type": "step", "step": 1, "loss": 5.73, "ppl": 307.56, ...}
{"type": "window", "step": 50, "vram_peak_mib": 2048, ...}
{"type": "eval", "step": 1000, "eval_ppl_512": 12.5, ...}
```

For details, see [COMPREHENSIVE_METRICS_GUIDE.md](COMPREHENSIVE_METRICS_GUIDE.md)

## Project Structure

```
orion/
├── attention/              # Attention mechanisms (dense, sparse, window)
├── data/                   # Data loading (Shakespeare dataset)
├── models/                 # Model components (transformer blocks)
├── model.py                # Main models (TinyDecoderOnly, OrionDecoder)
├── config.py               # Configuration loading
├── train.py                # Training loop with metrics
├── eval.py                 # Evaluation at long contexts
├── metrics.py              # Metrics tracking system
├── logging_utils.py        # JSONL metrics logging
└── train_utils.py          # Training utilities

configs/                     # Training configurations
tests/                       # 136 tests (sparse, dense, metrics, models)
runs/                        # Training outputs (checkpoints, metrics)
```

## Logging & Checkpoints

**Run Directory:**
```
runs/
├── latest/                 # Most recent run
│   ├── checkpoint.pt      # Model weights + optimizer state
│   └── metrics.jsonl      # Training metrics
├── exp_shakespeare_dense/  # Named experiment
└── exp_sparse_smoke/       # Another experiment
```

**Checkpoint Format:**
```python
{
    "model": model.state_dict(),      # Model weights
    "opt": optimizer.state_dict(),    # Optimizer state
    "step": 100,                      # Training step
    "seed": 123,                      # Random seed
    "config": cfg.raw,                # Full config dict
}
```

**Load Checkpoint:**
```python
import torch
ckpt = torch.load("runs/latest/checkpoint.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
seed = ckpt["seed"]
```

## Testing

**Run Tests:**
```bash
make test                   # All 136 tests
pytest tests/test_sparse_attention.py -v  # Specific file
pytest --cov=orion tests/   # With coverage
make smoke                  # Quick 5-step test
```

## Development

**Setup:**
```bash
make dev                    # Install dev dependencies
```

**Commands:**
```bash
make train                  # Full training (configs/golden.yaml)
make smoke                  # Quick 5-step test
make test                   # Run all 136 tests
make lint                   # Lint check (ruff)
make format                 # Auto-format code
make format-check           # Check formatting
make eval                   # Evaluate checkpoint
```

**Training Examples:**
```bash
# Default config
python -m orion.train --config configs/golden.yaml

# Custom parameters
python -m orion.train --config configs/golden.yaml --save-every 50

# Resume from latest
python -m orion.train --config configs/golden.yaml --resume

# Different models/attention
python -m orion.train --config configs/exp_sparse_smoke.yaml
python -m orion.train --config configs/exp_window_256.yaml
```

**Metrics Inspection:**
```bash
tail -n 5 runs/latest/metrics.jsonl                    # Last 5 steps
cat runs/latest/metrics.jsonl | jq .                   # Pretty print
cat runs/latest/metrics.jsonl | jq 'select(.type == "step") | .loss'  # Extract field
```

**Code Quality:**
- Linter: Ruff (E, F, I, B, UP rules)
- Formatter: Ruff format
- Type Checking: Full type annotations
- Tests: 136 tests (sparse, dense, metrics, models)
- Python: 3.10+ (target 3.11)

**CI Pipeline:**
```bash
make test lint format-check  # Local CI (before commit)
```

GitHub Actions runs on every PR:
1. Lint & Format checks
2. Full test suite (136 tests)
3. Smoke test (5-step training)

---

## Contributing

**To contribute:**
1. Fork the repo and create a branch: `git checkout -b my-feature main`
2. Make changes and test: `make test lint format-check`
3. Push and create a PR
4. Wait for CI checks to pass

**Code Style:**
- Follow PEP 8
- Use type annotations
- Add docstrings
- Keep functions focused

---

## License

Apache-2.0
