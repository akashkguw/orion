# Orion

A research framework for long-context, decoder-only Transformers with structured sparse attention (sliding window + expander links) and stability controls (QK-norm, orthogonal init, spectral normalization). Includes reproducible training/eval configs and benchmarks across 512–4K context lengths.

**Table of Contents**
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Logging & Checkpoints](#logging--checkpoints)
- [Testing](#testing)
- [Development](#development)
- [License](#license)

---

## Quick Start

### Google Colab (Recommended)

Use our pre-configured notebook: [Orion-Master.ipynb](https://colab.research.google.com/drive/1loF_wQVC2-tFcUZaBAK22Pz4mQKy9gvG#scrollTo=2mw0bvkX1nqj)

```bash
!git clone https://github.com/akashkguw/orion.git
%cd orion
!pip -q install -r requirements.txt -r requirements-dev.txt
!python -m orion.train --config configs/golden.yaml
```

### Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt -r requirements-dev.txt
python -m orion.train --config configs/golden.yaml
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

### Setup

```bash
# Development setup (includes dev tools)
make dev

# Or manual setup
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

---

## Usage

### Training

```bash
python -m orion.train --config configs/golden.yaml
```

Override the checkpoint interval on the command line:

```bash
python -m orion.train --config configs/golden.yaml --save-every 50
```

Resume from the latest checkpoint in the run directory:

```bash
python -m orion.train --config configs/golden.yaml --resume
```

Resume from a specific checkpoint path:

```bash
python -m orion.train --config configs/golden.yaml --resume runs/exp_shakespeare_dense/checkpoint.pt
```

Outputs:
- `runs/latest/metrics.jsonl` - Training metrics
- `runs/latest/checkpoint.pt` - Model checkpoint

### Evaluation

```bash
python -m orion.eval --config configs/golden.yaml --checkpoint runs/latest/checkpoint.pt
```

### Configuration

Configs are YAML files in `configs/` directory. Example:

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
  name: tiny
  d_model: 256
  n_layers: 4
  n_heads: 4
  mlp_mult: 4

optim:
  lr: 3e-4
```

---

## Project Structure

```
orion/
├── attention/          # Attention mechanisms
│   ├── base.py        # Base attention interface
│   ├── sparse.py      # Sparse attention implementation
│   └── mask/          # Attention mask builders
├── data/              # Data loading & preprocessing
│   └── shakespeare.py # Shakespeare dataset
├── models/            # Model components
│   └── blocks.py      # Transformer blocks
├── model.py           # Main model (TinyDecoderOnly)
├── config.py          # Configuration loading
├── train.py           # Training loop
├── eval.py            # Evaluation script
└── logging_utils.py   # Metrics logging

configs/               # Training configurations
tests/                 # Test suite
runs/                  # Training outputs
```

---

## Logging & Checkpoints

### Run Directory Convention

All runs follow this structure:

```
runs/
├── latest/                    # Most recent run
│   ├── checkpoint.pt         # Model checkpoint
│   └── metrics.jsonl         # Training metrics
├── exp_shakespeare_dense/     # Named experiment
│   ├── checkpoint.pt
│   └── metrics.jsonl
```

### Metrics Schema

Metrics are logged to `metrics.jsonl` in JSONL format (one JSON object per line):

```json
{
  "step": 1,
  "loss": 5.728669166564941,
  "ppl": 307.5596923828125,
  "wall_time_s": 0.03218197822570801,
  "vram_max_mb": 2048
}
```

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Training step (1-indexed) |
| `loss` | float | Cross-entropy loss |
| `ppl` | float | Perplexity (exp(loss)) |
| `wall_time_s` | float | Cumulative wall-clock time |
| `vram_max_mb` | int | Peak GPU memory (GPU only) |

### Checkpoint Format

Checkpoints are PyTorch `.pt` files containing:

```python
{
    "model": model.state_dict(),      # Model weights
    "opt": optimizer.state_dict(),    # Optimizer state
    "scheduler": scheduler.state_dict() if scheduler else None,
    "step": 100,                      # Training step
    "epoch": 100,                      # Training epoch
    "seed": 123,                      # Random seed
    "config": cfg.raw,                # Full config
    "rng_state": {...},                # RNG state for deterministic resume
}
```

**Load a checkpoint:**

```python
import torch
from orion.model import TinyDecoderOnly

ckpt = torch.load("runs/latest/checkpoint.pt", map_location="cpu")
model = TinyDecoderOnly(...)
model.load_state_dict(ckpt["model"])
```

---

## Sparse Attention

### Overview

Orion implements **sparse attention** combining a local sliding window with structured long-range expander edges. This reduces complexity from O(T²) to O(T·(W+d)) where W is window size and d is expander degree, enabling efficient long-context modeling.

### Architecture

Each query position attends to:

1. **Local Window** - Dense context for short-range dependencies
   - Positions: [q-W+1, ..., q-1, q]
   - Always present (W ≥ 1)
   - Captures immediate context

2. **Expander Edges** - Structured long-range neighbors
   - Positions: [q-offset₁, q-offset₂, ..., q-offsetₐ]
   - Offsets computed via modular arithmetic: offset = (s² + head_offset) mod n
   - Per-head variation: different offsets per head for diverse patterns
   - Causality enforced: only attend to positions ≤ q

### Example Pattern

For query at position q=10 with window_size=4, expander_degree=3:

```
Window:   [7, 8, 9, 10]
Expander: [10-1²=9, 10-2²=6, 10-3²=1]  (with per-head offset variation)
Union:    [1, 6, 7, 8, 9, 10]  (deduplicated, sorted)
```

### Key Properties

- **Causality**: All attended positions k ≤ q (enforced by construction)
- **Masking**: Respects padding masks and segment boundaries (gathered per neighbor)
- **Determinism**: Reproducible across runs (seed-based per-head offsets)
- **Robustness**: Handles variable sequence lengths, early tokens, and edge cases
- **Complexity**: O(T·(W+d)·Dₕ) vs O(T²·Dₕ) for dense attention

### Configuration

In YAML configs:

```yaml
model:
  attention_type: sparse
  window_size: 64        # Local window size
  expander_degree: 8     # Number of long-range neighbors
```

### Complexity Comparison

| Attention Type | Complexity | Memory | Example (T=512, W=64, d=8) |
|---|---|---|---|
| Dense | O(T²) | O(T²) | 262K positions/query |
| Sparse | O(T·(W+d)) | O(T·(W+d)) | 36.8K positions/query |
| **Speedup** | **~7x** | **~7x** | - |

### Implementation Details

**Index Generation** (`build_sparse_indices`):
- Combines window + expander edges
- Deduplicates overlapping positions
- Refills with additional window positions if needed (maintains degree)
- Pads with -1 for invalid positions (masked in attention)

**Forward Pass**:
- Gathers K, V using sparse indices
- Computes attention scores over sparse neighbors
- Applies validity mask (invalid -1 indices)
- Applies padding/segment masks (gathered per neighbor)
- Softmax and value aggregation

**Per-Head Variation**:
- Each head uses different offset: `head_offset = (head_idx * 7) % n`
- Ensures diverse sparse patterns across heads
- Improves coverage of long-range structure

### Usage Example

```python
from orion.model import TinyDecoderOnly
from orion.config import Config

# Load config with sparse attention
cfg = Config.from_yaml("configs/exp_sparse_smoke.yaml")

# Build model (sparse attention automatically used)
model = TinyDecoderOnly(cfg.model)

# Train normally - sparse attention is transparent
output = model(input_ids)
loss = model.compute_loss(output, targets)
```

### When to Use

- **Long sequences** (512+): Sparse attention reduces memory and compute
- **Limited GPU memory**: O(T·(W+d)) vs O(T²) saves significant VRAM
- **Stable training**: Window ensures short-range context, expander adds long-range
- **Diverse patterns**: Per-head variation helps model learn multiple sparse views

### Limitations

- **Early tokens**: Effective degree smaller for q < W+d (only q+1 valid positions)
- **Gather overhead**: expand().gather() pattern is correct but memory-bandwidth heavy at very long T
- **Not block-sparse**: Uses dense gather, not specialized sparse kernels (future optimization)

---

## Testing

### Run Tests

```bash
# Full test suite
make test

# Specific test file
pytest tests/test_model.py -v

# With coverage
pytest --cov=orion tests/
```

### Test Coverage

**30 total tests** across 6 test files:

| Category | Tests | Coverage |
|----------|-------|----------|
| Config | 6 | YAML loading, hierarchical access, error handling |
| Model | 3 | Forward pass, loss computation, device handling |
| Data | 3 | Tokenizer creation, encoding, roundtrip |
| Causal Mask | 6 | Mask pattern, future attention, variable seq_len |
| Checkpoint | 4 | Save/load, metadata, device transfer |
| Logging | 10 | JSONL format, field validation, structure |

---

## Development

### Setup

```bash
make dev
```

### Common Commands

```bash
make lint          # Run linter (ruff)
make format        # Auto-format code
make format-check  # Check formatting
make test          # Run unit tests
make smoke         # Run 20-step smoke test
make train         # Full training run
make eval          # Evaluate checkpoint
```

### Code Quality

- **Linter**: Ruff (E, F, I, B, UP rules)
- **Formatter**: Ruff format
- **Tests**: Pytest with 29 tests
- **Python**: 3.10+ (target 3.11)

### CI/CD Pipeline

GitHub Actions runs on every PR and push to main:

1. **Lint & Format** - Code quality checks
2. **Unit Tests** - Full test suite (29 tests)
3. **Smoke Test** - 20-step training on CPU

All checks must pass before merge.

---

## Editing the Colab Notebook

1. Create a branch from `main`: `git checkout -b my-branch main`
2. Push the branch: `git push -u origin my-branch`
3. Open the notebook in Colab (Search how to do it) and point it to this repo and that branch, for example: `https://colab.research.google.com/github/akashkguw/orion/blob/my-branch/orion.ipynb`
4. Make your changes (add cells, update commands, etc.)
5. In Colab, go to **File → Save a copy in GitHub**
6. Select the `my-branch` branch and save
7. Open a PR from `my-branch` into `main`

---

## License

Apache-2.0
