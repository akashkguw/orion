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

## License

Apache-2.0
