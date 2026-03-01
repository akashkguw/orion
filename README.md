# Orion

A research framework for long-context, decoder-only Transformers with **structured sparse attention** (sliding window + expander edges) and stability controls (QK-norm, orthogonal init, spectral normalization).

**Key Features:**
- Sparse Attention - O(T*(W+d)) complexity vs O(T^2) for dense (7x faster on 512 tokens)
- Reproducible - Deterministic training with seed control and checkpoint management
- Configurable - YAML-based configs for easy experimentation
- Well-tested - 82+ tests covering all components
- Benchmarked - Configs for 256-4K context lengths

**Table of Contents**
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Sparse Attention](#sparse-attention)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Logging & Checkpoints](#logging--checkpoints)
- [Testing](#testing)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

### Google Colab (Recommended - No Setup Required)

Click to open in Colab: [Orion-Master.ipynb](https://colab.research.google.com/github/akashkguw/orion/blob/main/orion.ipynb)

```bash
# In Colab cell:
!git clone https://github.com/akashkguw/orion.git
%cd orion
!pip -q install -r requirements.txt -r requirements-dev.txt
!python -m orion.train --config configs/golden.yaml
```

**Colab Tips:**
- Pre-configured with GPU access
- All dependencies pre-installed
- Perfect for quick experiments
- Save results to Google Drive

### Local Setup (5 minutes)

```bash
# Clone and setup
git clone https://github.com/akashkguw/orion.git
cd orion

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run first training
python -m orion.train --config configs/golden.yaml
```

---

## Installation

### Requirements

- **Python**: 3.10+
- **PyTorch**: 2.0+
- **GPU** (optional): CUDA 11.8+ for GPU acceleration

### Setup Options

**Option 1: Development Setup (Recommended)**
```bash
make dev
```

**Option 2: Manual Setup**
```bash
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

**Verify Installation:**
```bash
python -c "import orion; print('Orion installed')"
pytest tests/ -q  # Run quick test
```

---

## Sparse Attention

### What is Sparse Attention?

Sparse attention reduces the computational complexity of self-attention from **O(T²)** to **O(T·(W+d))** by attending to only a subset of positions:

- **Local Window** - Dense context for nearby tokens
- **Expander Edges** - Structured long-range connections

This enables efficient modeling of long sequences while maintaining model capacity.

### Architecture

Each query position q attends to:

```
Window:   [q-W+1, ..., q-1, q]           (local context)
Expander: [q-offset₁, q-offset₂, ...]    (long-range)
```

**Example:** Query at position 10 with W=4, d=3:
```
Window:   [7, 8, 9, 10]
Expander: [9, 6, 1]  (via quadratic residues)
Total:    [1, 6, 7, 8, 9, 10]  (7 positions vs 11 for dense)
```

### Performance

| Metric | Dense | Sparse | Speedup |
|--------|-------|--------|---------|
| **Complexity** | O(T²) | O(T·(W+d)) | ~7x |
| **Memory** | O(T²·Dₕ) | O(T·(W+d)·Dₕ) | ~7x |
| **Example (T=512, W=64, d=8)** | 262K pos/query | 36.8K pos/query | 7.1x |

### Key Features

- **Causality** - Only attend to positions ≤ q  
- **Per-Head Variation** - Different patterns per head  
- **Masking** - Respects padding and segment boundaries  
- **Deterministic** - Reproducible across runs  
- **Robust** - Handles edge cases (T=0, early tokens, etc.)

### Configuration

```yaml
model:
  attention_type: sparse
  window_size: 64        # Local window (must be ≥ 1)
  expander_degree: 8     # Long-range neighbors
```

### When to Use

| Scenario | Recommendation |
|----------|---|
| **Long sequences** (512+) | Use sparse |
| **Limited GPU memory** | Use sparse |
| **Short sequences** (<256) | Dense is simpler |
| **Maximum accuracy** | Dense may be better |

### Learn More

For deep technical details, see [SPARSE_ATTENTION_ARCHITECTURE.md](SPARSE_ATTENTION_ARCHITECTURE.md)

---

## Usage

### Training

**Basic training:**
```bash
python -m orion.train --config configs/golden.yaml
```

**With custom parameters:**
```bash
python -m orion.train --config configs/golden.yaml \
  --save-every 50 \
  --seed 42
```

**Resume from checkpoint:**
```bash
# Resume from latest
python -m orion.train --config configs/golden.yaml --resume

# Resume from specific checkpoint
python -m orion.train --config configs/golden.yaml \
  --resume runs/exp_shakespeare_dense/checkpoint.pt
```

**Outputs:**
- `runs/latest/metrics.jsonl` - Training metrics (loss, perplexity, time)
- `runs/latest/checkpoint.pt` - Model checkpoint (weights, optimizer state, config)

### Evaluation

```bash
python -m orion.eval --config configs/golden.yaml \
  --checkpoint runs/latest/checkpoint.pt
```

### Configuration

Configs are YAML files in `configs/` directory. Example structure:

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
  attention_type: sparse
  window_size: 64
  expander_degree: 8

optim:
  lr: 3e-4
  weight_decay: 0.1
```

**Available Configs:**
- `golden.yaml` - Dense attention baseline
- `exp_sparse_smoke.yaml` - Sparse attention (quick test)
- `exp_sparse_window_256.yaml` - Sparse with larger window
- `tinyshakespeare_*.yaml` - Various configurations

---

## Comprehensive Metrics

Orion logs detailed metrics at multiple frequencies to track model quality, efficiency, stability, and sparse attention health.

### Metrics Categories

**Every Step (Always-On):**
- `loss` - Cross-entropy loss
- `ppl` - Perplexity (exp(loss))
- `throughput_tokens_per_sec` - Training speed
- `grad_norm` - Gradient norm (post-clip)
- `grad_norm_pre_clip` - Gradient norm (pre-clip, optional)
- `diverged` - Boolean divergence flag (NaN/Inf detected)

**Every 50 Steps (Windowed):**
- `vram_peak_mib` - Peak GPU memory in MiB
- `divergence_rate` - Fraction of diverged steps in window
- `activation_norm_rms` - Residual stream RMS
- `attention_entropy` - Raw entropy of attention weights
- `attention_entropy_normalized` - Normalized entropy (0-1)

**Once Per Run:**
- `attention_degree` - window_size + expander_degree
- `compute_proxy_per_token` - Attention compute per token
- `compute_proxy_per_step` - Attention compute per step

**Every 1000 Steps (Evaluation):**
- `eval_ppl_512` - Perplexity at 512 tokens
- `eval_ppl_1024` - Perplexity at 1024 tokens
- `eval_ppl_2048` - Perplexity at 2048 tokens
- `eval_ppl_4096` - Perplexity at 4096 tokens

### Metrics File Format

All metrics logged to `{run_dir}/metrics.jsonl` with type field:

```json
{"type": "run_metrics", "step": 1, "attention_degree": 72, ...}
{"type": "step", "step": 1, "loss": 5.73, "ppl": 307.56, ...}
{"type": "window", "step": 50, "vram_peak_mib": 2048, ...}
{"type": "eval", "step": 1000, "eval_ppl_512": 12.5, ...}
```

### Viewing Metrics

```bash
# View all metrics
cat runs/latest/metrics.jsonl | jq .

# View only step metrics
cat runs/latest/metrics.jsonl | jq 'select(.type == "step")'

# Extract specific field
cat runs/latest/metrics.jsonl | jq 'select(.type == "step") | .loss'

# Compare Dense vs Sparse throughput
paste <(cat runs/exp_dense/metrics.jsonl | jq -r 'select(.type == "step") | .throughput_tokens_per_sec') \
      <(cat runs/exp_sparse/metrics.jsonl | jq -r 'select(.type == "step") | .throughput_tokens_per_sec')
```

For detailed metrics documentation, see [COMPREHENSIVE_METRICS_GUIDE.md](COMPREHENSIVE_METRICS_GUIDE.md).

---

## Project Structure

```
orion/
├── attention/              # Attention mechanisms
│   ├── base.py            # Base attention interface & factory
│   ├── dense.py           # Dense attention
│   ├── sparse.py          # Sparse attention (window + expander)
│   ├── window.py          # Sliding window attention
│   └── mask/
│       └── builder.py     # Attention mask utilities
├── data/                  # Data loading
│   ├── __init__.py
│   └── shakespeare.py     # Shakespeare dataset
├── models/                # Model components
│   └── blocks.py          # Transformer blocks
├── model.py               # Main model (TinyDecoderOnly)
├── config.py              # Configuration loading
├── train.py               # Training loop
├── eval.py                # Evaluation script
├── logging_utils.py       # Metrics logging
├── run_paths.py           # Run directory management
└── train_utils.py         # Training utilities

configs/                   # Training configurations
tests/                     # Test suite (82+ tests)
runs/                      # Training outputs
```

---

## Logging & Checkpoints

### Run Directory Convention

All runs are organized under `runs/`:

```
runs/
├── latest/                    # Most recent run (symlink/copy)
│   ├── checkpoint.pt         # Model checkpoint
│   └── metrics.jsonl         # Training metrics
├── exp_shakespeare_dense/     # Named experiment
│   ├── checkpoint.pt
│   └── metrics.jsonl
└── exp_sparse_smoke/          # Another experiment
    ├── checkpoint.pt
    └── metrics.jsonl
```

### Metrics Format

Metrics are logged to `metrics.jsonl` (JSONL = JSON Lines, one object per line):

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

**View metrics:**
```bash
# Last 5 steps
tail -n 5 runs/latest/metrics.jsonl

# Pretty print
cat runs/latest/metrics.jsonl | jq .

# Extract specific field
cat runs/latest/metrics.jsonl | jq '.loss'
```

### Checkpoint Format

Checkpoints are PyTorch `.pt` files containing:

```python
{
    "model": model.state_dict(),           # Model weights
    "opt": optimizer.state_dict(),         # Optimizer state
    "scheduler": scheduler.state_dict(),   # LR scheduler state
    "step": 100,                           # Training step
    "epoch": 100,                          # Training epoch
    "seed": 123,                           # Random seed
    "config": cfg.raw,                     # Full config dict
    "rng_state": {...},                    # RNG state for deterministic resume
}
```

**Load checkpoint in code:**

```python
import torch
from orion.model import TinyDecoderOnly

# Load checkpoint
ckpt = torch.load("runs/latest/checkpoint.pt", map_location="cpu")

# Create model and load weights
model = TinyDecoderOnly(...)
model.load_state_dict(ckpt["model"])

# Access metadata
seed = ckpt["seed"]
config = ckpt["config"]
step = ckpt["step"]
```

---

## Testing

### Run Tests

```bash
# Full test suite
make test

# Specific test file
pytest tests/test_sparse_attention.py -v

# With coverage report
pytest --cov=orion tests/

# Quick smoke test (5 steps)
make smoke
```

### Test Coverage

**82+ tests** across 10 test files:

| Category | Tests | Coverage |
|----------|-------|----------|
| Sparse Attention | 21 | Index generation, forward pass, masking, edge cases |
| Model | 8 | Forward pass, loss computation, device handling |
| Config | 6 | YAML loading, hierarchical access, validation |
| Data | 3 | Tokenizer, encoding, roundtrip |
| Causal Mask | 6 | Mask patterns, causality, variable lengths |
| Checkpoint | 4 | Save/load, metadata, device transfer |
| Logging | 10 | JSONL format, field validation |
| CLI | 4 | Command parsing, argument handling |

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
make format-check  # Check formatting without changes
make test          # Run all tests
make smoke         # Quick 5-step training test
make train         # Full training run
make eval          # Evaluate checkpoint
```

### Code Quality

- **Linter**: Ruff (E, F, I, B, UP rules)
- **Formatter**: Ruff format
- **Type Checking**: Full type annotations
- **Tests**: Pytest with 82+ tests
- **Python**: 3.10+ (target 3.11)

### CI/CD Pipeline

GitHub Actions runs on every PR and push to main:

1. **Lint & Format** - Code quality checks
2. **Unit Tests** - Full test suite (82+ tests)
3. **Smoke Test** - 5-step training on CPU

All checks must pass before merge.

---

## Contributing

### Editing the Colab Notebook

1. **Create a branch:**
   ```bash
   git checkout -b my-feature main
   git push -u origin my-feature
   ```

2. **Open in Colab:**
   - Go to: `https://colab.research.google.com/github/akashkguw/orion/blob/my-feature/orion.ipynb`
   - Make your changes

3. **Save to GitHub:**
   - Click **File → Save a copy in GitHub**
   - Select your branch and save

4. **Create PR:**
   - Open PR from `my-feature` into `main`
   - Wait for CI checks to pass

### Code Style

- Follow PEP 8
- Use type annotations
- Add docstrings to functions
- Keep functions focused and small

---

## License

Apache-2.0
