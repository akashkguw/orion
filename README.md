# Orion
A research repo for long-context, decoder-only Transformers using structured sparse attention (sliding window + expander links) and stability controls (QK-norm, orthogonal init, spectral normalization). Includes reproducible training/eval configs and benchmarks across 512–4K context lengths for quality, throughput, VRAM, and training stability

This README includes the exact **Google Colab** steps we use to install, train, eval, and check logs.

Latest Collab Link - https://colab.research.google.com/drive/1loF_wQVC2-tFcUZaBAK22Pz4mQKy9gvG#scrollTo=2mw0bvkX1nqj
---

## Google Colab: Run the boilerplate end-to-end

### 0) Open the notebook
Use the project notebook:
- `Orion-Master.ipynb` OR any name of your choice.

### 1) Install the package (clone + deps)
Run this in a Colab cell:

```bash
!git clone https://github.com/akashkguw/orion.git
%cd orion
!pip -q install -r requirements.txt -r requirements-dev.txt
```

Expected: repo clones to `/content/orion`.

### 2) Check whether `torch` is available
Colab usually has `torch` preinstalled. Check:

```python
import importlib.util
print("torch installed:", importlib.util.find_spec("torch") is not None)
```

If it prints `False`, install torch (otherwise ignore this step):

```bash
!pip -q install torch
```

### 3) Execute training (smoke run)
Run:

```bash
!python -m orion.train --config configs/golden.yaml
```

Expected: prints step-by-step logs like:
- `{'step': 1, 'loss': ..., 'ppl': ...}`
- ...
- `{'step': 20, 'loss': ..., 'ppl': ...}`

It will also write outputs to:
- `runs/latest/metrics.jsonl`
- `runs/latest/checkpoint.pt`

You may see a warning like:
`UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True`

This warning is harmless for the smoke run.

### 4) Run eval
Run:

```bash
!python -m orion.eval --config configs/golden.yaml --checkpoint runs/latest/checkpoint.pt
```

Expected output (example):
- `{'loss': ..., 'ppl': ...}`

### 5) Check logs
Run:

```bash
!tail -n 5 runs/latest/metrics.jsonl
```

Expected: last 5 JSON lines with `step`, `loss`, `ppl`, and `wall_time_s`.

---

## Local run (optional)

If you want to run locally (Mac/Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt -r requirements-dev.txt
pip install torch
python -m orion.train --config configs/golden.yaml
python -m orion.eval --config configs/golden.yaml --checkpoint runs/latest/checkpoint.pt
```

## Run Directory & Logging Convention

All training runs follow a standard structure:

```
runs/
├── latest/                    # Most recent run
│   ├── checkpoint.pt         # Model checkpoint
│   └── metrics.jsonl         # Training metrics (JSONL format)
├── exp_shakespeare_dense/     # Named experiment
│   ├── checkpoint.pt
│   └── metrics.jsonl
```

### Metrics Schema

Metrics are logged to `metrics.jsonl` in JSONL format with standard fields:

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

Checkpoints contain:
- `model`: Model state dict
- `opt`: Optimizer state dict
- `step`: Training step at save time
- `seed`: Random seed for reproducibility
- `config`: Full YAML config

Load a checkpoint:
```python
import torch
from orion.model import TinyDecoderOnly

ckpt = torch.load("runs/latest/checkpoint.pt", map_location="cpu")
model = TinyDecoderOnly(...)
model.load_state_dict(ckpt["model"])
```

## Testing

Run the full test suite:

```bash
make test
```

Tests include:
- **Config tests** (6): YAML loading, hierarchical access, error handling
- **Model tests** (3): Forward pass, loss computation, device handling
- **Data tests** (3): Tokenizer creation, encoding, roundtrip
- **Causal mask tests** (6): Mask pattern, future attention prevention, variable seq_len
- **Checkpoint tests** (3): Save/load, metadata, device transfer
- **Logging tests** (10): JSONL format, field validation, checkpoint structure

Run specific test file:
```bash
pytest tests/test_model.py -v
```

Run with coverage:
```bash
pytest --cov=orion tests/
```

## CI/CD

GitHub Actions runs on every PR and push to main:

1. **Lint & Format** - Ruff checks and formatting
2. **Unit Tests** - Full test suite (29 tests)
3. **Smoke Test** - 20-step training on CPU

All checks must pass before merge.

## Development

Set up development environment:

```bash
make dev
```

Common commands:
```bash
make lint          # Run linter
make format        # Auto-format code
make test          # Run tests
make smoke         # Run 20-step smoke test
make train         # Full training run
make eval          # Evaluate checkpoint
```

## License

Apache-2.0

