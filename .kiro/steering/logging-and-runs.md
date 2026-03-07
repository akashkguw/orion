# Logging Schema & Run Directory Convention

## Run Directory Structure

All training runs are organized under the `runs/` directory with the following convention:

```
runs/
├── latest/                    # Symlink or copy of most recent run
│   ├── checkpoint.pt         # Model checkpoint
│   └── metrics.jsonl         # Training metrics (JSONL format)
├── exp_shakespeare_dense/     # Named experiment
│   ├── checkpoint.pt
│   └── metrics.jsonl
└── tinyshakespeare_smoke/     # Named experiment
    ├── checkpoint.pt
    └── metrics.jsonl
```

### Naming Convention

- **Named runs**: Use descriptive names like `exp_shakespeare_dense`, `exp_sparse_window_512`
- **Latest run**: Always maintain a `runs/latest/` directory pointing to the most recent run
- **Avoid**: Timestamps in directory names (use config names instead for clarity)

## Logging Schema

### Metrics File Format

Metrics are logged to `{run_dir}/metrics.jsonl` in JSONL (JSON Lines) format, one JSON object per line.

#### Standard Fields (per step)

```json
{
  "step": 1,
  "loss": 5.728669166564941,
  "ppl": 307.5596923828125,
  "wall_time_s": 0.03218197822570801
}
```

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Training step number (1-indexed) |
| `loss` | float | Cross-entropy loss value |
| `ppl` | float | Perplexity (exp(loss), clamped to 1e6) |
| `wall_time_s` | float | Cumulative wall-clock time since training start |

#### Optional Fields (GPU only)

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
| `vram_max_mb` | int | Peak GPU memory usage in MB (GPU runs only) |

### Checkpoint File Format

Checkpoints are saved to `{run_dir}/checkpoint.pt` as PyTorch `.pt` files.

#### Checkpoint Structure

```python
{
    "model": model.state_dict(),           # Model weights
    "opt": optimizer.state_dict(),         # Optimizer state
    "step": 100,                           # Training step at save time
    "seed": 123,                           # Random seed used
    "config": cfg.raw,                     # Full config dict
}
```

| Key | Type | Description |
|-----|------|-------------|
| `model` | dict | Model state dict (weights/biases) |
| `opt` | dict | Optimizer state dict (momentum, etc.) |
| `step` | int | Training step number when checkpoint was saved |
| `seed` | int | Random seed for reproducibility |
| `config` | dict | Full YAML config as dict |

## Configuration Convention

Run directories are specified in YAML configs under `run.out_dir`:

```yaml
run:
  out_dir: runs/exp_shakespeare_dense
  seed: 123
  steps: 1000
  log_every: 10
  save_every: 100
```

## Usage Examples

### Training with Named Run

```bash
python -m orion.train --config configs/exp_dense.yaml
# Creates: runs/exp_shakespeare_dense/
```

### Evaluating from Checkpoint

```bash
python -m orion.eval --config configs/exp_dense.yaml --checkpoint runs/exp_shakespeare_dense/checkpoint.pt
```

### Viewing Metrics

```bash
# Last 5 steps
tail -n 5 runs/latest/metrics.jsonl

# All metrics as formatted JSON
cat runs/latest/metrics.jsonl | jq .

# Extract specific field
cat runs/latest/metrics.jsonl | jq '.loss'
```

### Loading Checkpoint in Code

```python
import torch
from orion.model import TinyDecoderOnly

ckpt = torch.load("runs/latest/checkpoint.pt", map_location="cpu")
model = TinyDecoderOnly(...)
model.load_state_dict(ckpt["model"])
seed = ckpt["seed"]
config = ckpt["config"]
```

## Best Practices

1. **Use descriptive run names**: `exp_sparse_window_512` instead of `run_1`
2. **Keep configs in version control**: All configs should be in `configs/` directory
3. **Maintain runs/latest**: Always update or symlink to most recent run
4. **Archive old runs**: Move completed runs to separate directory if needed
5. **Document experiments**: Add comments in config files explaining the experiment
