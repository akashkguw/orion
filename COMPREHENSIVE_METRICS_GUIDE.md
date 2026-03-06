# Comprehensive Metrics Logging Guide

## Overview

Orion now logs comprehensive metrics across multiple frequencies to track model quality, efficiency, stability, and sparse attention health.

## Metrics Categories

### 1. Always-On (Every Training Step)

Logged every step to `metrics.jsonl` with `"type": "step"`.

#### Core Model Quality
- **loss** (float): Cross-entropy loss value
- **ppl** (float): Perplexity = exp(loss), clamped to avoid overflow

#### Efficiency
- **throughput_tokens_per_sec** (float): Tokens processed per second
  - Computed as: (batch_size * seq_len) / step_time_seconds
  - Accounts for non-padded tokens only

#### Stability
- **grad_norm** (float): Global gradient norm after clipping
  - Computed as: sqrt(sum(||grad_p||^2 for all parameters p))
- **grad_norm_pre_clip** (float, optional): Gradient norm before clipping
  - Useful for detecting gradient explosion before clipping
- **diverged** (bool): Divergence flag
  - True if: isnan(loss) or isinf(loss) or isnan(grad_norm) or isinf(grad_norm)

#### Example Step Metric
```json
{
  "type": "step",
  "step": 100,
  "loss": 2.5568,
  "ppl": 12.895,
  "throughput_tokens_per_sec": 1024.5,
  "grad_norm": 0.0234,
  "grad_norm_pre_clip": 0.0234,
  "diverged": false
}
```

### 2. Frequent (Every 50 Steps)

Logged every 50 steps to `metrics.jsonl` with `"type": "window"`.

#### Efficiency
- **vram_peak_mib** (int): Peak GPU memory in MiB over last 50 steps
  - Computed from: torch.cuda.max_memory_allocated() / (1024 * 1024)
  - Reset at start of each window

#### Stability
- **divergence_rate** (float): Fraction of diverged steps in last 50
  - Computed as: (# diverged steps) / 50
  - Range: [0.0, 1.0]

#### Activation Health
- **activation_norm_rms** (float): RMS of residual stream activations
  - Computed as: sqrt(mean(activation^2))
  - Averaged across layers or uses last layer

#### Sparse Attention Health
- **attention_entropy** (float): Raw entropy of attention weights
  - Computed as: -sum(p * log(p)) over sparse neighbors
  - Units: nats
- **attention_entropy_normalized** (float): Normalized entropy
  - Computed as: entropy / log(degree)
  - Range: [0.0, 1.0] where 1.0 = uniform distribution

#### Example Window Metric
```json
{
  "type": "window",
  "step": 50,
  "vram_peak_mib": 2048,
  "divergence_rate": 0.0,
  "activation_norm_rms": 0.125,
  "attention_entropy": 1.234,
  "attention_entropy_normalized": 0.85
}
```

### 3. Once Per Run

Logged at training start with `"type": "run_metrics"`.

#### Efficiency - Attention Compute Proxy
- **attention_degree** (int): Total attention degree
  - Computed as: window_size + expander_degree
- **compute_proxy_per_token** (int): Attention compute per token
  - Computed as: seq_len * degree
- **compute_proxy_per_step** (int): Attention compute per step
  - Computed as: batch_size * n_heads * seq_len * degree

#### Example Run Metric
```json
{
  "type": "run_metrics",
  "step": 1,
  "attention_degree": 72,
  "compute_proxy_per_token": 36864,
  "compute_proxy_per_step": 18874368
}
```

### 4. Evaluation Suite (Every 1000 Steps or Checkpoint)

Logged to `metrics.jsonl` with `"type": "eval"`.

#### Long Context Behavior
- **eval_ppl_512** (float): Perplexity at 512 token context
- **eval_ppl_1024** (float): Perplexity at 1024 token context
- **eval_ppl_2048** (float): Perplexity at 2048 token context
- **eval_ppl_4096** (float): Perplexity at 4096 token context

#### Example Eval Metric
```json
{
  "type": "eval",
  "step": 1000,
  "eval_ppl_512": 12.5,
  "eval_ppl_1024": 13.2,
  "eval_ppl_2048": 14.1,
  "eval_ppl_4096": 15.3
}
```

## Metrics File Format

All metrics are logged to `{run_dir}/metrics.jsonl` in JSONL format (one JSON object per line).

### File Structure

```
{"type": "run_metrics", "step": 1, "attention_degree": 72, ...}
{"type": "step", "step": 1, "loss": 5.73, "ppl": 307.56, ...}
{"type": "step", "step": 2, "loss": 5.70, "ppl": 297.91, ...}
...
{"type": "window", "step": 50, "vram_peak_mib": 2048, ...}
{"type": "step", "step": 51, "loss": 5.69, "ppl": 295.42, ...}
...
{"type": "eval", "step": 1000, "eval_ppl_512": 12.5, ...}
```

## Viewing Metrics

### View All Metrics
```bash
cat runs/latest/metrics.jsonl | jq .
```

### View Only Step Metrics
```bash
cat runs/latest/metrics.jsonl | jq 'select(.type == "step")'
```

### View Only Window Metrics
```bash
cat runs/latest/metrics.jsonl | jq 'select(.type == "window")'
```

### Extract Specific Field
```bash
# Get all loss values
cat runs/latest/metrics.jsonl | jq 'select(.type == "step") | .loss'

# Get all throughput values
cat runs/latest/metrics.jsonl | jq 'select(.type == "step") | .throughput_tokens_per_sec'

# Get divergence rates
cat runs/latest/metrics.jsonl | jq 'select(.type == "window") | .divergence_rate'
```

### Plot Metrics (Python)

```python
import json
import matplotlib.pyplot as plt

# Load metrics
step_metrics = []
window_metrics = []
with open('runs/latest/metrics.jsonl') as f:
    for line in f:
        m = json.loads(line)
        if m['type'] == 'step':
            step_metrics.append(m)
        elif m['type'] == 'window':
            window_metrics.append(m)

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Loss and PPL
steps = [m['step'] for m in step_metrics]
losses = [m['loss'] for m in step_metrics]
ppls = [m['ppl'] for m in step_metrics]

axes[0, 0].plot(steps, losses)
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')

axes[0, 1].plot(steps, ppls)
axes[0, 1].set_ylabel('Perplexity')
axes[0, 1].set_title('Training Perplexity')

# Throughput and Grad Norm
throughputs = [m['throughput_tokens_per_sec'] for m in step_metrics]
grad_norms = [m['grad_norm'] for m in step_metrics]

axes[1, 0].plot(steps, throughputs)
axes[1, 0].set_ylabel('Throughput (tok/s)')
axes[1, 0].set_title('Training Throughput')

axes[1, 1].plot(steps, grad_norms)
axes[1, 1].set_ylabel('Gradient Norm')
axes[1, 1].set_title('Gradient Norm')

plt.tight_layout()
plt.show()

# Plot window metrics
if window_metrics:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    w_steps = [m['step'] for m in window_metrics]
    vram = [m['vram_peak_mib'] for m in window_metrics]
    div_rate = [m['divergence_rate'] for m in window_metrics]
    act_norm = [m['activation_norm_rms'] for m in window_metrics]
    attn_entropy = [m['attention_entropy_normalized'] for m in window_metrics]
    
    axes[0, 0].plot(w_steps, vram)
    axes[0, 0].set_ylabel('VRAM (MiB)')
    axes[0, 0].set_title('Peak VRAM')
    
    axes[0, 1].plot(w_steps, div_rate)
    axes[0, 1].set_ylabel('Divergence Rate')
    axes[0, 1].set_title('Divergence Rate (windowed)')
    
    axes[1, 0].plot(w_steps, act_norm)
    axes[1, 0].set_ylabel('Activation RMS')
    axes[1, 0].set_title('Residual Stream RMS')
    
    axes[1, 1].plot(w_steps, attn_entropy)
    axes[1, 1].set_ylabel('Normalized Entropy')
    axes[1, 1].set_title('Attention Entropy')
    
    plt.tight_layout()
    plt.show()
```

## Comparing Dense vs Sparse

### Training Efficiency
```bash
# Compare throughput
paste <(cat runs/exp_dense/metrics.jsonl | jq -r 'select(.type == "step") | "\(.step) \(.throughput_tokens_per_sec)"') \
      <(cat runs/exp_sparse/metrics.jsonl | jq -r 'select(.type == "step") | "\(.step) \(.throughput_tokens_per_sec)"') | \
      awk '{print $1, $2, $4, $4/$2}'
```

### Memory Usage
```bash
# Compare peak VRAM
paste <(cat runs/exp_dense/metrics.jsonl | jq -r 'select(.type == "window") | "\(.step) \(.vram_peak_mib)"') \
      <(cat runs/exp_sparse/metrics.jsonl | jq -r 'select(.type == "window") | "\(.step) \(.vram_peak_mib)"')
```

### Model Quality
```bash
# Compare convergence
paste <(cat runs/exp_dense/metrics.jsonl | jq -r 'select(.type == "step") | "\(.step) \(.loss)"') \
      <(cat runs/exp_sparse/metrics.jsonl | jq -r 'select(.type == "step") | "\(.step) \(.loss)"')
```

## Implementation Details

### MetricsTracker Class

Located in `orion/metrics.py`, provides methods for:

- `compute_throughput()` - Tokens per second
- `compute_grad_norm()` - Global gradient norm
- `check_divergence()` - Detect NaN/Inf
- `compute_activation_norm()` - Residual stream RMS
- `compute_attention_entropy()` - Attention weight entropy
- `record_step_metrics()` - Log step-level metrics
- `record_window_metrics()` - Log windowed metrics
- `record_run_metrics()` - Log run-level metrics
- `record_eval_metrics()` - Log evaluation metrics

### Integration in Training Loop

The training loop (`orion/train.py`) now:

1. Creates `MetricsTracker` instance
2. Logs run metrics at start
3. For each step:
   - Computes throughput and gradient norm
   - Records step metrics
   - Every 50 steps: records window metrics
   - Every 1000 steps: runs evaluation suite

## Best Practices

1. **Monitor divergence_rate** - If > 0, training is unstable
2. **Track throughput** - Compare Dense vs Sparse efficiency
3. **Watch activation_norm_rms** - Sudden changes indicate issues
4. **Check attention_entropy** - Low entropy = concentrated attention
5. **Compare eval_ppl curves** - Sparse should match or beat Dense
6. **Use grad_norm_pre_clip** - Detect gradient explosion before clipping

## Future Enhancements

- [ ] Capture actual attention weights for entropy computation
- [ ] Add layer-wise activation norms
- [ ] Add loss landscape visualization
- [ ] Add gradient flow analysis
- [ ] Add weight distribution tracking
- [ ] Add learning rate scheduling metrics
