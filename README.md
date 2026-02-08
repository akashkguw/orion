# orion
A research repo for long-context, decoder-only Transformers using structured sparse attention (sliding window + expander links) and stability controls (QK-norm, orthogonal init, spectral normalization). Includes reproducible training/eval configs and benchmarks across 512–4K context lengths for quality, throughput, VRAM, and training stability

This README includes the exact **Google Colab** steps we use to install, train, eval, and check logs.

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

## License

Apache-2.0

