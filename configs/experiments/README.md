# Experiment Configs

This directory provides a config-first setup for `experiment.ipynb`.

## Layout

- `profiles/*.yaml`: Experiment-level sweeps (token budget, seq lengths, seeds, LR grid, batch rules, and variant list)
- `variants/*.yaml`: Concrete model/attention configurations for each experiment arm

## Notebook usage

In `experiment.ipynb`, set:

```python
PROFILE = "pilot9"  # or "pilot", "full", "pilot_norm"
```

The notebook loads:

`configs/experiments/profiles/<PROFILE>.yaml`

and builds trials from listed variant configs without hardcoded backend/window/sparse branching.

## Profiles

- `pilot9`: 9-run fast sanity (dense + window + sparse)
- `pilot`: broader sweep at moderate cost
- `full`: long-context sweep
- `pilot_norm`: sparse stability sweep (`qk_norm`, `ortho_init`, `spectral_norm`) vs window vs dense
