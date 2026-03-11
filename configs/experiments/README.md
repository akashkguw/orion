# Experiment Configs

This directory provides a config-first setup for `experiment.ipynb`.

## Layout

- `profiles/*.yaml`: Experiment-level sweeps (token budget, seq lengths, seeds, LR grid, batch rules, runner controls, analysis thresholds, and variant list)
- `variants/*.yaml`: Concrete model/attention configurations for each experiment arm

## Notebook usage

In `experiment.ipynb`, set:

```python
PROFILE = "pilot9"  # or "pilot", "full", "pilot_norm", "w64_d_sweep"
```

The notebook now delegates execution to `orion.experiments` and loads:

`configs/experiments/profiles/<PROFILE>.yaml`

All sweep controls come from config:
- `seeds`, `seq_lens`, `lr_grid`
- `fixed_batch_by_seq` / `batch_candidates_by_seq`
- `runner.*` (validation, model-only benchmark, probe disable, retries, overwrite, logging cadence)
- `analysis.*` (winner thresholds)
- `variants[]` list pointing to concrete variant YAML files

No backend-specific trial construction is hardcoded in the notebook.

## Profiles

- `pilot9`: 9-run fast sanity (dense + window + sparse)
- `pilot`: broader sweep at moderate cost
- `full`: long-context sweep
- `pilot_norm`: sparse stability sweep (`qk_norm`, `ortho_init`, `spectral_norm`) vs window vs dense
- `w64_d_sweep`: fixed `window_size=64`, sparse degree sweep (`d=8..256`) vs window+dense
- `sparse_formula_ablation_t4`: sparse expander formula-coefficient sweep (`a,b,c,d`) vs dense/window
- `pg19_core_a100`: PG-19 long-context validation profile for A100 (dense vs window vs sparse)
