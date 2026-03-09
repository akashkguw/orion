from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from orion.experiments import build_trial_specs, load_profile_context, load_variant_definitions

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pilot9_profile_builds_expected_trials_from_configs():
    ctx = load_profile_context("pilot9", repo_root=REPO_ROOT)
    variants = load_variant_definitions(ctx)
    specs = build_trial_specs(ctx, variants)

    assert ctx.train_tokens_target == 4_000_000
    assert ctx.run_val_eval is True
    assert ctx.log_every == 10

    variant_ids = {variant.variant_id for variant in variants}
    assert variant_ids == {"dense", "window_w128", "sparse_w64_d8"}
    assert len(specs) == 9


def test_window_and_sparse_trials_skip_window_ge_seq_len():
    ctx = load_profile_context("pilot9", repo_root=REPO_ROOT)
    variants = load_variant_definitions(ctx)

    small_ctx = replace(ctx, seq_lens=[128])
    specs = build_trial_specs(small_ctx, variants)
    backends = [spec.backend for spec in specs]

    # window_w128 is invalid when seq_len=128 and should be filtered.
    assert len(specs) == 2
    assert backends.count("dense") == 1
    assert backends.count("sparse") == 1


def test_w64_d_sweep_profile_contains_sparse_degree_sweep_to_256():
    ctx = load_profile_context("w64_d_sweep", repo_root=REPO_ROOT)
    variants = load_variant_definitions(ctx)

    sparse_variants = [variant for variant in variants if variant.backend == "sparse"]
    degrees = sorted(variant.expander_degree for variant in sparse_variants)
    windows = {variant.window_size for variant in sparse_variants}

    assert windows == {64}
    assert degrees == [8, 16, 32, 64, 128, 256]
