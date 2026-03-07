from __future__ import annotations

import json

import torch

from orion.benchmark_regression import BENCHMARK_SPECS, run_benchmarks


def test_run_benchmarks_writes_summary(tmp_path):
    payload = run_benchmarks(
        out_dir=tmp_path,
        device=torch.device("cpu"),
        steps=1,
        seq_len=16,
        batch_size=1,
        seed=0,
        clean=True,
    )
    assert payload["device"] == "cpu"
    assert len(payload["results"]) == len(BENCHMARK_SPECS)

    summary_json = tmp_path / "benchmark_summary.json"
    summary_md = tmp_path / "benchmark_summary.md"
    assert summary_json.exists()
    assert summary_md.exists()

    parsed = json.loads(summary_json.read_text())
    names = {row["name"] for row in parsed["results"]}
    assert names == {spec.name for spec in BENCHMARK_SPECS}
