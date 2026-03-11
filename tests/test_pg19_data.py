from __future__ import annotations

import pytest
import torch

from orion.data import pg19


class _FakeDatasetsModule:
    def __init__(self, payload):
        self.payload = payload
        self.calls: list[tuple[str, bool, str | None]] = []
        self.script_failures_remaining = 0

    def load_dataset(
        self,
        dataset_id: str,
        streaming: bool = True,
        revision: str | None = None,
    ):
        self.calls.append((dataset_id, streaming, revision))
        if self.script_failures_remaining > 0:
            self.script_failures_remaining -= 1
            raise RuntimeError("Dataset scripts are no longer supported, but found pg19.py")
        return self.payload


def test_load_pg19_builds_cache_and_roundtrips(tmp_path, monkeypatch: pytest.MonkeyPatch):
    fake_module = _FakeDatasetsModule(
        {
            "train": [
                {"text": "alpha beta gamma"},
                {"text": "delta epsilon zeta"},
                {"text": "eta theta iota"},
            ],
            "validation": [
                {"text": "validation one"},
                {"text": "validation two"},
            ],
        }
    )
    monkeypatch.setattr(pg19, "_import_hf_datasets", lambda: fake_module)

    train_ids, val_ids, tok = pg19.load_pg19(
        tmp_path,
        dataset_id="deepmind/pg19",
        train_docs=2,
        val_docs=1,
        streaming=True,
    )

    assert train_ids.numel() > 0
    assert val_ids.numel() > 0
    assert tok.vocab_size > 0
    assert fake_module.calls == [("deepmind/pg19", True, None)]

    # Cached call should not re-hit datasets loader.
    monkeypatch.setattr(
        pg19,
        "_import_hf_datasets",
        lambda: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )
    train_cached, val_cached, tok_cached = pg19.load_pg19(
        tmp_path,
        dataset_id="deepmind/pg19",
        train_docs=2,
        val_docs=1,
        streaming=True,
    )
    assert torch.equal(train_ids, train_cached)
    assert torch.equal(val_ids, val_cached)
    assert tok_cached.vocab_size == tok.vocab_size


def test_load_pg19_uses_test_split_when_validation_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    fake_module = _FakeDatasetsModule(
        {
            "train": [{"text": "train sample one"}],
            "test": [{"text": "test sample one"}],
        }
    )
    monkeypatch.setattr(pg19, "_import_hf_datasets", lambda: fake_module)

    train_ids, val_ids, _ = pg19.load_pg19(
        tmp_path,
        dataset_id="deepmind/pg19",
        streaming=True,
        force_rebuild=True,
    )
    assert train_ids.numel() > 0
    assert val_ids.numel() > 0


def test_load_pg19_normalizes_short_dataset_id(tmp_path, monkeypatch: pytest.MonkeyPatch):
    fake_module = _FakeDatasetsModule(
        {
            "train": [{"text": "train sample one"}],
            "validation": [{"text": "val sample one"}],
        }
    )
    monkeypatch.setattr(pg19, "_import_hf_datasets", lambda: fake_module)

    pg19.load_pg19(tmp_path, dataset_id="pg19", streaming=True, force_rebuild=True)

    assert fake_module.calls == [("deepmind/pg19", True, None)]


def test_load_pg19_falls_back_to_parquet_revision(tmp_path, monkeypatch: pytest.MonkeyPatch):
    fake_module = _FakeDatasetsModule(
        {
            "train": [{"text": "train sample one"}],
            "validation": [{"text": "val sample one"}],
        }
    )
    fake_module.script_failures_remaining = 1
    monkeypatch.setattr(pg19, "_import_hf_datasets", lambda: fake_module)

    train_ids, val_ids, _ = pg19.load_pg19(
        tmp_path,
        dataset_id="deepmind/pg19",
        streaming=True,
        force_rebuild=True,
    )

    assert train_ids.numel() > 0
    assert val_ids.numel() > 0
    assert fake_module.calls == [
        ("deepmind/pg19", True, None),
        ("deepmind/pg19", True, pg19.PARQUET_FALLBACK_REVISION),
    ]


def test_load_pg19_uses_default_dataset_id_after_multiple_script_failures(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    fake_module = _FakeDatasetsModule(
        {
            "train": [{"text": "train sample one"}],
            "validation": [{"text": "val sample one"}],
        }
    )
    fake_module.script_failures_remaining = 3
    monkeypatch.setattr(pg19, "_import_hf_datasets", lambda: fake_module)

    train_ids, val_ids, _ = pg19.load_pg19(
        tmp_path,
        dataset_id="custom/pg19",
        streaming=True,
        force_rebuild=True,
    )

    assert train_ids.numel() > 0
    assert val_ids.numel() > 0
    assert fake_module.calls == [
        ("custom/pg19", True, None),
        ("custom/pg19", True, pg19.PARQUET_FALLBACK_REVISION),
        ("deepmind/pg19", True, None),
        ("deepmind/pg19", True, pg19.PARQUET_FALLBACK_REVISION),
    ]


def test_load_pg19_requires_datasets_dependency(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        pg19,
        "_import_hf_datasets",
        lambda: (_ for _ in ()).throw(
            RuntimeError("Loading PG-19 requires the optional 'datasets' package.")
        ),
    )

    with pytest.raises(RuntimeError, match="datasets"):
        pg19.load_pg19(tmp_path, force_rebuild=True)
