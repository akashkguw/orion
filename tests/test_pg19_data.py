from __future__ import annotations

import pytest
import torch

from orion.data import pg19


class _FakeDatasetsModule:
    def __init__(self, payload):
        self.payload = payload
        self.calls: list[tuple[str, bool]] = []

    def load_dataset(self, dataset_id: str, streaming: bool = True):
        self.calls.append((dataset_id, streaming))
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
    assert fake_module.calls == [("deepmind/pg19", True)]

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
