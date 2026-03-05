from __future__ import annotations

import torch

from orion.attention.base import AttentionConfig
from orion.models_factory import build_model
from orion.train import _find_attention_backends


def _build_orion(attention_cfg: AttentionConfig):
    return build_model(
        name="orion",
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=2,
        mlp_mult=2,
        device=torch.device("cpu"),
        attention_cfg=attention_cfg,
    )


def test_find_sparse_backend_from_decoder_block_attn():
    model = _build_orion(AttentionConfig(backend="sparse", window_size=8, expander_degree=2))
    sparse_backend, dense_backend = _find_attention_backends(model)

    assert sparse_backend is not None
    assert dense_backend is None


def test_find_dense_backend_from_decoder_block_attn():
    model = _build_orion(AttentionConfig(backend="dense"))
    sparse_backend, dense_backend = _find_attention_backends(model)

    assert sparse_backend is None
    assert dense_backend is not None
