from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .attention.base import AttentionConfig


@dataclass(frozen=True)
class OrionConfig:
    raw: dict[str, Any]

    @property
    def out_dir(self) -> Path:
        return Path(self.raw["run"]["out_dir"])

    def get(self, *keys: str, default=None):
        d: Any = self.raw
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    def attention_config(self) -> AttentionConfig:
        """Parse attention config.

        Preferred schema:
            attention.backend / attention.window_size / attention.expander_degree
            attention.sparse_impl / attention.sparse_block_size

        Backward-compatible fallback:
            model.attention_type / model.window_size / model.expander_degree
            model.sparse_impl / model.sparse_block_size
        """
        attention_section = self.get("attention", default={})
        model_section = self.get("model", default={})
        if not isinstance(attention_section, dict):
            attention_section = {}
        if not isinstance(model_section, dict):
            model_section = {}

        backend = attention_section.get("backend")
        if backend is None:
            backend = model_section.get("attention_type")
        if backend is None:
            backend = "dense"

        window = attention_section.get("window_size")
        if window is None:
            window = model_section.get("window_size")

        expander = attention_section.get("expander_degree")
        if expander is None:
            expander = model_section.get("expander_degree")

        sparse_impl = attention_section.get("sparse_impl")
        if sparse_impl is None:
            sparse_impl = model_section.get("sparse_impl")
        if sparse_impl is None:
            sparse_impl = "auto"

        sparse_block_size = attention_section.get("sparse_block_size")
        if sparse_block_size is None:
            sparse_block_size = model_section.get("sparse_block_size")
        if sparse_block_size is None:
            sparse_block_size = 128

        return AttentionConfig(
            backend=str(backend),
            window_size=int(window) if window is not None else None,
            expander_degree=int(expander) if expander is not None else None,
            sparse_impl=str(sparse_impl),
            sparse_block_size=int(sparse_block_size),
        )


def load_config(path: str) -> OrionConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    return OrionConfig(raw=raw)
