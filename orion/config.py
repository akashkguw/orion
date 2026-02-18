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
        """Build AttentionConfig from the 'attention' section, defaulting to dense."""
        backend = str(self.get("attention", "backend", default="dense"))
        window = self.get("attention", "window_size")
        expander = self.get("attention", "expander_degree")
        return AttentionConfig(
            backend=backend,
            window_size=int(window) if window is not None else None,
            expander_degree=int(expander) if expander is not None else None,
        )


def load_config(path: str) -> OrionConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    return OrionConfig(raw=raw)
