from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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


def load_config(path: str) -> OrionConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    return OrionConfig(raw=raw)
