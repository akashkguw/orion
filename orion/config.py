from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .attention.base import AttentionConfig


def _parse_bool(raw: Any, *, field_name: str) -> bool:
    """Parse a strict bool-ish config value.

    Accepts: bool, 0/1 ints, common true/false strings.
    Raises ValueError for ambiguous values.
    """
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(
        f"Invalid boolean value for stability.{field_name}: {raw!r}. Use true/false (or 1/0)."
    )


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
            sparse_impl = "flex"

        sparse_block_size = attention_section.get("sparse_block_size")
        if sparse_block_size is None:
            sparse_block_size = model_section.get("sparse_block_size")
        if sparse_block_size is None:
            sparse_block_size = 128

        sparse_probe_every = attention_section.get("sparse_probe_every")
        if sparse_probe_every is None:
            sparse_probe_every = model_section.get("sparse_probe_every")
        if sparse_probe_every is None:
            sparse_probe_every = 0

        sparse_probe_tokens = attention_section.get("sparse_probe_tokens")
        if sparse_probe_tokens is None:
            sparse_probe_tokens = model_section.get("sparse_probe_tokens")
        if sparse_probe_tokens is None:
            sparse_probe_tokens = 256

        window_probe_every = attention_section.get("window_probe_every")
        if window_probe_every is None:
            window_probe_every = model_section.get("window_probe_every")
        if window_probe_every is None:
            window_probe_every = 50

        window_probe_tokens = attention_section.get("window_probe_tokens")
        if window_probe_tokens is None:
            window_probe_tokens = model_section.get("window_probe_tokens")
        if window_probe_tokens is None:
            window_probe_tokens = 256

        return AttentionConfig(
            backend=str(backend),
            window_size=int(window) if window is not None else None,
            expander_degree=int(expander) if expander is not None else None,
            sparse_impl=str(sparse_impl),
            sparse_block_size=int(sparse_block_size),
            sparse_probe_every=int(sparse_probe_every),
            sparse_probe_tokens=int(sparse_probe_tokens),
            window_probe_every=int(window_probe_every),
            window_probe_tokens=int(window_probe_tokens),
        )

    def stability_config(self):
        from .stability import StabilityConfig

        s = self.get("stability", default={})
        if s is None:
            s = {}
        if not isinstance(s, dict):
            raise ValueError(f"'stability' must be a mapping, got {type(s).__name__}.")
        return StabilityConfig(
            qk_norm=_parse_bool(s.get("qk_norm", False), field_name="qk_norm"),
            ortho_init=_parse_bool(s.get("ortho_init", False), field_name="ortho_init"),
            spectral_norm=_parse_bool(s.get("spectral_norm", False), field_name="spectral_norm"),
        )


def load_config(path: str) -> OrionConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    return OrionConfig(raw=raw)
