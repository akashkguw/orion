from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import TinyDecoderOnly  # existing working model


@dataclass(frozen=True)
class ModelSpec:
    name: str  # "tiny" | "orion_decoder"


class OrionDecoder(nn.Module):
    """
    Placeholder for the real Orion decoder-only model (custom blocks + pluggable attention).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("OrionDecoder not implemented yet. Set model.name: tiny for now.")


def build_model(
    *,
    name: str,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    mlp_mult: int,
    device: torch.device,
) -> nn.Module:
    name = (name or "tiny").lower()

    if name == "tiny":
        return TinyDecoderOnly(vocab_size, d_model, n_layers, n_heads, mlp_mult).to(device)

    if name in {"orion", "orion_decoder", "sparse_transformer"}:
        # Future: pass attention config, mask config, etc.
        return OrionDecoder().to(device)

    raise ValueError(f"Unknown model.name={name!r}")
