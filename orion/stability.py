from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class StabilityConfig:
    qk_norm: bool = False
    ortho_init: bool = False
    spectral_norm: bool = False


def any_stability_enabled(cfg: StabilityConfig) -> bool:
    return bool(cfg.qk_norm or cfg.ortho_init or cfg.spectral_norm)


def effective_stability_for_backend(
    cfg: StabilityConfig, *, attention_backend: str
) -> StabilityConfig:
    """Apply stability controls only for sparse attention backends."""
    if attention_backend.lower() != "sparse":
        return StabilityConfig()
    return cfg


def apply_ortho_init(model: nn.Module) -> None:
    """Apply orthogonal init to attention projection weights in all DecoderBlocks."""
    from orion.models.blocks import DecoderBlock

    for module in model.modules():
        if isinstance(module, DecoderBlock):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = getattr(module, name, None)
                if proj is not None and hasattr(proj, "weight"):
                    nn.init.orthogonal_(proj.weight)


def apply_spectral_norm(model: nn.Module) -> None:
    """Wrap q_proj and k_proj in all DecoderBlocks with spectral_norm."""
    from orion.models.blocks import DecoderBlock

    for module in model.modules():
        if isinstance(module, DecoderBlock):
            module.q_proj = nn.utils.spectral_norm(module.q_proj)
            module.k_proj = nn.utils.spectral_norm(module.k_proj)
