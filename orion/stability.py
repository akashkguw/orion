from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class StabilityConfig:
    qk_norm: bool = False
    ortho_init: bool = False
    spectral_norm: bool = False


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
