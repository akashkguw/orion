from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention.base import AttentionConfig
from .model import TinyDecoderOnly  # existing working model
from .models.blocks import DecoderBlock


@dataclass(frozen=True)
class ModelSpec:
    name: str  # "tiny" | "orion_decoder"


class OrionDecoder(nn.Module):
    """GPT-style decoder with pluggable attention backends.

    Same forward interface as TinyDecoderOnly (idx → logits) so they're
    interchangeable from the training loop.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_mult: int,
        attention_cfg: AttentionConfig,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  # learned, not sinusoidal
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, attention_cfg, mlp_mult) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)  # final norm before output head
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # bias=False: redundant with LN

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: [B, T] token IDs → logits: [B, T, vocab_size]"""
        _B, T = idx.shape

        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos[None, :, :]  # broadcast pos [T, D] → [1, T, D] over batch

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)


def build_model(
    *,
    name: str,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    mlp_mult: int,
    device: torch.device,
    attention_cfg: AttentionConfig | None = None,
) -> nn.Module:
    """Build a model by name. attention_cfg only affects the 'orion' path."""
    name = (name or "tiny").lower()

    if name == "tiny":
        return TinyDecoderOnly(vocab_size, d_model, n_layers, n_heads, mlp_mult).to(device)

    if name in {"orion", "orion_decoder", "sparse_transformer"}:
        if attention_cfg is None:
            attention_cfg = AttentionConfig(backend="dense")
        return OrionDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_mult=mlp_mult,
            attention_cfg=attention_cfg,
        ).to(device)

    raise ValueError(f"Unknown model.name={name!r}")
