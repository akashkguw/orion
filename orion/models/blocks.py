from __future__ import annotations

import torch
import torch.nn as nn

from orion.attention.base import AttentionBackend, AttentionConfig, build_attention_backend


class DecoderBlock(nn.Module):
    """Pre-LN transformer layer (attention + MLP, both with residuals).

    https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html

    Q/K/V projections live here so backends share weights.
    """

    def __init__(
        self, d_model: int, n_heads: int, attention_cfg: AttentionConfig, mlp_mult: int = 4
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn: AttentionBackend = build_attention_backend(attention_cfg)

        # Separate Q/K/V projections for readability.
        # https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html#positionwise-feed-forward-networks
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_mult),
            nn.GELU(),
            nn.Linear(d_model * mlp_mult, d_model),
        )

    def _attend(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        """Q/K/V project, attend, concat heads, output project."""
        B, T, _D = x.shape

        # [B, T, D] → [B, H, T, Dh] — split d_model into heads
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = self.attn.forward(q, k, v, attn_mask=attn_mask)

        # [B, H, T, Dh] → [B, T, D]
        # .contiguous() because transpose makes strides non-contiguous and view needs contiguous
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(attn_out)

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # https://d2l.ai/chapter_convolutional-modern/resnet.html (residual connections)
        x = x + self._attend(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x
