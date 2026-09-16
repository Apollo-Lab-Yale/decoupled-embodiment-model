"""Building blocks of the MeanFlow action head.

Attention is written out explicitly (no fused SDPA kernels) because the
MeanFlow training loss differentiates the network with a forward-mode
Jacobian-vector product (``torch.func.jvp``), which the fused kernels do
not support. Sequence lengths are short (the action chunk on the query
side, a few hundred context tokens on the key side), so the cost is small.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of a scalar per batch element: ``[B] -> [B, dim]``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ManualAttention(nn.Module):
    """Multi-head self-attention with explicit matmuls (forward-AD compatible)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        assert dim % heads == 0
        self.h, self.dk = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        out = att.softmax(-1) @ v
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class ManualCrossAttention(nn.Module):
    """Multi-head cross-attention: queries from ``x``, keys/values from ``ctx``, boolean key mask."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        assert dim % heads == 0
        self.h, self.dk = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        S = ctx.shape[1]
        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k, v = self.kv(ctx).chunk(2, dim=-1)
        k = k.view(B, S, self.h, self.dk).transpose(1, 2)
        v = v.view(B, S, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if ctx_mask is not None:
            att = att.masked_fill(~ctx_mask[:, None, None, :], float("-inf"))
        out = att.softmax(-1) @ v
        return self.proj(out.transpose(1, 2).reshape(B, T, D))


class AdaLNBlock(nn.Module):
    """DiT block: self-attention and MLP, both modulated by adaptive layer norm from ``c``.

    The modulation MLP is zero-initialized, so at initialization the block
    is a plain pre-norm transformer block with zero residual gates.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = ManualAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1[:, None]) + b1[:, None]
        x = x + g1[:, None] * self.attn(h)
        h = self.norm2(x) * (1 + s2[:, None]) + b2[:, None]
        x = x + g2[:, None] * self.mlp(h)
        return x


class CrossBlock(nn.Module):
    """AdaLN self-attention block followed by gated cross-attention into the context tokens.

    The per-channel gate starts at zero, so each block begins as an
    unconditional denoiser and learns how much of the context to read.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.ada_block = AdaLNBlock(dim, heads)
        self.norm_x = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross = ManualCrossAttention(dim, heads)
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
        x = self.ada_block(x, c)
        return x + self.gate * self.cross(self.norm_x(x), ctx, ctx_mask)
