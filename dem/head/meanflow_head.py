"""MeanFlow action head with token-level conditioning.

The head predicts the average velocity ``u(z_t, r, t | C)`` of a MeanFlow
model over a chunk of ``H`` actions. The noisy chunk enters as ``H`` action
tokens with a learned position embedding; the interval ``(r, t)`` enters
through adaptive layer norm; the observation enters only through
cross-attention into the context

    C = [ V W_v + e_v || L W_l + e_l || q W_p + e_p ]

built from the vision tokens ``V``, the language tokens ``L`` (with their
mask) and the proprioceptive state ``q``, each projected to the head width
and tagged with a learned modality embedding. In the paper's configuration
(512 vision tokens, 32 language tokens, one proprio token) the context has
545 tokens. The output layer is zero-initialized.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dem.head.layers import CrossBlock, SinusoidalPosEmb


class MeanFlowTokenHead(nn.Module):
    # Named configurations (dim, depth, heads). Parameter counts at
    # vis_dim 512, proprio 16, action 12, chunk 16:
    #   lang_dim 384 (mmBERT-small):  small 26.97M, new100m 107.03M, large 282.25M
    #   lang_dim 768 (NeoBERT):       small 27.12M, new100m 107.33M, large 282.64M
    # "new100m" is the configuration reported in the paper (d = 768, D = 8, 12 heads).
    SIZES = {
        "small": (384, 8, 8),
        "new100m": (768, 8, 12),
        "large": (1024, 12, 16),
    }

    @classmethod
    def from_size(cls, size: str, vis_dim: int, lang_dim: int = 768, proprio_dim: int = 16,
                  action_dim: int = 12, chunk: int = 16) -> "MeanFlowTokenHead":
        if size not in cls.SIZES:
            raise ValueError(f"unknown size {size!r}; choose from {list(cls.SIZES)}")
        d, depth, heads = cls.SIZES[size]
        return cls(vis_dim, lang_dim, proprio_dim, action_dim, chunk, dim=d, depth=depth, heads=heads)

    def __init__(self, vis_dim: int, lang_dim: int = 768, proprio_dim: int = 16,
                 action_dim: int = 12, chunk: int = 16,
                 dim: int = 768, depth: int = 8, heads: int = 12) -> None:
        super().__init__()
        self.chunk, self.action_dim = chunk, action_dim
        self.vis_dim, self.lang_dim, self.proprio_dim = vis_dim, lang_dim, proprio_dim
        self.dim, self.depth, self.heads = dim, depth, heads
        self.vis_proj = nn.Linear(vis_dim, dim)
        self.lang_proj = nn.Linear(lang_dim, dim)
        self.prop_proj = nn.Linear(proprio_dim, dim)
        self.modality = nn.Parameter(torch.randn(3, dim) * 0.02)      # vision / language / proprio
        self.r_emb = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.t_emb = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.a_in = nn.Linear(action_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, chunk, dim) * 0.02)
        self.blocks = nn.ModuleList(CrossBlock(dim, heads) for _ in range(depth))
        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out = nn.Linear(dim, action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def build_ctx(self, cond: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and concatenate the conditioning tokens.

        ``cond`` keys: ``vis_tokens [B, Nv, Dv]``, ``lang_tokens [B, Nl, Dl]``,
        optional ``lang_mask [B, Nl]`` (bool, True = valid), ``proprio [B, P]``.
        Returns ``(ctx [B, Nv + Nl + 1, dim], mask [B, Nv + Nl + 1])``.
        """
        vis = self.vis_proj(cond["vis_tokens"]) + self.modality[0]
        lang = self.lang_proj(cond["lang_tokens"]) + self.modality[1]
        prop = self.prop_proj(cond["proprio"])[:, None] + self.modality[2]
        ctx = torch.cat([vis, lang, prop], dim=1)
        B, dev = vis.shape[0], vis.device
        vmask = torch.ones(B, vis.shape[1], dtype=torch.bool, device=dev)
        pmask = torch.ones(B, 1, dtype=torch.bool, device=dev)
        lmask = cond.get("lang_mask")
        if lmask is None:
            lmask = torch.ones(B, lang.shape[1], dtype=torch.bool, device=dev)
        return ctx, torch.cat([vmask, lmask.bool(), pmask], dim=1)

    def forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor, cond: dict,
                return_hidden: bool = False):
        """
        Args:
            z: noisy action chunk ``[B, H, A]``.
            r, t: interval endpoints ``[B]`` (``t`` is the current time, ``r <= t``).
            cond: conditioning dict (see ``build_ctx``). A precomputed
                ``(ctx, mask)`` pair may be passed under the key ``"_ctx"`` to
                reuse the context across several calls.
            return_hidden: also return the hidden state after block ``depth // 2``
                (used by the dispersive regularizer).

        Returns:
            average velocity ``u [B, H, A]`` (and the mid-network hidden state if requested).
        """
        ctx, mask = cond["_ctx"] if "_ctx" in cond else self.build_ctx(cond)
        c = self.r_emb(r) + self.t_emb(t)
        h = self.a_in(z) + self.pos
        mid = None
        for i, blk in enumerate(self.blocks):
            h = blk(h, c, ctx, mask)
            if return_hidden and i == len(self.blocks) // 2:
                mid = h
        u = self.out(self.out_norm(h))
        return (u, mid) if return_hidden else u

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
