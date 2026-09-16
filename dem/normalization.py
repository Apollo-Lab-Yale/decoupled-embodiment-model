"""Affine normalizers for actions and proprioceptive state.

Statistics live in buffers so they are saved with the policy. The
identity is the default; call ``fit_mean_std`` on the training data, or
build from quantiles when the dataset ships ``q01 / q99`` statistics. The
two-stage scheme used for the paper's simulation runs (quantile scaling to
[-1, 1], then z-scoring) is covered by ``ChainedNormalizer``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """``normalize(x) = (x - offset) / scale``; ``unnormalize`` inverts it. Per-feature over the last dim."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("offset", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim))
        # Dims that were constant in the fitting data; `unnormalize` restores the constant exactly.
        self.register_buffer("constant", torch.zeros(dim, dtype=torch.bool))

    @classmethod
    def from_mean_std(cls, mean: torch.Tensor, std: torch.Tensor, min_std: float = 1e-4) -> "Normalizer":
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(std, dtype=torch.float32).flatten()
        n = cls(mean.numel())
        n.offset.copy_(mean)
        n.scale.copy_(std.clamp_min(min_std))
        return n

    @classmethod
    def from_quantiles(cls, q_low: torch.Tensor, q_high: torch.Tensor, eps: float = 1e-6) -> "Normalizer":
        """Map ``[q_low, q_high]`` to ``[-1, 1]``.

        A dim whose range is at most ``eps`` is treated as constant: its range is
        widened to 1 so the constant maps to -1, and ``unnormalize`` returns
        ``q_low`` for it regardless of the input.
        """
        lo = torch.as_tensor(q_low, dtype=torch.float32).flatten()
        hi = torch.as_tensor(q_high, dtype=torch.float32).flatten()
        valid = hi - lo > eps
        rng = torch.where(valid, hi - lo, torch.ones_like(hi))
        n = cls(lo.numel())
        n.offset.copy_(lo + rng / 2)
        n.scale.copy_(rng / 2)
        n.constant.copy_(~valid)
        return n

    def fit_mean_std(self, x: torch.Tensor, min_std: float = 1e-4) -> "Normalizer":
        """Set offset/scale to the mean/std of ``x`` over all leading dims."""
        flat = x.reshape(-1, self.dim).float()
        self.offset.copy_(flat.mean(0))
        self.scale.copy_(flat.std(0).clamp_min(min_std))
        self.constant.zero_()
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset) / self.scale

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.scale + self.offset
        if bool(self.constant.any()):
            y = torch.where(self.constant, (self.offset - self.scale).to(y.dtype), y)
        return y

    forward = normalize


class ChainedNormalizer(nn.Module):
    """Apply several normalizers in order; ``unnormalize`` runs them in reverse."""

    def __init__(self, *stages: Normalizer) -> None:
        super().__init__()
        if not stages:
            raise ValueError("ChainedNormalizer needs at least one stage")
        self.stages = nn.ModuleList(stages)
        self.dim = stages[0].dim

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        for s in self.stages:
            x = s.normalize(x)
        return x

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        for s in reversed(self.stages):
            x = s.unnormalize(x)
        return x

    forward = normalize
