"""Common interface of the vision encoders.

Every encoder maps a batch of images in [0, 1] to a sequence of spatial
tokens, `[B, 3, S, S] -> [B, N, D]`, and exposes `embed_dim`, `image_size`
and `tokens_per_image`. The action head only depends on this interface, so
swapping the encoder changes nothing but the head's input projection.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Keys under which earlier research checkpoints stored the inner model's weights.
_LEGACY_WRAPPER_KEYS = ("student_timm", "tower_timm", "model", "state_dict")


class VisionEncoder(nn.Module):
    """Base class: ImageNet normalization buffers plus checkpoint loading."""

    embed_dim: int
    image_size: int
    tokens_per_image: int

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("px_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("px_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] in [0, 1] -> ImageNet-normalized."""
        return (images - self.px_mean) / self.px_std

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def load_weights(self, source: str | os.PathLike | Mapping[str, Any], strict: bool = True) -> None:
        """Load encoder weights from a path or a state dict.

        Accepts three layouts: the state dict of this wrapper (keys start
        with ``model.``), the state dict of the inner ``self.model``, or a
        dict that wraps the inner state dict under one of the legacy keys
        ``student_timm`` / ``tower_timm`` / ``model`` / ``state_dict``.
        """
        sd = source
        if not isinstance(source, Mapping):
            sd = torch.load(source, map_location="cpu", weights_only=True)
        for key in _LEGACY_WRAPPER_KEYS:
            if key in sd and isinstance(sd[key], Mapping):
                sd = sd[key]
                break
        if all(k.startswith("model.") for k in sd):
            self.load_state_dict(dict(sd), strict=strict)
        else:
            self.model.load_state_dict(dict(sd), strict=strict)

    def export_weights(self) -> dict[str, torch.Tensor]:
        """State dict of the inner model, on CPU."""
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
