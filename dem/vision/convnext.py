"""DINOv3 ConvNeXt-B vision encoder.

Wraps the Hugging Face ``facebook/dinov3-convnext-base-pretrain-lvd1689m``
model and returns the stage-3 feature map as a token sequence. For a
256-pixel input that is a 16 x 16 grid of 512-dimensional features,
flattened row-major into 256 tokens per image. The token order preserves
the grid.

The Hub repository is gated. When ``pretrained=False`` and the Hub config
cannot be fetched, the architecture is built from the config vendored in
``dem/configs/dinov3_convnext_base``.
"""

from __future__ import annotations

import os

import torch

from dem.vision.base import VisionEncoder

DEFAULT_MODEL = "facebook/dinov3-convnext-base-pretrain-lvd1689m"
VENDORED_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "configs", "dinov3_convnext_base")


class DinoV3ConvNeXtEncoder(VisionEncoder):
    """DINOv3 ConvNeXt with stage-3 (stride 16, 512-d) or final (stride 32, 1024-d) tokens."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: bool = True,
        feature_stage: str = "stage3",
        image_size: int = 256,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if feature_stage not in ("stage3", "final"):
            raise ValueError(f"feature_stage must be 'stage3' or 'final', got {feature_stage!r}")
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            try:
                cfg = AutoConfig.from_pretrained(model_name)
            except Exception:  # gated repo without a token, or offline
                cfg = AutoConfig.from_pretrained(VENDORED_CONFIG_DIR)
            self.model = AutoModel.from_config(cfg)
        self.model_name = model_name
        self.feature_stage = feature_stage
        self.image_size = image_size
        hidden_sizes = list(self.model.config.hidden_sizes)
        if feature_stage == "stage3":
            self.embed_dim = int(hidden_sizes[-2])
            self.stride = 16
        else:
            self.embed_dim = int(hidden_sizes[-1])
            self.stride = 32
        if image_size % self.stride != 0:
            raise ValueError(f"image_size {image_size} must be a multiple of the stride {self.stride}")
        self.grid_size = image_size // self.stride
        self.tokens_per_image = self.grid_size ** 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, 3, S, S]`` in [0, 1] -> ``[B, N, D]`` spatial tokens (row-major grid)."""
        x = self.normalize(images)
        g = images.shape[-1] // self.stride
        if self.feature_stage == "stage3":
            out = self.model(pixel_values=x, output_hidden_states=True)
            h = out.hidden_states[3]                       # [B, 512, S/16, S/16]
            return h.flatten(2).transpose(1, 2)            # [B, N, 512]
        out = self.model(pixel_values=x)
        h = out.last_hidden_state                          # [B, 1 + N, 1024]; first token is pooled
        return h[:, -(g * g):]
