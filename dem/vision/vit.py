"""DINOv3 ViT-B/16 vision encoder (timm).

The paper's vision ablation compares the ConvNeXt-B against this encoder.
It returns the patch tokens of ``forward_features`` with the CLS and
register tokens removed: 256 tokens of 768 dimensions for a 256-pixel
input. Requires the optional ``timm`` dependency (``pip install dem[vit]``).
"""

from __future__ import annotations

import torch

from dem.vision.base import VisionEncoder

DEFAULT_MODEL = "vit_base_patch16_dinov3_qkvb.lvd1689m"


class DinoV3ViTEncoder(VisionEncoder):
    def __init__(self, model_name: str = DEFAULT_MODEL, pretrained: bool = True, image_size: int = 256) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ImportError("DinoV3ViTEncoder needs timm: pip install 'dem[vit]'") from e
        self.model = timm.create_model(model_name, pretrained=pretrained, img_size=image_size, num_classes=0)
        self.model_name = model_name
        self.image_size = image_size
        self.embed_dim = int(self.model.embed_dim)
        self.patch_size = int(self.model.patch_embed.patch_size[0])
        self.num_prefix_tokens = int(self.model.num_prefix_tokens)
        self.grid_size = image_size // self.patch_size
        self.tokens_per_image = self.grid_size ** 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, 3, S, S]`` in [0, 1] -> ``[B, N, D]`` patch tokens."""
        feats = self.model.forward_features(self.normalize(images))   # [B, prefix + N, D]
        return feats[:, self.num_prefix_tokens:]
