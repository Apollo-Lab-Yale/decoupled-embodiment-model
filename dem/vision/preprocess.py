"""Image preprocessing shared by training and inference.

uint8 RGB frames are scaled to [0, 1] and resized to a square with bilinear
interpolation and antialiasing. ImageNet normalization happens inside the
vision encoder, so the tensor returned here stays in [0, 1].
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def preprocess_images(
    frames,
    size: int = 256,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert camera frames to float tensors in [0, 1] at the encoder resolution.

    Args:
        frames: array or tensor with shape ``[..., H, W, 3]`` (uint8 or float
            in [0, 1]) or ``[..., 3, H, W]``. Leading dimensions such as
            batch and camera are kept.
        size: output side length in pixels.
        device: optional device for the result; the resize runs there.
        dtype: floating dtype of the result.

    Returns:
        Tensor of shape ``[..., 3, size, size]`` with values in [0, 1].
    """
    x = frames if torch.is_tensor(frames) else torch.as_tensor(np.asarray(frames))
    if x.ndim < 3:
        raise ValueError(f"expected at least 3 dims [H, W, 3] or [3, H, W], got shape {tuple(x.shape)}")
    lead = x.shape[:-3]
    x = x.reshape(-1, *x.shape[-3:])
    if x.shape[-3] == 3:                        # CHW (a 3-row HWC image is not supported)
        pass
    elif x.shape[-1] == 3:                      # HWC -> CHW
        x = x.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"expected 3 RGB channels in the last or third-last axis, got shape {tuple(frames.shape)}")
    if x.dtype == torch.uint8:
        x = x.to(dtype) / 255.0
    else:
        x = x.to(dtype)
        if x.numel() and (x.max() > 1.0 + 1e-3 or x.min() < -1e-3):
            raise ValueError("float frames must be in [0, 1]; divide uint8-range values by 255 first")
    if device is not None:
        x = x.to(device)
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return x.reshape(*lead, 3, size, size).contiguous()
