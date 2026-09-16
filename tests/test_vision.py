import importlib.util

import pytest
import torch

from dem.vision import DinoV3ConvNeXtEncoder, preprocess_images


def test_preprocess_uint8_hwc_multi_camera():
    frames = torch.randint(0, 256, (3, 2, 120, 160, 3), dtype=torch.uint8)  # [B, K, H, W, 3]
    x = preprocess_images(frames, size=64)
    assert x.shape == (3, 2, 3, 64, 64)
    assert x.dtype == torch.float32 and 0.0 <= x.min() and x.max() <= 1.0


def test_preprocess_chw_float_passthrough():
    x = torch.rand(2, 3, 64, 64)
    y = preprocess_images(x, size=64)
    assert torch.equal(x, y)


def test_preprocess_rejects_bad_inputs():
    with pytest.raises(ValueError):
        preprocess_images(torch.rand(40, 40, 3) * 255.0, size=32)      # float frames outside [0, 1]
    with pytest.raises(ValueError):
        preprocess_images(torch.randint(0, 255, (40, 40, 1), dtype=torch.uint8), size=32)   # not RGB
    chw = torch.randint(0, 255, (3, 40, 3), dtype=torch.uint8)          # ambiguous: treated as CHW
    assert preprocess_images(chw, size=8).shape == (3, 8, 8)


def test_convnext_stage3_tokens_from_vendored_config():
    enc = DinoV3ConvNeXtEncoder(pretrained=False, feature_stage="stage3", image_size=256).eval()
    assert enc.embed_dim == 512 and enc.tokens_per_image == 256 and enc.grid_size == 16
    with torch.no_grad():
        tok = enc(torch.rand(2, 3, 256, 256))
    assert tok.shape == (2, 256, 512)
    assert torch.isfinite(tok).all()


def test_convnext_load_weights_layouts():
    enc = DinoV3ConvNeXtEncoder(pretrained=False, feature_stage="stage3")
    inner = enc.export_weights()
    other = DinoV3ConvNeXtEncoder(pretrained=False, feature_stage="stage3")
    other.load_weights({"student_timm": inner})           # research-code layout
    for k, v in other.export_weights().items():
        assert torch.equal(v, inner[k])
    other.load_weights(enc.state_dict())                   # wrapper layout


@pytest.mark.skipif(importlib.util.find_spec("timm") is None, reason="timm not installed")
def test_vit_patch_tokens():
    from dem.vision import DinoV3ViTEncoder

    enc = DinoV3ViTEncoder(pretrained=False, image_size=256).eval()
    assert enc.embed_dim == 768 and enc.tokens_per_image == 256
    with torch.no_grad():
        tok = enc(torch.rand(1, 3, 256, 256))
    assert tok.shape == (1, 256, 768)
