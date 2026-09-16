"""DEM: Decoupled Embodiment Model.

Three separate networks meet inside the action head: a DINOv3 ConvNeXt-B
vision encoder that emits spatial tokens, a frozen NeoBERT language encoder
that emits instruction tokens, and a MeanFlow head that maps noise to an
action chunk in one forward pass, reading both token sets through
cross-attention.

    from dem import DEM, DEMConfig, preprocess_images

    policy = DEM(DEMConfig(action_dim=12, proprio_dim=16, chunk=16))
    policy.set_instruction("open the left drawer")
    actions = policy.act(preprocess_images(frames)[None], proprio[None])
"""

from dem.head import MeanFlowObjective, MeanFlowTokenHead
from dem.language import LanguageEncoder, NeoBERT, NeoBERTConfig, NeoBERTForMaskedLM
from dem.normalization import ChainedNormalizer, Normalizer
from dem.policy import DEM, DEMConfig, build_vision_encoder
from dem.vision import DinoV3ConvNeXtEncoder, DinoV3ViTEncoder, VisionEncoder, preprocess_images

__version__ = "0.1.0"

__all__ = [
    "DEM",
    "DEMConfig",
    "build_vision_encoder",
    "MeanFlowTokenHead",
    "MeanFlowObjective",
    "LanguageEncoder",
    "NeoBERT",
    "NeoBERTConfig",
    "NeoBERTForMaskedLM",
    "Normalizer",
    "ChainedNormalizer",
    "VisionEncoder",
    "DinoV3ConvNeXtEncoder",
    "DinoV3ViTEncoder",
    "preprocess_images",
    "__version__",
]
