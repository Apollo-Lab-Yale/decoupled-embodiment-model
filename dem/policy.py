"""The assembled DEM policy: vision encoder + language encoder + MeanFlow head.

The three modules meet only inside the head's cross-attention. The vision
encoder is trained together with the head (or frozen), the language
encoder stays frozen and its tokens are cached per instruction, and the
head is the only module that consumes the context.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from dem.head import MeanFlowObjective, MeanFlowTokenHead
from dem.language import LanguageEncoder
from dem.normalization import ChainedNormalizer, Normalizer
from dem.vision import DinoV3ConvNeXtEncoder, DinoV3ViTEncoder, VisionEncoder

VISION_ENCODERS = {
    "dinov3_convnext_base": DinoV3ConvNeXtEncoder,
    "dinov3_vit_base": DinoV3ViTEncoder,
}


@dataclass
class DEMConfig:
    # Vision
    vision_encoder: str = "dinov3_convnext_base"     # key of VISION_ENCODERS
    vision_model_name: str | None = None             # Hub id / timm name; None = encoder default
    vision_feature_stage: str = "stage3"             # DinoV3ConvNeXtEncoder only
    vision_pretrained: bool = True
    freeze_vision: bool = False
    image_size: int = 256
    num_cameras: int = 2
    # Language
    language_model: str = "chandar-lab/NeoBERT"
    language_backend: str = "neobert"                # "neobert" | "hf"
    language_pretrained: bool = True
    freeze_language: bool = True
    max_language_tokens: int = 32
    # Head
    head_size: str = "new100m"                       # key of MeanFlowTokenHead.SIZES
    proprio_dim: int = 16
    action_dim: int = 12
    chunk: int = 16
    # MeanFlow objective
    ratio_r_eq_t: float = 0.5
    dispersive_weight: float = 0.25
    dispersive_tau: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], ignore_unknown: bool = False) -> "DEMConfig":
        """Build from a dict. Unknown keys raise ``TypeError`` unless ``ignore_unknown`` is set."""
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown and not ignore_unknown:
            raise TypeError(f"unknown DEMConfig keys: {unknown}")
        return cls(**{k: v for k, v in d.items() if k in names})


def build_vision_encoder(cfg: DEMConfig) -> VisionEncoder:
    if cfg.vision_encoder not in VISION_ENCODERS:
        raise ValueError(f"unknown vision_encoder {cfg.vision_encoder!r}; choose from {list(VISION_ENCODERS)}")
    kwargs: dict[str, Any] = {"pretrained": cfg.vision_pretrained, "image_size": cfg.image_size}
    if cfg.vision_model_name:
        kwargs["model_name"] = cfg.vision_model_name
    if cfg.vision_encoder == "dinov3_convnext_base":
        kwargs["feature_stage"] = cfg.vision_feature_stage
    return VISION_ENCODERS[cfg.vision_encoder](**kwargs)


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def _normalizer_stages(m: nn.Module) -> int:
    return len(m.stages) if isinstance(m, ChainedNormalizer) else 1


def _build_normalizer(dim: int, stages: int) -> nn.Module:
    if stages <= 1:
        return Normalizer(dim)
    return ChainedNormalizer(*[Normalizer(dim) for _ in range(stages)])


def _chain(stages: list[Normalizer]) -> nn.Module:
    return stages[0] if len(stages) == 1 else ChainedNormalizer(*stages)


class DEM(nn.Module):
    """Decoupled Embodiment Model.

    Inputs at every control step: camera frames ``[B, K, 3, S, S]`` in [0, 1]
    (see ``dem.preprocess_images``), the proprioceptive state ``[B, P]`` in
    raw units, and an instruction (a string, or cached tokens). Output: a
    chunk of ``H`` actions ``[B, H, A]`` in raw units, provided
    ``action_norm`` / ``proprio_norm`` hold the training statistics.
    """

    def __init__(self, config: DEMConfig | None = None, vision: VisionEncoder | None = None,
                 language: LanguageEncoder | None = None) -> None:
        super().__init__()
        self.config = cfg = config or DEMConfig()
        self.vision = vision if vision is not None else build_vision_encoder(cfg)
        self.language = language if language is not None else LanguageEncoder(
            cfg.language_model, backend=cfg.language_backend, max_tokens=cfg.max_language_tokens,
            pretrained=cfg.language_pretrained, freeze=cfg.freeze_language)
        self.head = MeanFlowTokenHead.from_size(
            cfg.head_size, vis_dim=self.vision.embed_dim, lang_dim=self.language.embed_dim,
            proprio_dim=cfg.proprio_dim, action_dim=cfg.action_dim, chunk=cfg.chunk)
        self.action_norm: nn.Module = Normalizer(cfg.action_dim)
        self.proprio_norm: nn.Module = Normalizer(cfg.proprio_dim)
        self.objective = MeanFlowObjective(ratio_r_eq_t=cfg.ratio_r_eq_t,
                                           dispersive_weight=cfg.dispersive_weight,
                                           dispersive_tau=cfg.dispersive_tau)
        if cfg.freeze_vision:
            _freeze(self.vision)
        # Whether the frozen towers still hold the public Hub weights (then `save` may omit them).
        self._vision_weights_public = cfg.vision_pretrained and vision is None
        self._language_weights_public = cfg.language_pretrained and language is None
        self._instruction: tuple[torch.Tensor, torch.Tensor] | None = None
        self._lang_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------ encoders
    def train(self, mode: bool = True) -> "DEM":
        super().train(mode)
        if self.config.freeze_vision:
            self.vision.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.head.out.weight.device

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.head.vis_proj.weight.dtype

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, K, 3, S, S]`` (or ``[B, 3, S, S]``) in [0, 1] -> ``[B, K * N, D]``.

        Camera token sequences are concatenated along the token axis in camera order.
        """
        if images.ndim == 4:
            images = images[:, None]
        B, K = images.shape[:2]
        x = images.reshape(B * K, *images.shape[2:]).to(self.device)
        if self.config.freeze_vision:
            with torch.no_grad():
                tok = self.vision(x)
        else:
            tok = self.vision(x)
        return tok.reshape(B, K * tok.shape[1], tok.shape[-1])

    def encode_language(self, texts: Sequence[str] | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Instruction strings -> (``tokens [B, Nl, Dl]``, ``mask [B, Nl]``).

        With a frozen language encoder every string is encoded once and cached.
        With a trainable encoder nothing is cached, so gradients reach it.
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        dev = self.device
        if not self.language.frozen:
            tok, mask = self.language(texts)
            return tok.to(dev), mask.to(dev)
        missing = [t for t in dict.fromkeys(texts) if t not in self._lang_cache]
        if missing:
            tok, mask = self.language(missing)
            for i, t in enumerate(missing):
                self._lang_cache[t] = (tok[i].detach().to(dev), mask[i].to(dev))
        toks, masks = [], []
        for t in texts:
            tok_t, mask_t = self._lang_cache[t]
            if tok_t.device != dev:                      # the policy was moved after caching
                tok_t, mask_t = tok_t.to(dev), mask_t.to(dev)
                self._lang_cache[t] = (tok_t, mask_t)
            toks.append(tok_t)
            masks.append(mask_t)
        return torch.stack(toks), torch.stack(masks)

    def set_instruction(self, texts: Sequence[str] | str) -> None:
        """Encode and store the instruction used by ``act`` when no text is given."""
        self._instruction = self.encode_language(texts)

    def set_instruction_tokens(self, tokens: torch.Tensor, mask: torch.Tensor) -> None:
        """Store pre-encoded instruction tokens ``[B or 1, Nl, Dl]`` and mask ``[B or 1, Nl]`` for ``act``."""
        if tokens.ndim == 2:
            tokens, mask = tokens[None], mask[None]
        self._instruction = (tokens.detach().to(self.device), mask.to(self.device))

    def cache_instruction(self, text: str, tokens: torch.Tensor, mask: torch.Tensor) -> None:
        """Register pre-encoded tokens ``[Nl, Dl]`` and mask ``[Nl]`` for ``text`` (no tokenizer needed)."""
        self._lang_cache[text] = (tokens.detach().to(self.device), mask.to(self.device))

    def clear_language_cache(self) -> None:
        self._lang_cache.clear()
        self._instruction = None

    def build_cond(self, vis_tokens: torch.Tensor, lang_tokens: torch.Tensor,
                   lang_mask: torch.Tensor, proprio: torch.Tensor) -> dict:
        """Assemble the head's conditioning dict; ``proprio`` is in raw units.

        Every tensor is cast to the head's parameter dtype, so features computed
        under autocast or read from a half-precision cache can be passed directly.
        """
        dev, dt = self.device, self.compute_dtype
        prop = self.proprio_norm.normalize(proprio.to(dev).float())
        return {"vis_tokens": vis_tokens.to(dev, dt),
                "lang_tokens": lang_tokens.to(dev, dt),
                "lang_mask": lang_mask.to(dev),
                "proprio": prop.to(dt)}

    def _language_for(self, texts, lang, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if lang is not None:
            tok, mask = lang
        elif texts is not None:
            tok, mask = self.encode_language(texts)
        elif self._instruction is not None:
            tok, mask = self._instruction
            if tok.device != self.device:
                self._instruction = (tok.to(self.device), mask.to(self.device))
                tok, mask = self._instruction
        else:
            raise ValueError("no instruction: pass `texts=`, `lang=`, or call set_instruction() first")
        if tok.shape[0] == 1 and batch_size > 1:
            tok, mask = tok.expand(batch_size, -1, -1), mask.expand(batch_size, -1)
        if tok.shape[0] != batch_size:
            raise ValueError(f"language batch {tok.shape[0]} does not match observation batch {batch_size}")
        return tok.to(self.device), mask.to(self.device)

    # ------------------------------------------------------------------ training / inference
    def loss(self, images: torch.Tensor | None, proprio: torch.Tensor, actions: torch.Tensor,
             texts: Sequence[str] | None = None, lang: tuple[torch.Tensor, torch.Tensor] | None = None,
             vis_tokens: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """MeanFlow loss on a batch. ``actions`` ``[B, H, A]`` and ``proprio`` ``[B, P]`` are in raw units.

        Pass ``vis_tokens`` to skip ``encode_vision`` (e.g. to run the encoder
        under autocast or from a feature cache).
        """
        if vis_tokens is None:
            if images is None:
                raise ValueError("either images or vis_tokens is required")
            vis_tokens = self.encode_vision(images)
        B = vis_tokens.shape[0]
        lang_tok, lang_mask = self._language_for(texts, lang, B)
        cond = self.build_cond(vis_tokens, lang_tok, lang_mask, proprio)
        x = self.action_norm.normalize(actions.to(self.device).float()).to(self.compute_dtype)
        return self.objective.loss(self.head, x, cond)

    @torch.no_grad()
    def act(self, images: torch.Tensor | None, proprio: torch.Tensor,
            texts: Sequence[str] | str | None = None, lang: tuple[torch.Tensor, torch.Tensor] | None = None,
            vis_tokens: torch.Tensor | None = None, nfe: int = 1, z0: torch.Tensor | None = None) -> torch.Tensor:
        """One inference step -> action chunk ``[B, H, A]`` in raw units.

        With ``texts=None`` and ``lang=None`` the instruction set by
        ``set_instruction`` is reused, which is the cached language pathway.
        """
        if vis_tokens is None:
            if images is None:
                raise ValueError("either images or vis_tokens is required")
            vis_tokens = self.encode_vision(images)
        B = vis_tokens.shape[0]
        if isinstance(texts, str):
            texts = [texts]
        lang_tok, lang_mask = self._language_for(texts, lang, B)
        cond = self.build_cond(vis_tokens, lang_tok, lang_mask, proprio)
        z = self.objective.sample(self.head, cond, self.config.chunk, self.config.action_dim, nfe=nfe, z0=z0)
        return self.action_norm.unnormalize(z.float())

    # ------------------------------------------------------------------ utilities
    def parameter_groups(self, lr_head: float = 1e-4, lr_vision: float = 5e-5) -> list[dict]:
        """Optimizer parameter groups for the trainable modules (frozen modules are skipped)."""
        groups = [{"params": list(self.head.parameters()), "lr": lr_head}]
        if not self.config.freeze_vision:
            groups.append({"params": list(self.vision.parameters()), "lr": lr_vision})
        if not self.config.freeze_language:
            groups.append({"params": [p for p in self.language.parameters() if p.requires_grad], "lr": lr_vision})
        return groups

    def count_parameters(self) -> dict[str, int]:
        return {"vision": sum(p.numel() for p in self.vision.parameters()),
                "language": sum(p.numel() for p in self.language.parameters()),
                "head": sum(p.numel() for p in self.head.parameters())}

    def load_vision_weights(self, source, strict: bool = True) -> None:
        """Replace the vision encoder weights (path or state dict, any layout of ``VisionEncoder.load_weights``)."""
        self.vision.load_weights(source, strict=strict)
        self._vision_weights_public = False

    def save(self, path: str | os.PathLike, include_frozen: bool = False) -> None:
        """Save config, normalizers and weights.

        With ``include_frozen=False`` the weights of a frozen tower are omitted
        when they are still the public checkpoint the policy was built from;
        ``load`` then downloads them again. Towers whose weights were replaced
        (``load_vision_weights``, ``load_research_checkpoints``) are always saved.
        """
        sd = self.state_dict()
        dropped = []
        if not include_frozen:
            if self.config.freeze_vision and self._vision_weights_public:
                dropped.append("vision.")
            if self.config.freeze_language and self._language_weights_public:
                dropped.append("language.")
            sd = {k: v for k, v in sd.items() if not k.startswith(tuple(dropped))}
        torch.save({"format": "dem-v1", "config": self.config.to_dict(), "state_dict": sd,
                    "omitted_prefixes": dropped,
                    "normalizers": {"action": _normalizer_stages(self.action_norm),
                                    "proprio": _normalizer_stages(self.proprio_norm)}}, path)

    @classmethod
    def load(cls, path: str | os.PathLike, map_location="cpu", **config_overrides) -> "DEM":
        """Rebuild a policy saved with ``save``. ``config_overrides`` must be ``DEMConfig`` field names."""
        ck = torch.load(path, map_location=map_location, weights_only=True)
        cfg = DEMConfig.from_dict(ck["config"], ignore_unknown=True)
        unknown = sorted(set(config_overrides) - {f.name for f in fields(DEMConfig)})
        if unknown:
            raise TypeError(f"unknown DEMConfig overrides: {unknown}")
        cfg = replace(cfg, **config_overrides)
        sd = ck["state_dict"]
        has_vision = any(k.startswith("vision.") for k in sd)
        has_lang = any(k.startswith("language.") for k in sd)
        # Weights that are in the file need not be downloaded again.
        build_cfg = replace(cfg, vision_pretrained=cfg.vision_pretrained and not has_vision,
                            language_pretrained=cfg.language_pretrained and not has_lang)
        model = cls(build_cfg)
        model.config = cfg
        model._vision_weights_public = cfg.vision_pretrained and not has_vision
        model._language_weights_public = cfg.language_pretrained and not has_lang
        if has_lang and model.language.tokenizer is None and cfg.language_pretrained:
            from dem.language.neobert import load_tokenizer
            model.language.tokenizer = load_tokenizer(cfg.language_model)
        spec = ck.get("normalizers", {})
        model.action_norm = _build_normalizer(cfg.action_dim, spec.get("action", 1))
        model.proprio_norm = _build_normalizer(cfg.proprio_dim, spec.get("proprio", 1))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        omitted = tuple(ck.get("omitted_prefixes", []))
        bad_missing = [k for k in missing if not k.startswith(omitted)]
        if bad_missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing {bad_missing[:5]}, unexpected {unexpected[:5]}")
        if cfg.freeze_vision:
            _freeze(model.vision)
        return model

    def load_research_checkpoints(self, head: str | None = None, vision: str | None = None,
                                  norm: str | None = None, norm_stats=None) -> None:
        """Load artifacts produced by the research code.

        ``head``: ``state_dict`` of ``MeanFlowTokenHead`` (``mf_final.pt``) or a dict with a
        ``"head"`` entry (``uf_final.pt``). ``vision``: encoder weights in any layout accepted by
        ``VisionEncoder.load_weights`` (a ``uf_*.pt`` file also works).

        Normalization in the research runs had two stages: per-task quantile scaling of
        actions and states to [-1, 1] from the dataset's ``norm_stats.json`` (``q01`` / ``q99``
        quantiles), then z-scoring with the statistics in ``norm.pt``
        (``a_mean / a_std / p_mean / p_std``). Pass ``norm_stats`` (path or parsed dict with
        ``action`` and ``state`` entries) together with ``norm`` to rebuild the full chain;
        ``norm`` alone is only right for runs trained without the quantile stage.
        """
        if head is not None:
            sd = torch.load(head, map_location="cpu", weights_only=True)
            self.head.load_state_dict(sd["head"] if "head" in sd else sd, strict=True)
        if vision is not None:
            self.load_vision_weights(vision)
        if norm is not None or norm_stats is not None:
            stages_a, stages_p = [], []
            if norm_stats is not None:
                st = norm_stats
                if isinstance(norm_stats, (str, os.PathLike)):
                    with open(norm_stats) as fh:
                        st = json.load(fh)
                stages_a.append(Normalizer.from_quantiles(torch.tensor(st["action"]["q01"]),
                                                          torch.tensor(st["action"]["q99"])))
                stages_p.append(Normalizer.from_quantiles(torch.tensor(st["state"]["q01"]),
                                                          torch.tensor(st["state"]["q99"])))
            if norm is not None:
                n = torch.load(norm, map_location="cpu", weights_only=True)
                stages_a.append(Normalizer.from_mean_std(n["a_mean"], n["a_std"]))
                stages_p.append(Normalizer.from_mean_std(n["p_mean"], n["p_std"]))
            self.action_norm = _chain(stages_a).to(self.device)
            self.proprio_norm = _chain(stages_p).to(self.device)
