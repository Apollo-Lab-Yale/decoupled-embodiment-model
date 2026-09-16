"""NeoBERT (Le Breton et al., 2025) in plain PyTorch.

Re-implementation of the encoder released at
https://huggingface.co/chandar-lab/NeoBERT (weights and code under the MIT
license). The released code needs ``trust_remote_code=True`` and imports
``xformers.ops.SwiGLU``. This file has neither dependency and loads the
released ``model.safetensors`` unchanged: parameter names, shapes, and the
per-head interleaved QKV layout are the same, so ``load_state_dict`` is
strict.

Architecture: token embedding, 28 pre-RMSNorm blocks with rotary position
embedding on queries and keys, SwiGLU feed-forward without biases, and a
final RMSNorm. 768 hidden units, 12 heads, 4096 positions. The encoder has
221.7M parameters (23.4M of them in the token embedding table); with the
masked-LM decoder the released checkpoint has 245.1M, which the NeoBERT
paper rounds to 250M. The masked-LM decoder is included so the port can be
checked against fill-mask predictions.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

NEOBERT_REPO = "chandar-lab/NeoBERT"
_CONFIG_FILE = "config.json"
_WEIGHTS_FILE = "model.safetensors"


@dataclass
class NeoBERTConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    norm_eps: float = 1e-5          # value in the released config.json
    vocab_size: int = 30522
    pad_token_id: int = 0
    max_length: int = 4096
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

    @property
    def dim_head(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def ffn_hidden_size(self) -> int:
        """Llama-style SwiGLU width: 2/3 of intermediate_size rounded up to a multiple of 8."""
        h = int(2 * self.intermediate_size / 3)
        return 8 * ((h + 7) // 8)

    @classmethod
    def from_dict(cls, d: dict) -> "NeoBERTConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_pretrained(cls, repo_or_dir: str = NEOBERT_REPO, **hub_kwargs) -> "NeoBERTConfig":
        with open(_resolve_file(repo_or_dir, _CONFIG_FILE, **hub_kwargs)) as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_file(repo_or_dir: str, filename: str, **hub_kwargs) -> str:
    """Path of ``filename`` inside a local directory or a Hub repository."""
    if os.path.isdir(repo_or_dir):
        return os.path.join(repo_or_dir, filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_or_dir, filename, **hub_kwargs)


def rope_cos_sin(dim_head: int, max_length: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine tables of shape ``[max_length, dim_head // 2]``.

    Same frequencies as the released ``precompute_freqs_cis``; stored as
    real tensors instead of complex numbers so every backend can run it.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim_head, 2)[: dim_head // 2].float() / dim_head))
    angles = torch.outer(torch.arange(max_length).float(), freqs)
    return angles.cos(), angles.sin()


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channel pairs of ``x`` ``[B, L, H, Dh]`` by the position angles.

    Equivalent to the complex-number formulation used by the released code
    (``view_as_complex`` over adjacent pairs, multiply by ``e^{i * angle}``).
    Computed in float32 and cast back to the input dtype.
    """
    xf = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = xf[..., 0], xf[..., 1]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    out = torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1)
    return out.flatten(-2).type_as(x)


class SwiGLU(nn.Module):
    """Drop-in for ``xformers.ops.SwiGLU(in, hidden, out, bias=False)``: ``w3(silu(x W1) * (x W2))``."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class NeoBERTBlock(nn.Module):
    def __init__(self, config: NeoBERTConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.dim_head = config.dim_head
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.ffn = SwiGLU(config.hidden_size, config.ffn_hidden_size, config.hidden_size)
        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self._attention(self.attention_norm(x), attn_mask, cos, sin)
        return x + self.ffn(self.ffn_norm(x))

    def _attention(self, x: torch.Tensor, attn_mask: torch.Tensor | None,
                   cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        # The released weights interleave q/k/v per head: view as [B, L, H, 3 * Dh] and split the last axis.
        q, k, v = self.qkv(x).view(B, L, self.heads, 3 * self.dim_head).chunk(3, dim=-1)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attn_mask, dropout_p=0.0,
        )                                                    # [B, H, L, Dh]
        return self.wo(out.transpose(1, 2).reshape(B, L, self.heads * self.dim_head))


class NeoBERT(nn.Module):
    """Encoder. ``forward(input_ids, attention_mask) -> last_hidden_state [B, L, hidden]``."""

    def __init__(self, config: NeoBERTConfig | None = None) -> None:
        super().__init__()
        self.config = config or NeoBERTConfig()
        cfg = self.config
        self.encoder = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.transformer_encoder = nn.ModuleList(NeoBERTBlock(cfg) for _ in range(cfg.num_hidden_layers))
        self.layer_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        cos, sin = rope_cos_sin(cfg.dim_head, cfg.max_length, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Same initialization as the released code (uniform +-0.02); only matters for pretrained=False.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.uniform_(-0.02, 0.02)

    def _apply(self, fn, *args, **kwargs):
        # `.to(bfloat16)` / `.half()` must not round the rotary tables: the released model keeps its
        # complex64 table in full precision and only the projections run in low precision.
        out = super()._apply(fn, *args, **kwargs)
        self.rope_cos = self.rope_cos.float()
        self.rope_sin = self.rope_sin.float()
        return out

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ):
        """
        Args:
            input_ids: ``[B, L]`` token ids.
            attention_mask: ``[B, L]`` with 1 for tokens to attend to and 0 for padding.
            output_hidden_states: also return the output of every block (before the final norm).

        Returns:
            ``last_hidden_state`` ``[B, L, hidden]``, or ``(last_hidden_state, hidden_states)``.
        """
        L = input_ids.shape[1]
        if L > self.config.max_length:
            raise ValueError(f"sequence length {L} exceeds max_length {self.config.max_length}")
        mask = None
        if attention_mask is not None:
            mask = attention_mask.bool()[:, None, None, :]   # [B, 1, 1, L]: True = attend (key axis)
        cos, sin = self.rope_cos[:L], self.rope_sin[:L]
        x = self.encoder(input_ids)
        hidden = []
        for block in self.transformer_encoder:
            x = block(x, mask, cos, sin)
            if output_hidden_states:
                hidden.append(x)
        x = self.layer_norm(x)
        return (x, hidden) if output_hidden_states else x

    @classmethod
    def from_pretrained(cls, repo_or_dir: str = NEOBERT_REPO, dtype: torch.dtype | None = None,
                        **hub_kwargs) -> "NeoBERT":
        """Build the encoder and load the released weights (strict)."""
        model = cls(NeoBERTConfig.from_pretrained(repo_or_dir, **hub_kwargs))
        sd = load_pretrained_state_dict(repo_or_dir, **hub_kwargs)
        model.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}, strict=True)
        return model.to(dtype) if dtype is not None else model


class NeoBERTForMaskedLM(nn.Module):
    """Encoder plus the released masked-LM decoder; used to verify the port with fill-mask."""

    def __init__(self, config: NeoBERTConfig | None = None) -> None:
        super().__init__()
        self.model = NeoBERT(config)
        self.decoder = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, L]`` -> logits ``[B, L, vocab]``."""
        return self.decoder(self.model(input_ids, attention_mask))

    @classmethod
    def from_pretrained(cls, repo_or_dir: str = NEOBERT_REPO, **hub_kwargs) -> "NeoBERTForMaskedLM":
        model = cls(NeoBERTConfig.from_pretrained(repo_or_dir, **hub_kwargs))
        model.load_state_dict(load_pretrained_state_dict(repo_or_dir, **hub_kwargs), strict=True)
        return model


def load_pretrained_state_dict(repo_or_dir: str = NEOBERT_REPO, **hub_kwargs) -> dict[str, torch.Tensor]:
    """The released ``model.safetensors`` as a dict (keys ``model.*`` and ``decoder.*``)."""
    from safetensors.torch import load_file

    return load_file(_resolve_file(repo_or_dir, _WEIGHTS_FILE, **hub_kwargs))


def load_tokenizer(repo_or_dir: str = NEOBERT_REPO, **hub_kwargs):
    """The WordPiece tokenizer shipped with NeoBERT (``google-bert/bert-base-uncased`` vocabulary).

    Loaded through the BERT tokenizer class named in the repository's
    ``tokenizer_config.json``. ``AutoTokenizer`` is avoided on purpose: it
    reads ``config.json`` first, whose ``auto_map`` entry makes transformers
    ask interactively whether to trust remote code.
    """
    from transformers import BertTokenizerFast

    return BertTokenizerFast.from_pretrained(repo_or_dir, **hub_kwargs)
