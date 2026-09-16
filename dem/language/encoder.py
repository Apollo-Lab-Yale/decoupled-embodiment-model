"""Language encoder producing token-level instruction features.

The instruction is tokenized, padded or truncated to ``max_tokens`` (32 in
the paper, special tokens included), and the final-layer hidden state of
every token is kept together with its attention mask. Padded positions are
zeroed. The head reads the tokens through masked cross-attention, so a
pooled sentence vector is never formed.

Two backends share the interface:

* ``"neobert"``: the in-package NeoBERT port (default, 768-d).
* ``"hf"``: any Hugging Face encoder loaded with ``AutoModel``; for an
  encoder-decoder model only the encoder is used. This is the path for
  the language ablations (mmBERT, BERT, T5) and for research checkpoints
  trained with mmBERT-small (384-d).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from dem.language.neobert import NEOBERT_REPO, NeoBERT, NeoBERTConfig, load_tokenizer


class LanguageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = NEOBERT_REPO,
        backend: str = "neobert",
        max_tokens: int = 32,
        pretrained: bool = True,
        freeze: bool = True,
        neobert_config: NeoBERTConfig | None = None,
        tokenizer=None,
    ) -> None:
        """
        Args:
            model_name: Hub id or local directory of the language model.
            backend: ``"neobert"`` or ``"hf"``.
            max_tokens: fixed token count of the output (pad / truncate).
            pretrained: load weights; ``False`` builds a randomly initialized model.
            freeze: disable gradients and keep the model in eval mode.
            neobert_config: architecture override for ``pretrained=False`` (tests, tiny models).
            tokenizer: optional tokenizer object; loaded from ``model_name`` when ``None``.
        """
        super().__init__()
        if backend not in ("neobert", "hf"):
            raise ValueError(f"backend must be 'neobert' or 'hf', got {backend!r}")
        self.model_name = model_name
        self.backend = backend
        self.max_tokens = max_tokens
        self.frozen = freeze
        self.tokenizer = tokenizer

        if backend == "neobert":
            if pretrained:
                self.model = NeoBERT.from_pretrained(model_name)
            else:
                cfg = neobert_config
                if cfg is None:
                    try:
                        cfg = NeoBERTConfig.from_pretrained(model_name)
                    except Exception:  # offline: fall back to the released architecture
                        cfg = NeoBERTConfig()
                self.model = NeoBERT(cfg)
            self.embed_dim = self.model.config.hidden_size
            if self.tokenizer is None and pretrained:
                self.tokenizer = load_tokenizer(model_name)
        else:
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            if pretrained:
                lm = AutoModel.from_pretrained(model_name, trust_remote_code=False)
            else:
                lm = AutoModel.from_config(AutoConfig.from_pretrained(model_name, trust_remote_code=False))
            if getattr(lm.config, "is_encoder_decoder", False):
                lm = lm.encoder
            self.model = lm
            self.embed_dim = int(lm.config.hidden_size if hasattr(lm.config, "hidden_size") else lm.config.d_model)
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> "LanguageEncoder":
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Tokenize with truncation to ``max_tokens`` and padding to the longest text."""
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer available; pass `tokenizer=` or use pretrained=True")
        enc = self.tokenizer(list(texts), return_tensors="pt", padding=True,
                             truncation=True, max_length=self.max_tokens)
        return {"input_ids": enc["input_ids"].to(self.device),
                "attention_mask": enc["attention_mask"].to(self.device)}

    def _hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.backend == "neobert":
            return self.model(input_ids, attention_mask)
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def encode_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-tokenized ``[B, L <= max_tokens]`` -> (``tokens [B, max_tokens, D]``, ``mask [B, max_tokens]``)."""
        if input_ids.shape[1] > self.max_tokens:
            raise ValueError(f"got {input_ids.shape[1]} tokens, max_tokens is {self.max_tokens}")
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            h = self._hidden(input_ids, attention_mask)                # [B, L, D]
        B, L, D = h.shape
        valid = attention_mask.bool()
        tokens = h.new_zeros(B, self.max_tokens, D)
        tokens[:, :L] = h * valid[..., None].to(h.dtype)               # zero the padded positions
        mask = torch.zeros(B, self.max_tokens, dtype=torch.bool, device=h.device)
        mask[:, :L] = valid
        return tokens, mask

    def forward(self, texts: Sequence[str] | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Instruction strings -> (``tokens [B, max_tokens, D]``, ``mask [B, max_tokens]``)."""
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenize(texts)
        return self.encode_ids(enc["input_ids"], enc["attention_mask"])
