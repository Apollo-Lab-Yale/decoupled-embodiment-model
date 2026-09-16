import pytest
import torch

from dem.language import LanguageEncoder, NeoBERT, NeoBERTConfig, NeoBERTForMaskedLM
from dem.language.neobert import apply_rotary_emb, rope_cos_sin

TINY = NeoBERTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
                     vocab_size=100, max_length=64)


def test_config_derived_sizes():
    cfg = NeoBERTConfig()
    assert cfg.dim_head == 64 and cfg.ffn_hidden_size == 2048   # 3072 -> 2/3 -> rounded to 8
    assert TINY.ffn_hidden_size == 48


def test_parameter_names_match_released_checkpoint():
    m = NeoBERTForMaskedLM(NeoBERTConfig())
    sd = m.state_dict()
    assert sd["model.encoder.weight"].shape == (30522, 768)
    assert sd["model.transformer_encoder.0.qkv.weight"].shape == (2304, 768)
    assert sd["model.transformer_encoder.0.wo.weight"].shape == (768, 768)
    assert sd["model.transformer_encoder.0.ffn.w12.weight"].shape == (4096, 768)
    assert sd["model.transformer_encoder.0.ffn.w3.weight"].shape == (768, 2048)
    assert sd["model.transformer_encoder.0.attention_norm.weight"].shape == (768,)
    assert sd["model.transformer_encoder.0.ffn_norm.weight"].shape == (768,)
    assert sd["model.layer_norm.weight"].shape == (768,)
    assert sd["decoder.weight"].shape == (30522, 768) and sd["decoder.bias"].shape == (30522,)
    assert len(sd) == 1 + 1 + 28 * 6 + 2          # embedding, final norm, 6 tensors per block, decoder w/b
    assert sum(v.numel() for v in sd.values()) == 245_136_954   # sum over the released safetensors


def test_rotary_matches_complex_formulation():
    cos, sin = rope_cos_sin(8, 16)
    x = torch.randn(2, 16, 3, 8)
    out = apply_rotary_emb(x, cos, sin)
    # Reference: the released complex-number implementation.
    freqs = 1.0 / (10000.0 ** (torch.arange(0, 8, 2)[:4].float() / 8))
    fc = torch.polar(torch.ones(16, 4), torch.outer(torch.arange(16).float(), freqs))
    xc = torch.view_as_complex(x.float().reshape(2, 16, 3, 4, 2))
    ref = torch.view_as_real(xc * fc[None, :, None, :]).flatten(3)
    assert torch.allclose(out, ref, atol=1e-6)


def test_encoder_shapes_and_padding_invariance():
    m = NeoBERT(TINY).eval()
    ids = torch.randint(1, 100, (2, 7))
    with torch.no_grad():
        full = m(ids, torch.ones(2, 7))
        assert full.shape == (2, 7, 32)
        # Pad the batch: real-token outputs must not change when padding is masked out.
        ids_p = torch.cat([ids, torch.zeros(2, 5, dtype=torch.long)], dim=1)
        mask_p = torch.cat([torch.ones(2, 7), torch.zeros(2, 5)], dim=1)
        padded = m(ids_p, mask_p)
    assert torch.allclose(full, padded[:, :7], atol=1e-5)
    # And it must change when the padding is attended to.
    with torch.no_grad():
        unmasked = m(ids_p, None)
    assert not torch.allclose(full, unmasked[:, :7], atol=1e-3)


def test_rope_tables_stay_float32_in_low_precision():
    m = NeoBERT(TINY).to(torch.bfloat16)
    assert m.rope_cos.dtype == torch.float32 and m.rope_sin.dtype == torch.float32
    assert m.encoder.weight.dtype == torch.bfloat16
    out = m(torch.randint(1, 100, (1, 5)), torch.ones(1, 5))
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()


def test_hidden_states_and_mlm_head():
    m = NeoBERTForMaskedLM(TINY).eval()
    ids = torch.randint(1, 100, (1, 5))
    with torch.no_grad():
        logits = m(ids, torch.ones(1, 5))
        _, hidden = m.model(ids, torch.ones(1, 5), output_hidden_states=True)
    assert logits.shape == (1, 5, 100) and len(hidden) == 2


def test_language_encoder_encode_ids_pads_to_max_tokens():
    enc = LanguageEncoder(backend="neobert", pretrained=False, neobert_config=TINY, max_tokens=12, freeze=True)
    assert enc.embed_dim == 32
    ids = torch.randint(1, 100, (2, 5))
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    tokens, m = enc.encode_ids(ids, mask)
    assert tokens.shape == (2, 12, 32) and m.shape == (2, 12)
    assert m[0].tolist() == [True] * 5 + [False] * 7
    assert m[1].tolist() == [True] * 3 + [False] * 9
    assert torch.all(tokens[1, 3:] == 0) and torch.all(tokens[0, 5:] == 0)
    assert not tokens.requires_grad and all(not p.requires_grad for p in enc.parameters())
    enc.train()
    assert not enc.model.training, "a frozen encoder stays in eval mode"


class _FakeTokenizer:
    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        lens = [min(len(t.split()) + 2, max_length) for t in texts]
        L = max(lens)
        ids = torch.zeros(len(texts), L, dtype=torch.long)
        mask = torch.zeros(len(texts), L, dtype=torch.long)
        for i, n in enumerate(lens):
            ids[i, :n] = torch.arange(1, n + 1)
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_language_encoder_forward_with_tokenizer():
    enc = LanguageEncoder(backend="neobert", pretrained=False, neobert_config=TINY, max_tokens=8,
                          tokenizer=_FakeTokenizer())
    tokens, mask = enc(["open the drawer", "pick the mug from the counter and place it in the sink"])
    assert tokens.shape == (2, 8, 32)
    assert mask.sum(1).tolist() == [5, 8]   # second text is truncated to max_tokens


@pytest.mark.network
def test_released_weights_load_and_fill_mask():
    from dem.language.neobert import load_tokenizer

    tok = load_tokenizer()                        # BERT tokenizer class, no remote code, no prompt
    assert type(tok).__name__.startswith("BertTokenizer") and tok.is_fast
    mlm = NeoBERTForMaskedLM.from_pretrained().eval()
    assert sum(p.numel() for p in mlm.parameters()) == 245_136_954
    enc = tok(["The capital of France is [MASK].", "Paris is the capital of [MASK]."],
              return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = mlm(enc["input_ids"], enc["attention_mask"])
    preds = []
    for i in range(2):
        pos = (enc["input_ids"][i] == tok.mask_token_id).nonzero().item()
        preds.append(tok.decode(logits[i, pos].argmax()).strip())
    assert preds == ["paris", "france"], preds

    lang = LanguageEncoder()                      # default: pretrained NeoBERT, frozen, 32 tokens
    tokens, mask = lang("open the left drawer")
    assert tokens.shape == (1, 32, 768) and mask.sum().item() == 6   # [CLS] open the left drawer [SEP]
    assert torch.all(tokens[0, 6:] == 0)
