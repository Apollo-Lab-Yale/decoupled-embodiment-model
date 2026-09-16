import os
import tempfile

import pytest
import torch
import torch.nn as nn

from dem import ChainedNormalizer, DEM, DEMConfig, LanguageEncoder, Normalizer, VisionEncoder
from dem.language import NeoBERTConfig

TINY_LM = NeoBERTConfig(hidden_size=24, num_hidden_layers=1, num_attention_heads=4, intermediate_size=48,
                        vocab_size=50, max_length=32)


def _second_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


class StubVision(VisionEncoder):
    """Tiny stand-in with the encoder interface: 4x4 patches of a 16-px image -> 16 tokens."""

    def __init__(self, dim=12, image_size=16):
        super().__init__()
        self.model = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.embed_dim, self.image_size, self.tokens_per_image = dim, image_size, (image_size // 4) ** 2

    def forward(self, images):
        return self.model(self.normalize(images)).flatten(2).transpose(1, 2)


class _Tok:
    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        n = [min(len(t) % 5 + 3, max_length) for t in texts]
        L = max(n)
        ids = torch.zeros(len(texts), L, dtype=torch.long)
        mask = torch.zeros(len(texts), L, dtype=torch.long)
        for i, k in enumerate(n):
            ids[i, :k] = torch.randint(1, 50, (k,))
            mask[i, :k] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _policy(**cfg_kw):
    cfg = DEMConfig(head_size="small", proprio_dim=5, action_dim=4, chunk=6, image_size=16, num_cameras=2,
                    max_language_tokens=8, dispersive_weight=0.1, **cfg_kw)
    lang = LanguageEncoder(backend="neobert", pretrained=False, neobert_config=TINY_LM, max_tokens=8,
                           freeze=cfg.freeze_language, tokenizer=_Tok())
    return DEM(cfg, vision=StubVision(), language=lang)


def test_act_and_loss_shapes():
    pol = _policy()
    imgs = torch.rand(3, 2, 3, 16, 16)
    prop = torch.randn(3, 5)
    pol.set_instruction("open the drawer")
    a = pol.act(imgs, prop)                                   # cached instruction broadcast over the batch
    assert a.shape == (3, 6, 4)
    a2 = pol.act(imgs, prop, texts=["a", "b", "c"], nfe=2)
    assert a2.shape == (3, 6, 4)
    loss, aux = pol.loss(imgs, prop, torch.randn(3, 6, 4), texts=["a", "a", "b"])
    assert torch.isfinite(loss) and "disp" in aux


def test_context_token_count_and_camera_concat():
    pol = _policy()
    vt = pol.encode_vision(torch.rand(2, 2, 3, 16, 16))
    assert vt.shape == (2, 2 * 16, 12)
    tok, mask = pol.encode_language(["x", "y"])
    ctx, m = pol.head.build_ctx(pol.build_cond(vt, tok, mask, torch.randn(2, 5)))
    assert ctx.shape == (2, 32 + 8 + 1, pol.head.dim) and m.shape == (2, 41)


def test_language_cache_reused():
    pol = _policy()
    tok1, _ = pol.encode_language("open the drawer")
    calls = {"n": 0}
    orig = pol.language.forward

    def counting(texts):
        calls["n"] += 1
        return orig(texts)

    pol.language.forward = counting
    tok2, _ = pol.encode_language("open the drawer")
    assert calls["n"] == 0 and torch.equal(tok1, tok2)
    pol.encode_language(["open the drawer", "close the fridge"])
    assert calls["n"] == 1                                     # only the new string is encoded


def test_pre_encoded_instruction_api():
    pol = _policy()
    tok, mask = pol.language.encode_ids(torch.randint(1, 50, (1, 4)), torch.ones(1, 4))
    pol.set_instruction_tokens(tok[0], mask[0])
    assert pol.act(torch.rand(2, 2, 3, 16, 16), torch.zeros(2, 5)).shape == (2, 6, 4)
    pol.cache_instruction("custom", tok[0], mask[0])
    t2, m2 = pol.encode_language(["custom"])
    assert torch.equal(t2[0], tok[0]) and torch.equal(m2[0], mask[0])


def test_freeze_flags_control_gradients():
    pol = _policy(freeze_vision=False)
    loss, _ = pol.loss(torch.rand(2, 2, 3, 16, 16), torch.randn(2, 5), torch.randn(2, 6, 4), texts=["a", "b"])
    loss.backward()
    assert all(p.grad is not None for p in pol.vision.parameters())
    assert all(p.grad is None for p in pol.language.parameters())
    assert len(pol.parameter_groups()) == 2

    frozen = _policy(freeze_vision=True)
    assert not frozen.vision.training, "a frozen encoder is in eval mode right after construction"
    loss, _ = frozen.loss(torch.rand(2, 2, 3, 16, 16), torch.randn(2, 5), torch.randn(2, 6, 4), texts=["a", "b"])
    loss.backward()
    assert all(p.grad is None for p in frozen.vision.parameters())
    assert len(frozen.parameter_groups()) == 1
    frozen.train()
    assert not frozen.vision.training and frozen.head.training


def test_trainable_language_receives_gradients_and_is_not_cached():
    pol = _policy(freeze_language=False)
    assert len(pol.parameter_groups()) == 3
    with torch.no_grad():   # open the zero-initialized gates and output layer so gradients reach the context
        for blk in pol.head.blocks:
            blk.gate.fill_(1.0)
        pol.head.out.weight.normal_()
    loss, _ = pol.loss(torch.rand(2, 2, 3, 16, 16), torch.randn(2, 5), torch.randn(2, 6, 4), texts=["a", "b"])
    loss.backward()
    grads = [p.grad for p in pol.language.parameters() if p.grad is not None]
    assert grads and sum(g.abs().sum() for g in grads) > 0
    assert not pol._lang_cache, "no caching while the language encoder trains"


@pytest.mark.skipif(_second_device() is None, reason="needs a second device (cuda or mps)")
def test_language_cache_follows_device_moves():
    dev = _second_device()
    pol = _policy()
    pol.set_instruction("go")
    pol.encode_language("stay")
    pol.to(dev)
    tok, mask = pol.encode_language(["stay", "new text"])      # old cache entry + fresh one
    assert tok.device.type == dev and mask.device.type == dev
    a = pol.act(torch.rand(2, 2, 3, 16, 16), torch.zeros(2, 5))  # cached instruction migrates too
    assert a.device.type == dev and pol._instruction[0].device.type == dev


def test_build_cond_casts_to_head_dtype():
    pol = _policy()
    pol.set_instruction("go")
    vt = pol.encode_vision(torch.rand(1, 2, 3, 16, 16)).to(torch.bfloat16)   # e.g. from autocast or a cache
    assert pol.act(None, torch.zeros(1, 5), vis_tokens=vt).shape == (1, 6, 4)
    loss, _ = pol.loss(None, torch.zeros(1, 5), torch.zeros(1, 6, 4), vis_tokens=vt)
    assert torch.isfinite(loss)


def test_normalizers_are_applied():
    pol = _policy()
    pol.action_norm = Normalizer.from_mean_std(torch.full((4,), 10.0), torch.full((4,), 2.0))
    pol.set_instruction("go")
    z0 = torch.randn(1, 6, 4)
    a = pol.act(torch.rand(1, 2, 3, 16, 16), torch.zeros(1, 5), z0=z0)
    # Zero-initialized head: u = 0, so the normalized sample equals z0 and the output is z0 * std + mean.
    assert torch.allclose(a, z0 * 2.0 + 10.0, atol=1e-5)


def test_config_from_dict_rejects_unknown_keys():
    with pytest.raises(TypeError):
        DEMConfig.from_dict({"freeze_visoin": True})
    assert DEMConfig.from_dict({"freeze_visoin": True, "chunk": 8}, ignore_unknown=True).chunk == 8


def test_save_state_dict_and_omission_rules():
    pol = _policy()
    pol.action_norm.fit_mean_std(torch.randn(50, 6, 4))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dem.pt")
        pol.save(path, include_frozen=True)
        ck = torch.load(path, weights_only=True)
        assert ck["config"]["action_dim"] == 4 and ck["format"] == "dem-v1"
        assert ck["normalizers"] == {"action": 1, "proprio": 1}
        other = _policy()
        other.load_state_dict(ck["state_dict"], strict=True)
        for (k, v), (k2, v2) in zip(pol.state_dict().items(), other.state_dict().items()):
            assert k == k2 and torch.equal(v, v2)
        # Injected towers are never assumed to be the public weights: nothing is omitted.
        pol.save(path)
        assert torch.load(path, weights_only=True)["omitted_prefixes"] == []
        # A frozen tower built from the public checkpoint may be omitted ...
        pol._language_weights_public = True
        pol.save(path)
        ck = torch.load(path, weights_only=True)
        assert ck["omitted_prefixes"] == ["language."]
        assert not any(k.startswith("language.") for k in ck["state_dict"])
        # ... unless its weights were replaced.
        frozen = _policy(freeze_vision=True)
        frozen._vision_weights_public = True
        frozen.load_vision_weights({"tower_timm": _policy().vision.export_weights()})
        frozen.save(path)
        assert "vision." not in torch.load(path, weights_only=True)["omitted_prefixes"]


def test_load_research_checkpoints_two_stage_normalization():
    pol = _policy()
    q01 = torch.tensor([-1.0, 0.0, 2.0, 5.0])
    q99 = torch.tensor([1.0, 4.0, 2.0, 5.0])                  # last two dims are constant
    stats = {"action": {"q01": q01.tolist(), "q99": q99.tolist()},
             "state": {"q01": [0.0] * 5, "q99": [2.0] * 5}}
    with tempfile.TemporaryDirectory() as d:
        head_p, vis_p, norm_p = (os.path.join(d, n) for n in ("mf_final.pt", "tower.pt", "norm.pt"))
        ref = _policy()
        torch.save(ref.head.state_dict(), head_p)
        torch.save({"tower_timm": ref.vision.export_weights()}, vis_p)
        torch.save({"a_mean": torch.ones(4), "a_std": torch.full((4,), 3.0),
                    "p_mean": torch.zeros(5), "p_std": torch.ones(5)}, norm_p)
        pol.load_research_checkpoints(head=head_p, vision=vis_p, norm=norm_p, norm_stats=stats)
    assert torch.equal(pol.head.out.bias, ref.head.out.bias)
    assert torch.equal(pol.vision.model.weight, ref.vision.model.weight)
    assert isinstance(pol.action_norm, ChainedNormalizer) and len(pol.action_norm.stages) == 2
    raw = torch.tensor([[0.5, 1.0, 2.0, 5.0]])
    q = torch.tensor([[0.5, -0.5, -1.0, -1.0]])                # quantile stage (constants map to -1)
    assert torch.allclose(pol.action_norm.normalize(raw), (q - 1.0) / 3.0, atol=1e-6)
    back = pol.action_norm.unnormalize(torch.randn(3, 4))
    assert torch.allclose(back[:, 2], torch.full((3,), 2.0)) and torch.allclose(back[:, 3], torch.full((3,), 5.0))
    # z-score only: single stage, unchanged behaviour
    pol2 = _policy()
    with tempfile.TemporaryDirectory() as d:
        norm_p = os.path.join(d, "norm.pt")
        torch.save({"a_mean": torch.ones(4), "a_std": torch.full((4,), 3.0),
                    "p_mean": torch.zeros(5), "p_std": torch.ones(5)}, norm_p)
        pol2.load_research_checkpoints(norm=norm_p)
    assert isinstance(pol2.action_norm, Normalizer) and torch.allclose(pol2.action_norm.scale, torch.full((4,), 3.0))


def test_chained_normalizer_spec_is_saved_and_rebuilt():
    from dem.policy import _build_normalizer

    pol = _policy()
    pol.action_norm = ChainedNormalizer(Normalizer.from_quantiles(torch.zeros(4), torch.full((4,), 2.0)),
                                        Normalizer.from_mean_std(torch.ones(4), torch.full((4,), 0.5)))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dem.pt")
        pol.save(path, include_frozen=True)
        ck = torch.load(path, weights_only=True)
    assert ck["normalizers"] == {"action": 2, "proprio": 1}
    other = _policy()
    other.action_norm = _build_normalizer(4, ck["normalizers"]["action"])   # what DEM.load does before loading
    missing, unexpected = other.load_state_dict(ck["state_dict"], strict=False)
    assert not missing and not unexpected
    x = torch.randn(2, 6, 4)
    assert torch.allclose(other.action_norm.normalize(x), pol.action_norm.normalize(x))


@pytest.mark.network
@pytest.mark.slow
def test_dem_load_roundtrip_through_public_api():
    """save -> DEM.load end to end: random-init ConvNeXt, a tiny Hub BERT as the `hf` language backend."""
    cfg = DEMConfig(vision_pretrained=False, freeze_vision=True, head_size="small", proprio_dim=5, action_dim=4,
                    chunk=6, language_backend="hf", language_model="hf-internal-testing/tiny-random-bert",
                    max_language_tokens=8)
    pol = DEM(cfg)
    pol.action_norm = ChainedNormalizer(Normalizer.from_quantiles(torch.zeros(4), torch.full((4,), 2.0)),
                                        Normalizer.from_mean_std(torch.ones(4), torch.full((4,), 0.5)))
    pol.set_instruction("open the drawer")
    imgs, prop, z0 = torch.rand(1, 2, 3, 256, 256), torch.zeros(1, 5), torch.randn(1, 6, 4)
    a_ref = pol.act(imgs, prop, z0=z0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dem.pt")
        pol.save(path, include_frozen=True)                    # tower weights travel with the file
        loaded = DEM.load(path)
        assert not any(k.startswith("language.") for k in torch.load(path, weights_only=True)["omitted_prefixes"])
        with pytest.raises(TypeError):
            DEM.load(path, freeze_visoin=True)
    assert isinstance(loaded.action_norm, ChainedNormalizer)
    assert not loaded.vision.training and all(not p.requires_grad for p in loaded.vision.parameters())
    loaded.set_instruction("open the drawer")
    assert torch.allclose(loaded.act(imgs, prop, z0=z0), a_ref, atol=1e-5)


@pytest.mark.slow
def test_full_size_random_init_forward():
    """Real architectures at random initialization (no downloads): shapes and a training step on CPU."""
    cfg = DEMConfig(vision_pretrained=False, language_pretrained=False, num_cameras=2, freeze_vision=True)
    lang = LanguageEncoder(backend="neobert", pretrained=False, neobert_config=NeoBERTConfig(num_hidden_layers=2),
                           max_tokens=32, tokenizer=_Tok())
    pol = DEM(cfg, language=lang).eval()
    counts = pol.count_parameters()
    assert counts["vision"] == 87_566_464                     # DINOv3 ConvNeXt-B, measured on the research code
    assert counts["head"] == 107_030_796 + (768 - 384) * 768  # new100m head with 768-d language tokens
    imgs = torch.rand(1, 2, 3, 256, 256)
    pol.set_instruction("open the left drawer")
    with torch.no_grad():
        vt = pol.encode_vision(imgs)
        assert vt.shape == (1, 512, 512)
        tok, mask = pol._instruction
        ctx, _ = pol.head.build_ctx(pol.build_cond(vt, tok, mask, torch.zeros(1, 16)))
        assert ctx.shape == (1, 545, 768)
    a = pol.act(imgs, torch.zeros(1, 16))
    assert a.shape == (1, 16, 12)
    loss, _ = pol.loss(None, torch.zeros(1, 16), torch.zeros(1, 16, 12), vis_tokens=vt)
    loss.backward()
    assert torch.isfinite(loss)
