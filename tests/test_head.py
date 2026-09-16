import torch

from dem.head import MeanFlowTokenHead


def _cond(B=2, nv=6, dv=8, nl=4, dl=5, p=3, valid=(4, 2)):
    mask = torch.zeros(B, nl, dtype=torch.bool)
    for i, n in enumerate(valid):
        mask[i, :n] = True
    return {"vis_tokens": torch.randn(B, nv, dv), "lang_tokens": torch.randn(B, nl, dl),
            "lang_mask": mask, "proprio": torch.randn(B, p)}


def _tiny(**kw):
    args = dict(vis_dim=8, lang_dim=5, proprio_dim=3, action_dim=4, chunk=6, dim=32, depth=2, heads=4)
    args.update(kw)
    return MeanFlowTokenHead(**args)


def test_output_shape_and_zero_init():
    head = _tiny()
    cond = _cond()
    z = torch.randn(2, 6, 4)
    u = head(z, torch.zeros(2), torch.ones(2), cond)
    assert u.shape == (2, 6, 4)
    assert torch.all(u == 0), "output layer is zero-initialized"


def test_context_layout():
    head = _tiny()
    ctx, mask = head.build_ctx(_cond())
    assert ctx.shape == (2, 6 + 4 + 1, 32)
    assert mask.shape == (2, 11)
    assert mask[0].tolist() == [True] * 6 + [True] * 4 + [True]
    assert mask[1].tolist() == [True] * 6 + [True, True, False, False] + [True]


def test_masked_language_tokens_do_not_change_output():
    head = _tiny()
    with torch.no_grad():  # open the gates and the output layer so the context matters
        for blk in head.blocks:
            blk.gate.fill_(1.0)
        head.out.weight.normal_()
    cond = _cond(valid=(2, 2))
    z = torch.randn(2, 6, 4)
    r, t = torch.zeros(2), torch.ones(2)
    u1 = head(z, r, t, cond)
    cond2 = dict(cond)
    cond2["lang_tokens"] = cond["lang_tokens"].clone()
    cond2["lang_tokens"][:, 2:] = torch.randn(2, 2, 5) * 10  # only the masked positions change
    u2 = head(z, r, t, cond2)
    assert torch.allclose(u1, u2, atol=1e-6)
    cond3 = dict(cond)
    cond3["lang_tokens"] = cond["lang_tokens"].clone()
    cond3["lang_tokens"][:, :2] += 1.0  # valid positions change
    assert not torch.allclose(u1, head(z, r, t, cond3), atol=1e-6)


def test_precomputed_ctx_matches():
    head = _tiny()
    with torch.no_grad():
        for blk in head.blocks:
            blk.gate.fill_(1.0)
        head.out.weight.normal_()
    cond = _cond()
    z = torch.randn(2, 6, 4)
    r, t = torch.rand(2), torch.rand(2)
    ref = head(z, r, t, cond)
    out = head(z, r, t, {"_ctx": head.build_ctx(cond)})
    assert torch.equal(ref, out)


def test_return_hidden():
    head = _tiny(depth=4)
    u, mid = head(torch.randn(2, 6, 4), torch.zeros(2), torch.ones(2), _cond(), return_hidden=True)
    assert u.shape == (2, 6, 4) and mid.shape == (2, 6, 32)


def test_paper_configuration_parameter_counts():
    # Count measured on the research code: new100m head with mmBERT-small (384-d) tokens = 107,030,796.
    h384 = MeanFlowTokenHead.from_size("new100m", vis_dim=512, lang_dim=384, proprio_dim=16, action_dim=12, chunk=16)
    assert h384.num_parameters() == 107_030_796
    h768 = MeanFlowTokenHead.from_size("new100m", vis_dim=512, lang_dim=768, proprio_dim=16, action_dim=12, chunk=16)
    # Only lang_proj changes: (768 - 384) * 768 extra weights.
    assert h768.num_parameters() == 107_030_796 + (768 - 384) * 768
    assert h768.dim == 768 and h768.depth == 8 and h768.heads == 12
