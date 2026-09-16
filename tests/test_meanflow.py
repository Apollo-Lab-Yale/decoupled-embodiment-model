"""MeanFlow objective: identity at r = t, gradient flow, and two toy problems solved with 1-NFE sampling."""

import pytest
import torch

from dem.head import MeanFlowObjective, MeanFlowTokenHead


def _head(action_dim, chunk, dim=64, depth=2):
    return MeanFlowTokenHead(vis_dim=4, lang_dim=4, proprio_dim=4, action_dim=action_dim, chunk=chunk,
                             dim=dim, depth=depth, heads=4)


def _const_cond(B, device="cpu"):
    return {"vis_tokens": torch.zeros(B, 2, 4, device=device), "lang_tokens": torch.zeros(B, 1, 4, device=device),
            "proprio": torch.zeros(B, 4, device=device)}


def test_r_eq_t_reduces_to_flow_matching():
    obj = MeanFlowObjective(ratio_r_eq_t=1.0)
    head = _head(3, 4)
    x, cond = torch.randn(16, 4, 3), _const_cond(16)
    r = torch.full((16,), 0.3)
    t = r.clone()
    eps = torch.randn_like(x)
    z = (1 - t[:, None, None]) * x + t[:, None, None] * eps
    v = eps - x
    u, dudt = torch.func.jvp(lambda z_, r_, t_: head(z_, r_, t_, cond), (z, r, t),
                             (v, torch.zeros_like(r), torch.ones_like(t)))
    assert torch.allclose(v - (t - r)[:, None, None] * dudt, v)
    loss, aux = obj.loss(head, x, cond)
    assert torch.isfinite(loss) and "raw_mse" in aux


def test_gradients_finite_with_and_without_dispersive():
    for w in (0.0, 0.25):
        head = _head(3, 4)
        obj = MeanFlowObjective(ratio_r_eq_t=0.5, dispersive_weight=w)
        loss, aux = obj.loss(head, torch.randn(8, 4, 3), _const_cond(8))
        loss.backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
        assert ("disp" in aux) == (w > 0)


def test_sample_shapes_and_fixed_noise():
    head = _head(3, 4)
    obj = MeanFlowObjective()
    cond = _const_cond(5)
    a1 = obj.sample(head, cond, chunk=4, action_dim=3, nfe=1)
    a4 = obj.sample(head, cond, chunk=4, action_dim=3, nfe=4)
    assert a1.shape == a4.shape == (5, 4, 3)
    z0 = torch.randn(5, 4, 3)
    assert torch.equal(obj.sample(head, cond, 4, 3, z0=z0), obj.sample(head, cond, 4, 3, z0=z0))


def test_sample_accepts_precomputed_context_and_reuses_it():
    head = _head(3, 4)
    with torch.no_grad():
        for blk in head.blocks:
            blk.gate.fill_(1.0)
        head.out.weight.normal_()
    obj = MeanFlowObjective()
    cond = _const_cond(5)
    cond["vis_tokens"] = torch.randn(5, 2, 4)
    z0 = torch.randn(5, 4, 3)
    a = obj.sample(head, cond, chunk=4, action_dim=3, nfe=3, z0=z0)
    b = obj.sample(head, {"_ctx": head.build_ctx(cond)}, chunk=4, action_dim=3, nfe=3, z0=z0)
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        obj.sample(head, {"nothing": 1}, chunk=4, action_dim=3)


@pytest.mark.slow
def test_toy_unimodal_convergence():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    obj = MeanFlowObjective(ratio_r_eq_t=0.5)
    head = _head(2, 1, dim=128, depth=3).to(dev)
    x_star = torch.tensor([1.5, -0.7], device=dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
    cond = _const_cond(256, dev)
    for _ in range(1200):
        loss, _ = obj.loss(head, x_star[None, None].repeat(256, 1, 1), cond)
        opt.zero_grad()
        loss.backward()
        opt.step()
    samp = obj.sample(head, _const_cond(64, dev), chunk=1, action_dim=2, nfe=1)
    err = (samp.mean(0).squeeze() - x_star).abs().max().item()
    spread = samp.std(0).max().item()
    assert err < 0.15, f"1-NFE sample missed the target: err={err:.3f}"
    assert spread < 0.3, f"samples did not contract onto the point: spread={spread:.3f}"


@pytest.mark.slow
def test_toy_multimodal_no_mean_collapse():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    obj = MeanFlowObjective(ratio_r_eq_t=0.5)
    head = _head(1, 1, dim=128, depth=3).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
    cond = _const_cond(512, dev)
    modes = torch.tensor([-2.0, 2.0], device=dev)
    for _ in range(2000):
        x = modes[torch.randint(0, 2, (512,), device=dev)][:, None, None]
        loss, _ = obj.loss(head, x, cond)
        opt.zero_grad()
        loss.backward()
        opt.step()
    samp = obj.sample(head, cond, chunk=1, action_dim=1, nfe=1).squeeze()
    p_lo = (samp < -1.0).float().mean().item()
    p_hi = (samp > 1.0).float().mean().item()
    p_mid = ((samp > -0.5) & (samp < 0.5)).float().mean().item()
    assert p_lo > 0.25 and p_hi > 0.25, f"both modes should be covered: {p_lo:.2f} / {p_hi:.2f}"
    assert p_mid < 0.2, f"mean collapse: {p_mid:.2f} of samples near 0"
