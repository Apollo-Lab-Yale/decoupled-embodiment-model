"""MeanFlow training objective and one-step sampler.

MeanFlow (Geng et al., 2025) learns the average velocity over an interval,

    u(z_t, r, t) = 1 / (t - r) * integral_r^t v(z_tau, tau) dtau,

so that the displacement across the interval is a single product,
``z_r = z_t - (t - r) u(z_t, r, t)``. Differentiating in ``t`` gives the
MeanFlow identity ``u = v - (t - r) du/dt``, which the network is trained
to satisfy. Along the linear interpolation ``z_t = (1 - t) x + t eps`` the
instantaneous velocity is ``v = eps - x`` in closed form, and the total
derivative ``du/dt = v . dz u + dt u`` is one Jacobian-vector product.

Conventions: ``t = 0`` is the data end and ``t = 1`` the noise end. The
one-step sample is ``x_hat = eps - u(eps, r=0, t=1)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanFlowObjective:
    """Loss and sampler; independent of the network as long as it has the signature
    ``model(z, r, t, cond, return_hidden=False)``.

    Time sampling follows the MeanFlow recipe: ``(t, r)`` are drawn independently
    from a logit-normal distribution and sorted (``t = max``, ``r = min``); with
    probability ``ratio_r_eq_t`` they are set equal, in which case the target
    reduces to the flow-matching velocity and no JVP term is present. The loss
    uses the adaptive weight ``w = (||delta||^2 + c)^(-p)`` with ``p = 0.5``,
    ``c = 1e-3``, applied with a stop-gradient.

    ``dispersive_weight > 0`` adds the dispersive regularizer (InfoNCE-L2 form,
    Wang & He 2025) on the l2-normalized mid-network hidden state, which
    discourages collapse of the action-token representation across a batch.
    """

    def __init__(self, ratio_r_eq_t: float = 0.5, mu: float = -0.4, sigma: float = 1.0,
                 adaptive_p: float = 0.5, adaptive_c: float = 1e-3,
                 dispersive_weight: float = 0.0, dispersive_tau: float = 0.5) -> None:
        self.ratio = ratio_r_eq_t
        self.mu, self.sigma = mu, sigma
        self.p, self.c = adaptive_p, adaptive_c
        self.disp_w, self.disp_tau = dispersive_weight, dispersive_tau

    def sample_rt(self, B: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        n = torch.randn(B, 2, device=device) * self.sigma + self.mu
        rt = torch.sigmoid(n)
        t = rt.max(dim=1).values
        r = rt.min(dim=1).values
        eq = torch.rand(B, device=device) < self.ratio
        r = torch.where(eq, t, r)
        return r, t

    def loss(self, model: nn.Module, x: torch.Tensor, cond) -> tuple[torch.Tensor, dict]:
        """
        Args:
            model: the velocity network.
            x: normalized action chunk ``[B, H, A]``.
            cond: conditioning passed through to the model.

        Returns:
            ``(loss, aux)`` with ``aux["raw_mse"]`` (unweighted squared error) and,
            when enabled, ``aux["disp"]``.
        """
        B = x.shape[0]
        dev = x.device
        r, t = self.sample_rt(B, dev)
        eps = torch.randn_like(x)
        t_ = t[:, None, None]
        z = (1 - t_) * x + t_ * eps
        v = eps - x

        def u_fn(z_in, r_in, t_in):
            return model(z_in, r_in, t_in, cond)

        # du/dt along the trajectory: tangent (v, 0, 1) for (z, r, t).
        u, dudt = torch.func.jvp(u_fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
        u_tgt = (v - (t - r)[:, None, None] * dudt).detach()
        sq = (u - u_tgt).pow(2).mean(dim=(1, 2))
        w = (sq.detach() + self.c).pow(-self.p)
        loss = (w * sq).mean() / w.detach().mean().clamp_min(1e-8)
        aux = {"raw_mse": sq.mean().item()}
        if self.disp_w > 0:
            _, mid = model(z, r, t, cond, return_hidden=True)
            f = F.normalize(mid.mean(dim=1), dim=-1)
            d2 = torch.cdist(f, f).pow(2)
            disp = torch.log(torch.exp(-d2 / self.disp_tau).mean())
            loss = loss + self.disp_w * disp
            aux["disp"] = disp.item()
        return loss, aux

    @torch.no_grad()
    def sample(self, model: nn.Module, cond, chunk: int, action_dim: int,
               nfe: int = 1, z0: torch.Tensor | None = None) -> torch.Tensor:
        """Generate a normalized action chunk ``[B, H, A]``.

        ``nfe = 1`` is the one-step sample ``eps - u(eps, 0, 1)``. For ``nfe > 1``
        the interval ``[1, 0]`` is split into equal segments and the displacement
        identity is applied per segment. ``z0`` fixes the initial noise
        (``None`` draws fresh noise each call).
        """
        B, dev = _batch_info(cond)
        if nfe > 1 and isinstance(cond, dict) and "_ctx" not in cond and hasattr(model, "build_ctx"):
            cond = {**cond, "_ctx": model.build_ctx(cond)}       # build the context once, not per step
        z = torch.randn(B, chunk, action_dim, device=dev) if z0 is None else z0.to(dev)
        ts = torch.linspace(1, 0, nfe + 1, device=dev)
        for i in range(nfe):
            t_, r_ = ts[i], ts[i + 1]
            u = model(z, r_.expand(B), t_.expand(B), cond)
            z = z - (t_ - r_) * u
        return z


def _batch_info(cond) -> tuple[int, torch.device]:
    """Batch size and device of a conditioning dict (or tensor)."""
    if torch.is_tensor(cond):
        return cond.shape[0], cond.device
    if isinstance(cond, dict):
        if "_ctx" in cond:
            ref = cond["_ctx"][0]
        elif "proprio" in cond:
            ref = cond["proprio"]
        else:
            ref = next((v for v in cond.values() if torch.is_tensor(v)), None)
        if ref is not None:
            return ref.shape[0], ref.device
    raise ValueError("cannot infer batch size and device from cond; pass a dict with tensors or '_ctx'")
