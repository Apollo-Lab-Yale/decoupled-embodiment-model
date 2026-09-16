import numpy as np
import torch

from dem.normalization import ChainedNormalizer, Normalizer


def test_identity_default_and_fit_roundtrip():
    n = Normalizer(3)
    x = torch.randn(10, 3) * 5 + 2
    assert torch.equal(n.normalize(x), x)
    n.fit_mean_std(x)
    y = n.normalize(x)
    assert torch.allclose(y.mean(0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(y.std(0), torch.ones(3), atol=1e-5)
    assert torch.allclose(n.unnormalize(y), x, atol=1e-5)


def test_from_quantiles_matches_reference_formula():
    q01 = torch.tensor([-1.0, 0.0, 2.0, 5.0])
    q99 = torch.tensor([1.0, 4.0, 2.0, 5.0 + 1e-7])   # last two ranges are degenerate
    n = Normalizer.from_quantiles(q01, q99)
    x = torch.tensor([[0.5, 1.0, 2.0, 5.0], [-1.0, 4.0, 3.0, 4.0]])
    lo, hi = q01.numpy(), q99.numpy()
    rng = np.where(hi - lo > 1e-6, hi - lo, 1.0)
    ref = (x.numpy() - lo) / rng * 2.0 - 1.0
    assert np.allclose(n.normalize(x).numpy(), ref, atol=1e-6)
    back = n.unnormalize(n.normalize(x))
    assert torch.allclose(back[:, :2], x[:, :2], atol=1e-6)            # round trip on regular dims
    assert torch.all(back[:, 2] == 2.0) and torch.all(back[:, 3] == 5.0)  # constant dims return q01


def test_constant_dims_restore_the_constant():
    n = Normalizer.from_quantiles(torch.tensor([0.0, 5.0]), torch.tensor([2.0, 5.0]))
    assert n.constant.tolist() == [False, True]
    assert torch.allclose(n.normalize(torch.tensor([[0.0, 5.0]])), torch.tensor([[-1.0, -1.0]]))
    back = n.unnormalize(torch.tensor([[0.0, 0.7], [1.0, -3.0]]))
    assert torch.allclose(back[:, 0], torch.tensor([1.0, 2.0]))
    assert torch.all(back[:, 1] == 5.0)
    n.fit_mean_std(torch.randn(10, 2))
    assert not n.constant.any()


def test_chained_normalizer_order():
    q = Normalizer.from_quantiles(torch.zeros(2), torch.full((2,), 4.0))
    z = Normalizer.from_mean_std(torch.tensor([0.5, -0.5]), torch.tensor([2.0, 0.25]))
    c = ChainedNormalizer(q, z)
    x = torch.tensor([[1.0, 3.0]])
    expected = z.normalize(q.normalize(x))
    assert torch.allclose(c.normalize(x), expected)
    assert torch.allclose(c.unnormalize(c.normalize(x)), x, atol=1e-6)
    sd = c.state_dict()
    assert "stages.0.offset" in sd and "stages.1.scale" in sd
