"""Numerical parity of the NeoBERT port against the released reference code.

Downloads only ``model.py`` and ``rotary.py`` from ``chandar-lab/NeoBERT`` (a
few KB), imports them with ``xformers.ops.SwiGLU`` replaced by a plain PyTorch
module of the same parameterization, and compares the two implementations on
a tiny random configuration with identical weights. Weights are not downloaded.
"""

import importlib
import os
import shutil
import sys
import tempfile
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dem.language import NeoBERT, NeoBERTConfig

pytestmark = pytest.mark.network


class _SwiGLU(nn.Module):
    """Same parameters and math as xformers.ops.SwiGLU with packed weights."""

    def __init__(self, in_features, hidden_features, out_features, bias=True):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def _import_reference():
    from huggingface_hub import hf_hub_download

    pkg_dir = tempfile.mkdtemp(prefix="neobert_ref_")
    for fn in ("model.py", "rotary.py"):
        shutil.copy(hf_hub_download("chandar-lab/NeoBERT", fn), os.path.join(pkg_dir, fn))
    open(os.path.join(pkg_dir, "__init__.py"), "w").close()
    fake_ops = types.ModuleType("xformers.ops")
    fake_ops.SwiGLU = _SwiGLU
    fake = types.ModuleType("xformers")
    fake.ops = fake_ops
    sys.modules.setdefault("xformers", fake)
    sys.modules.setdefault("xformers.ops", fake_ops)
    sys.path.insert(0, os.path.dirname(pkg_dir))
    return importlib.import_module(os.path.basename(pkg_dir) + ".model")


def test_port_matches_reference_on_random_weights():
    ref_mod = _import_reference()
    kwargs = dict(hidden_size=32, num_hidden_layers=3, num_attention_heads=4, intermediate_size=64,
                  vocab_size=97, max_length=40, norm_eps=1e-5, pad_token_id=0)
    ref = ref_mod.NeoBERT(ref_mod.NeoBERTConfig(**kwargs)).eval()
    port = NeoBERT(NeoBERTConfig(**kwargs)).eval()
    port.load_state_dict(ref.state_dict(), strict=True)   # same names and shapes

    torch.manual_seed(1)
    ids = torch.randint(1, 97, (3, 11))
    mask = torch.ones(3, 11)
    mask[1, 7:] = 0
    mask[2, 4:] = 0
    with torch.no_grad():
        out_ref = ref(input_ids=ids, attention_mask=mask).last_hidden_state
        out_port = port(ids, mask)
    diff = (out_ref - out_port).abs().max().item()
    assert diff < 1e-5, f"max abs difference vs reference {diff}"

    # Real-token rows must agree with the per-sequence unpadded forward as well.
    with torch.no_grad():
        single = port(ids[1:2, :7], torch.ones(1, 7))
    assert torch.allclose(single[0], out_port[1, :7], atol=1e-5)
