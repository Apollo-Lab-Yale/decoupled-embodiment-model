# DEM: Decoupled Embodiment Model

DEM is a robot policy built from three separate networks: a DINOv3 ConvNeXt-B
vision encoder, a frozen NeoBERT language encoder, and a MeanFlow action head.
The encoders never see each other. Their token sequences meet only in the
cross-attention of the head, which maps noise to a chunk of actions in one
forward pass. This repository is the model as an importable PyTorch package:
the three modules, the MeanFlow objective and sampler, normalizers, and a
`DEM` class that assembles them. Datasets, simulators, baselines and the
ablation code of the research repository are not part of it, and the trained
weights of the paper are not distributed here yet.

```
scene + wrist frames ─► DINOv3 ConvNeXt-B (stage 3) ─► 2 × 256 tokens × 512 ─┐
instruction          ─► NeoBERT (frozen, cached)     ─► 32 tokens × 768     ─┼─► context (545 tokens)
proprioceptive state ─► linear                       ─► 1 token             ─┘        │
                                                                                     ▼ cross-attention in every block
noise ε [H × A] ─► MeanFlow head (8 AdaLN blocks, d = 768) ─► u(ε, r=0, t=1) ─► actions = ε − u
```

## Contents

| Module | Class | File | Parameters |
|---|---|---|---|
| Vision encoder | `DinoV3ConvNeXtEncoder` | `dem/vision/convnext.py` | 87.6M |
| Vision encoder (ablation) | `DinoV3ViTEncoder` | `dem/vision/vit.py` | 85.7M |
| Language encoder | `LanguageEncoder`, `NeoBERT` | `dem/language/` | 221.7M as loaded, of which 23.4M is the token embedding table |
| Action head | `MeanFlowTokenHead` | `dem/head/meanflow_head.py` | 107.3M (107.0M with 384-d language tokens) |
| Objective and sampler | `MeanFlowObjective` | `dem/head/meanflow.py` | |
| Normalizers | `Normalizer`, `ChainedNormalizer` | `dem/normalization.py` | |
| Assembled policy | `DEM`, `DEMConfig` | `dem/policy.py` | |

Parameter counts are measured on the instantiated modules. The NeoBERT
paper reports 250M, which includes the masked-LM decoder that the policy
does not load.

## Installation

```bash
pip install -e .            # torch >= 2.4, transformers >= 5.13, safetensors, huggingface_hub >= 1.0
pip install -e ".[vit]"     # adds timm for the DINOv3 ViT-B/16 encoder
pip install -e ".[dev]"     # adds pytest
```

Two checkpoints come from the Hugging Face Hub. `chandar-lab/NeoBERT` is
public (MIT). `facebook/dinov3-convnext-base-pretrain-lvd1689m` is gated:
accept the terms on its model page and run `hf auth login` once. The
ConvNeXt architecture config is vendored in `dem/configs/`, so building the
encoder with `pretrained=False` works offline.

## Inference

```python
import torch
from dem import DEM, DEMConfig, Normalizer, preprocess_images

cfg = DEMConfig(action_dim=12, proprio_dim=16, chunk=16, num_cameras=2)
policy = DEM(cfg).eval().cuda()
policy.action_norm = Normalizer.from_mean_std(action_mean, action_std).cuda()   # training statistics
policy.proprio_norm = Normalizer.from_mean_std(state_mean, state_std).cuda()

policy.set_instruction("open the left drawer")     # encoded once, reused below

frames = ...                                        # uint8 array [2, H, W, 3]: scene camera, wrist camera
state = ...                                         # float tensor [16] in raw units
images = preprocess_images(frames, size=256, device="cuda")[None]   # [1, 2, 3, 256, 256] in [0, 1]
actions = policy.act(images, state[None])           # [1, 16, 12] in raw units, one head forward pass
```

`act` runs the vision encoder on every call and the language encoder only
when the instruction changes. Pass `texts=` to `act` to encode a new
instruction on the fly; strings are cached, so repeating one costs nothing.
`set_instruction_tokens` and `cache_instruction` accept pre-encoded tokens
for deployments without a tokenizer. `nfe` splits the interval into several
MeanFlow steps (the paper uses one). `z0` fixes the initial noise across
calls.

Frames must have the orientation of the training videos. Renders from
robosuite and other OpenGL pipelines are upside down and need a vertical
flip before `preprocess_images`, which scales uint8 values to [0, 1] and
resizes once from the native resolution with antialiased bilinear
interpolation.

The normalizers default to the identity. Set them to the statistics used in
training, either with `Normalizer.fit_mean_std` / `Normalizer.from_quantiles`
or by loading a saved policy; otherwise the head sees inputs on the wrong
scale and the output is not in raw units.

## Training

`DEM.loss` returns the MeanFlow loss for a batch of frames, states and
action chunks. The recipe used for the paper's runs: AdamW with learning
rate 1e-4 for the head and 5e-5 for the vision encoder, weight decay 1e-4,
betas (0.9, 0.95), a cosine schedule, gradient clipping at 1.0, the
dispersive regularizer at weight 0.25, and `ratio_r_eq_t = 0.5`. The vision
encoder ran under bf16 autocast while the head and the loss stayed in fp32.
`examples/train_minimal.py` is a complete loop on synthetic data:

```python
opt = torch.optim.AdamW(policy.parameter_groups(lr_head=1e-4, lr_vision=5e-5),
                        weight_decay=1e-4, betas=(0.9, 0.95))
with torch.autocast("cuda", dtype=torch.bfloat16):
    vis_tokens = policy.encode_vision(images)                 # [B, 512, 512]
loss, aux = policy.loss(None, proprio, actions, texts=instructions, vis_tokens=vis_tokens)
loss.backward()
```

`build_cond` casts the features to the head's parameter dtype, so
`vis_tokens` from autocast or from a half-precision cache can be passed as
they are. `DEMConfig.freeze_vision` switches between fine-tuning the
encoder with the head (the paper's final configuration) and keeping it
frozen. The language encoder is frozen by default; with
`freeze_language=False` it joins the optimizer groups and its tokens are no
longer cached.

## Checkpoints

`policy.save(path)` writes the config, the normalizers and the weights.
A frozen tower is omitted from the file when it still holds the public
Hub checkpoint the policy was built from, and `DEM.load` downloads it
again; a tower whose weights were replaced is always written, and
`include_frozen=True` writes everything. `DEM.load(path, **overrides)`
rebuilds the policy including its normalizers, and rejects override names
that are not `DEMConfig` fields.

`policy.load_research_checkpoints(head=, vision=, norm=, norm_stats=)`
reads the artifacts of the research code, which are not part of this
repository: a head `state_dict` (`mf_final.pt`, or the `"head"` entry of
`uf_final.pt`), encoder weights in any of the layouts that
`VisionEncoder.load_weights` accepts, and the two normalization stages of
those runs. Actions and states were first scaled per task to [-1, 1] from
the `q01` / `q99` quantiles in the dataset's `norm_stats.json` (dims with
a constant value map to -1 and are restored exactly), then z-scored with
the statistics stored in `norm.pt`; pass both files to get raw-unit
actions out of `act`. Heads trained in that code used mmBERT-small
tokens (384-d); to run them, build the policy with
`DEMConfig(language_backend="hf", language_model="jhu-clsp/mmBERT-small")`.

## Interface details

The head takes a conditioning dict with `vis_tokens [B, Nv, Dv]`,
`lang_tokens [B, Nl, Dl]`, `lang_mask [B, Nl]` (bool, True = valid) and
`proprio [B, P]` (already normalized). `DEM.build_cond` produces it. Each
entry is projected to the head width, tagged with a learned modality
embedding and concatenated: 512 vision tokens, 32 language tokens and one
proprio token give the 545 context tokens of the paper's configuration.
Camera token sequences are concatenated along the token axis in camera
order, and the 16 x 16 grid order of each camera is preserved. A
precomputed context can be passed under the key `_ctx`; the sampler does
this itself when `nfe > 1`.

MeanFlow conventions follow the paper: `t = 0` is the data end and `t = 1`
the noise end of `z_t = (1 - t) a + t eps`. The head predicts the average
velocity `u(z_t, r, t)`, trained with the identity
`u = v - (t - r) du/dt` where `du/dt` is one forward-mode JVP, and the
one-step sample is `eps - u(eps, 0, 1)`. Attention inside the head is
written with explicit matmuls because fused attention kernels do not
support forward-mode differentiation.

## The NeoBERT port

The released NeoBERT loads through `trust_remote_code=True` and imports
`xformers.ops.SwiGLU` without a fallback. `dem/language/neobert.py` is the
same network in plain PyTorch: token embedding, 28 pre-RMSNorm blocks with
rotary embeddings on queries and keys, SwiGLU feed-forward, final RMSNorm.
Parameter names and the per-head interleaved QKV layout match the released
`model.safetensors`, so loading is strict. The tokenizer is loaded through
the BERT tokenizer class named in the repository, which avoids the
remote-code prompt that `AutoTokenizer` raises for this repository.

Verification in `tests/`: the rotary embedding is checked against the
complex-number formulation of the original; masked padding does not change
the outputs of real tokens; `test_neobert_reference_parity.py` imports the
released `model.py` (with a plain PyTorch stand-in for the xformers SwiGLU)
and compares both implementations on identical random weights, agreeing to
1e-5; and with the released weights the masked-LM head fills "The capital
of France is [MASK]." with "paris". The port accepts `attention_mask=None`
and keeps its rotary tables in float32 under `.to(torch.bfloat16)`.

## Tests

```bash
pytest -m "not slow"                        # unit tests, no downloads, about 15 s on CPU
pytest -m slow                              # toy MeanFlow convergence, full-size random-init forward pass
DEM_NETWORK_TESTS=1 pytest -m network       # Hub downloads: NeoBERT weights (about 1 GB), reference parity, DEM.load round trip
```
