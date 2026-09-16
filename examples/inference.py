"""Run one DEM inference step on random frames.

    python examples/inference.py                      # public DINOv3 + NeoBERT weights, random head
    python examples/inference.py --checkpoint dem.pt  # a policy saved with DEM.save
    python examples/inference.py --offline            # random-init towers (no downloads)

The head is untrained unless a checkpoint is given, so the printed actions
are meaningless; the script shows the API and measures latency.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from dem import DEM, DEMConfig, preprocess_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--offline", action="store_true", help="random-init towers, no Hub downloads")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--instruction", default="open the left drawer")
    ap.add_argument("--nfe", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if args.checkpoint:
        policy = DEM.load(args.checkpoint)
    else:
        cfg = DEMConfig(vision_pretrained=not args.offline, language_pretrained=not args.offline)
        if args.offline:
            from dem import LanguageEncoder
            from dem.language import NeoBERTConfig
            lang = LanguageEncoder(backend="neobert", pretrained=False, neobert_config=NeoBERTConfig(),
                                   max_tokens=cfg.max_language_tokens)
            policy = DEM(cfg, language=lang)
        else:
            policy = DEM(cfg)
    policy = policy.to(args.device).eval()
    cfg = policy.config
    print({k: f"{v / 1e6:.1f}M" for k, v in policy.count_parameters().items()})

    # Two uint8 camera frames (scene + wrist) and a proprioceptive state, as a robot would provide them.
    frames = np.random.randint(0, 255, (cfg.num_cameras, 480, 640, 3), dtype=np.uint8)
    proprio = torch.zeros(1, cfg.proprio_dim)
    images = preprocess_images(frames, size=cfg.image_size, device=args.device)[None]   # [1, K, 3, S, S]

    if policy.language.tokenizer is not None:
        policy.set_instruction(args.instruction)        # encoded once, reused at every step below
    else:                                               # offline mode has no tokenizer: feed token ids directly
        ids = torch.randint(1, 100, (1, 6), device=args.device)
        policy.set_instruction_tokens(*policy.language.encode_ids(ids, torch.ones_like(ids)))

    with torch.no_grad():
        actions = policy.act(images, proprio, nfe=args.nfe)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            actions = policy.act(images, proprio, nfe=args.nfe)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.iters * 1000
    print(f"action chunk {tuple(actions.shape)}; {dt:.2f} ms per step on {args.device} (language cached)")


if __name__ == "__main__":
    main()
