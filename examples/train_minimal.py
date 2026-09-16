"""Minimal training loop for DEM on a synthetic dataset.

Shows the recipe used for the paper's runs: AdamW with separate learning
rates for the head and the vision encoder, cosine schedule, gradient
clipping at 1.0, MeanFlow loss with the dispersive regularizer, vision
encoder under bf16 autocast while the head and the loss stay in fp32.
Replace ``SyntheticDataset`` with your own data (frames, proprio, action
chunks, instruction strings).

    python examples/train_minimal.py --steps 50 --offline
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Dataset

from dem import DEM, DEMConfig, Normalizer, preprocess_images


class SyntheticDataset(Dataset):
    def __init__(self, n: int, cfg: DEMConfig, instructions: list[str]) -> None:
        self.n, self.cfg, self.instructions = n, cfg, instructions

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        cfg = self.cfg
        frames = torch.randint(0, 256, (cfg.num_cameras, cfg.image_size, cfg.image_size, 3), dtype=torch.uint8)
        proprio = torch.randn(cfg.proprio_dim)
        actions = torch.randn(cfg.chunk, cfg.action_dim)
        return frames, proprio, actions, self.instructions[i % len(self.instructions)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-vision", type=float, default=5e-5)
    ap.add_argument("--freeze-vision", action="store_true")
    ap.add_argument("--offline", action="store_true", help="random-init towers, no Hub downloads")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="dem_synthetic.pt")
    args = ap.parse_args()
    torch.manual_seed(0)

    cfg = DEMConfig(freeze_vision=args.freeze_vision, vision_pretrained=not args.offline,
                    language_pretrained=not args.offline, dispersive_weight=0.25)
    if args.offline:
        from dem import LanguageEncoder
        from dem.language import NeoBERTConfig
        lang = LanguageEncoder(backend="neobert", pretrained=False, neobert_config=NeoBERTConfig(num_hidden_layers=2),
                               max_tokens=cfg.max_language_tokens)
        policy = DEM(cfg, language=lang)
    else:
        policy = DEM(cfg)
    policy = policy.to(args.device)

    instructions = ["open the left drawer", "close the fridge", "turn on the electric kettle"]
    ds = SyntheticDataset(1000, cfg, instructions)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    # Normalization statistics from the training set (here: a few synthetic batches).
    sample = torch.cat([ds[i][2][None] for i in range(64)])
    policy.action_norm = Normalizer(cfg.action_dim).fit_mean_std(sample).to(args.device)
    policy.proprio_norm = Normalizer(cfg.proprio_dim).fit_mean_std(
        torch.stack([ds[i][1] for i in range(64)])).to(args.device)

    if policy.language.tokenizer is None:   # offline mode has no tokenizer; use fixed random ids per instruction
        for text in instructions:
            ids = torch.randint(1, 100, (1, 6), device=args.device)
            tok, mask = policy.language.encode_ids(ids, torch.ones_like(ids))
            policy.cache_instruction(text, tok[0], mask[0])

    opt = torch.optim.AdamW(policy.parameter_groups(args.lr_head, args.lr_vision),
                            weight_decay=1e-4, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    trainable = [p for g in opt.param_groups for p in g["params"]]
    use_autocast = args.device.startswith("cuda")

    policy.train()
    step = 0
    while step < args.steps:
        for frames, proprio, actions, texts in dl:
            images = preprocess_images(frames, size=cfg.image_size, device=args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                vis_tokens = policy.encode_vision(images)
            loss, aux = policy.loss(None, proprio, actions, texts=list(texts), vis_tokens=vis_tokens.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 10 == 0 or step == args.steps:
                print(f"step {step}: loss {loss.item():.4f} raw_mse {aux['raw_mse']:.4f} "
                      f"disp {aux.get('disp', float('nan')):.4f}")
            if step >= args.steps:
                break
    policy.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
