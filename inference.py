"""Sample from a trained FlowQuant model.

Usage (Phase 1):
  python inference.py --checkpoint checkpoints/exps/phase1-gaussians \
                      --n_samples 1000 --num_steps 100 --output samples.pt

Phase 2-3 (image): add --save_grid to save a PNG grid.
"""
import argparse
import json
import os
from pathlib import Path

import torch

from modules.builder import build_model


def load_model(checkpoint_dir: str, device: torch.device):
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    model = build_model(cfg_dict)
    ckpt = os.path.join(checkpoint_dir, "model.safetensors")
    from safetensors.torch import load_file
    state = load_file(ckpt)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--output", default="samples.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_grid", action="store_true")
    parser.add_argument("--return_trajectory", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, device)
    cfg = model.config

    x0 = torch.randn(args.n_samples, cfg.data_dim, device=device) * cfg.flow_config.source_std

    with torch.no_grad():
        out = model.sample(
            x0=x0,
            num_steps=args.num_steps,
            return_trajectory=args.return_trajectory,
        )

    torch.save({"x1": out.x1.cpu(), "trajectory": out.trajectory}, args.output)
    print(f"saved {args.n_samples} samples → {args.output}")

    if args.save_grid and cfg.image_size is not None:
        from torchvision.utils import save_image
        grid_path = Path(args.output).with_suffix(".png")
        save_image((out.x1[:64].clamp(-1, 1) + 1) / 2, str(grid_path), nrow=8)
        print(f"saved grid → {grid_path}")


if __name__ == "__main__":
    main()
