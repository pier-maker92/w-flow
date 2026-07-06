"""Standalone evaluation script.

Phase 1: trajectory curvature + codebook usage plots.
Phase 2: dSprites disentanglement (factor swap).
Phase 3: FID.

Usage:
  python evaluate.py --checkpoint checkpoints/exps/phase1-gaussians \
                     --phase 1 --n_samples 2000 --output_dir evaluation/phase1-gaussians
"""
import argparse
import json
import os
from pathlib import Path

import torch

from modules.configs import FlowQuantConfig
from util import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from inference import load_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, device)
    cfg = model.config
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    x0 = torch.randn(args.n_samples, cfg.data_dim, device=device) * cfg.flow_config.source_std
    with torch.no_grad():
        out = model.sample(x0=x0, num_steps=args.num_steps, return_trajectory=True)

    if args.phase == 1:
        from evaluation.trajectory import plot_trajectories, trajectory_curvature, plot_codebook_usage
        plot_trajectories(
            out.trajectory, os.path.join(args.output_dir, "trajectories.png"),
            t_q=cfg.flow_config.t_q, n_show=200,
        )
        curvature = trajectory_curvature(out.trajectory)
        print(f"mean squared curvature: {curvature:.6f}")

        if out.quantizer_output is not None and out.quantizer_output.indices is not None:
            n_emb = cfg.vq_config.num_embeddings if cfg.vq_config is not None else None
            if n_emb is not None:
                plot_codebook_usage(
                    out.quantizer_output.indices.reshape(-1),
                    n_emb,
                    os.path.join(args.output_dir, "codebook_usage.png"),
                )

        results = {"curvature": float(curvature)}
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

    elif args.phase == 2:
        print("Phase 2 disentanglement evaluation — see evaluation/disentanglement.py")

    elif args.phase == 3:
        from evaluation.fid import compute_fid
        from torch.utils.data import DataLoader
        _, collator = build_dataset(cfg.training_config, split="test")
        eval_ds, _ = build_dataset(cfg.training_config, split="test")
        loader = DataLoader(eval_ds, batch_size=64, collate_fn=collator, num_workers=4)
        fid = compute_fid(loader, out.x1, device, max_samples=args.n_samples)
        print(f"FID: {fid:.2f}")
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump({"fid": fid}, f, indent=2)


if __name__ == "__main__":
    main()
