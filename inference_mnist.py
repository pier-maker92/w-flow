"""Sample from a trained FlowQuant model for MNIST and create an animation.

Usage:
  python inference_mnist.py --checkpoint checkpoints/exps/09-July-2026/your-experiment \
                            --n_samples 64 --num_steps 100 --output mnist_anim.mp4
"""
import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torchvision.utils import make_grid

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
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--output", default="mnist_animation.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--disable_quantization", action="store_true")
    parser.add_argument("--gravity_g", type=float, default=0.0, help="Gravitational constant (default: 0.0)")
    parser.add_argument("--gravity_clusters", type=int, nargs="+", default=None, help="List of centroid indices to apply gravity to.")
    parser.add_argument("--start_from_centroids", action="store_true", help="Start the ODE from the centroids instead of noise.")
    parser.add_argument("--start_clusters", type=int, nargs="+", default=[-1], help="Clusters to start from. -1 means all.")
    parser.add_argument("--start_noise", type=float, default=0.0, help="Standard deviation of Gaussian noise added to centroids.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, device)
    cfg = model.config

    if cfg.image_size is not None and cfg.in_channels is not None:
        shape = (args.n_samples, cfg.in_channels, cfg.image_size, cfg.image_size)
    else:
        raise ValueError("This script is intended for image datasets (like MNIST).")
    
    if args.start_from_centroids:
        if not model._uses_feature_cluster:
            raise ValueError("Model does not use feature clusters. Cannot start from centroids.")
        fc = model.feature_clusters[0]
        K = fc.centroids.shape[0]
        if -1 in args.start_clusters:
            clusters = list(range(K))
        else:
            clusters = args.start_clusters
            
        idx = torch.arange(args.n_samples) % len(clusters)
        cluster_idx = torch.tensor(clusters, device=device)[idx]
        
        x0 = fc.centroids[cluster_idx].reshape(shape).to(device)
        if args.start_noise > 0.0:
            x0 = x0 + torch.randn_like(x0) * args.start_noise
        t_start = cfg.flow_config.waypoints()[0]
    else:
        x0 = torch.randn(*shape, device=device) * cfg.flow_config.source_std
        t_start = 0.0

    print("Generating trajectory...")
    with torch.no_grad():
        out = model.sample(
            x0=x0,
            num_steps=args.num_steps,
            return_trajectory=True,  # Always request trajectory to animate it
            disable_quantization=args.disable_quantization,
            gravity_g=args.gravity_g,
            gravity_clusters=args.gravity_clusters,
            t_start=t_start,
        )

    if out.trajectory is None or len(out.trajectory) == 0:
        print("No trajectory returned from model.")
        return

    # Prepare trajectory tensor
    if isinstance(out.trajectory, torch.Tensor):
        traj = out.trajectory.cpu()
    else:
        traj = torch.stack(out.trajectory).cpu()
        
    # Ensure traj is [steps, B, C, H, W]
    if traj.shape[0] == args.n_samples:
        traj = traj.transpose(0, 1)

    print(f"Trajectory shape: {traj.shape}. Creating animation...")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    
    # Initialize with the first frame (clamped and normalized to [0, 1])
    first_frame_imgs = (traj[0].clamp(-1, 1) + 1) / 2
    grid = make_grid(first_frame_imgs, nrow=int(np.sqrt(args.n_samples)))
    img_plot = ax.imshow(grid.permute(1, 2, 0).numpy())
    title = ax.set_title("Step 0")

    def update(frame):
        imgs = (traj[frame].clamp(-1, 1) + 1) / 2
        grid = make_grid(imgs, nrow=int(np.sqrt(args.n_samples)))
        img_plot.set_data(grid.permute(1, 2, 0).numpy())
        title.set_text(f"Step {frame}/{traj.shape[0]-1}")
        return [img_plot, title]

    ani = FuncAnimation(fig, update, frames=traj.shape[0], interval=1000//args.fps, blit=True)
    video_path = Path(args.output).with_suffix(".mp4")
    ani.save(str(video_path), writer='ffmpeg', fps=args.fps)
    plt.close()
    print(f"Animation saved to {video_path}")


if __name__ == "__main__":
    main()
