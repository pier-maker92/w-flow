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
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--trail_width", type=float, default=0.1)
    parser.add_argument("--return_trajectory", action="store_true")
    parser.add_argument("--disable_quantization", action="store_true")
    parser.add_argument("--force_waypoints", type=int, nargs="+", default=None)
    parser.add_argument("--gravity_g", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, device)
    cfg = model.config

    if cfg.image_size is not None and cfg.in_channels is not None:
        shape = (args.n_samples, cfg.in_channels, cfg.image_size, cfg.image_size)
    else:
        shape = (args.n_samples, cfg.data_dim)
    
    x0 = torch.randn(*shape, device=device) * cfg.flow_config.source_std

    with torch.no_grad():
        out = model.sample(
            x0=x0,
            num_steps=args.num_steps,
            return_trajectory=True,  # Always request trajectory to plot it
            disable_quantization=args.disable_quantization,
            force_waypoints=args.force_waypoints,
            gravity_g=args.gravity_g,
        )

    # Non salviamo più i tensori su disco come richiesto dal comando "non voglio salvare i tensori generati"
    # torch.save({"x1": out.x1.cpu(), "trajectory": out.trajectory}, args.output)
    # print(f"saved {args.n_samples} samples → {args.output}")

    if args.animate or args.save_grid:
        if cfg.image_size is not None and args.save_grid:
            from torchvision.utils import save_image
            grid_path = Path(args.output).with_suffix(".png")
            save_image((out.x1[:64].clamp(-1, 1) + 1) / 2, str(grid_path), nrow=8)
            print(f"saved grid → {grid_path}")
        elif cfg.data_dim == 2:
            import matplotlib.pyplot as plt
            import numpy as np
            
            x1_pts = out.x1.cpu().numpy()
            x0_pts = x0.cpu().numpy()
            
            if out.trajectory is not None and len(out.trajectory) > 0:
                if isinstance(out.trajectory, torch.Tensor):
                    traj = out.trajectory.cpu().numpy()
                else:
                    traj = torch.stack(out.trajectory).cpu().numpy()
                    
                # Trajectory is [B, steps, 2]. We need it to be [steps, B, 2] for the animation
                if traj.shape[0] == args.n_samples:
                    traj = np.transpose(traj, (1, 0, 2))
                
                if args.animate:
                    from matplotlib.animation import FuncAnimation
                    
                    # Compute colors based on final destination (to see where they end up)
                    final_pts = traj[-1]
                    angles = np.arctan2(final_pts[:, 1], final_pts[:, 0])
                    angles[angles < 0] += 2 * np.pi
                    classes = np.floor((angles + np.pi/8) / (2 * np.pi / 8)) % 8
                    
                    fig, ax = plt.subplots(figsize=(10, 10))
                    
                    # Calculate absolute dynamic limits across ALL frames to prevent moving scale
                    x_min, x_max = traj[:, :, 0].min(), traj[:, :, 0].max()
                    y_min, y_max = traj[:, :, 1].min(), traj[:, :, 1].max()
                    
                    # Make it square if you want an equal aspect
                    max_range = max(x_max - x_min, y_max - y_min)
                    x_mid = (x_max + x_min) / 2
                    y_mid = (y_max + y_min) / 2
                    
                    x_lim = (x_mid - max_range*0.6, x_mid + max_range*0.6)
                    y_lim = (y_mid - max_range*0.6, y_mid + max_range*0.6)
                    
                    def update(frame):
                        ax.clear()
                        ax.set_xlim(x_lim)
                        ax.set_ylim(y_lim)
                        ax.set_aspect('equal')
                        ax.autoscale(False)
                        ax.grid(True, alpha=0.3)
                        
                        # Plot centroids if feature cluster is used
                        if model._uses_feature_cluster and model.feature_clusters is not None:
                            colors = ['red', 'cyan', 'orange', 'purple', 'black']
                            for i, fc in enumerate(model.feature_clusters):
                                c = colors[i % len(colors)]
                                centroids = fc.centroids.cpu().numpy()
                                ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c=c, edgecolor='black', zorder=0)
                                
                        # Plot faint green trails up to the current frame
                        if frame > 0:
                            trail = traj[:frame+1] # [steps_so_far, B, 2]
                            # matplotlib plots columns of a 2D array as separate lines, which is perfect and very fast!
                            ax.plot(trail[:, :, 0], trail[:, :, 1], color='green', linewidth=args.trail_width, alpha=0.15, zorder=1)
                                
                        pts = traj[frame]
                        ax.scatter(pts[:, 0], pts[:, 1], s=12, c=classes, cmap='tab10', alpha=0.8, vmin=0, vmax=9, zorder=2)
                        ax.set_title(f"Integration Step {frame}/{traj.shape[0]-1}")
                        
                    ani = FuncAnimation(fig, update, frames=traj.shape[0], interval=1000//args.fps, blit=False)
                    video_path = Path(args.output).with_suffix(".mp4")
                    ani.save(str(video_path), writer='ffmpeg', fps=args.fps)
                    plt.close()
                    print(f"saved video → {video_path}")
            
            if not args.save_grid:
                return
                
            grid_path = Path(args.output).with_suffix(".png")
            plt.figure(figsize=(10, 10))
            
            # Plot max 500 trajectories to avoid crowding the plot too much
            n_plot = min(traj.shape[1], 500)
            plt.plot(traj[:, :n_plot, 0], traj[:, :n_plot, 1], color='green', linewidth=args.trail_width, alpha=0.3, label='Trajectories')
                
            # Plot x0 in blue
            plt.scatter(x0_pts[:, 0], x0_pts[:, 1], s=4, alpha=0.5, c='blue', label='x0 (Noise)')
            
            # Color x1 by angle to distinguish the 8 Gaussians
            angles = np.arctan2(x1_pts[:, 1], x1_pts[:, 0])
            angles[angles < 0] += 2 * np.pi
            classes = np.floor((angles + np.pi/8) / (2 * np.pi / 8)) % 8
            plt.scatter(x1_pts[:, 0], x1_pts[:, 1], s=6, alpha=0.8, c=classes, cmap='tab10', label='x1 (Generated)')
            
            # Plot centroids if feature cluster is used
            if model._uses_feature_cluster and model.feature_clusters is not None:
                colors = ['red', 'cyan', 'orange', 'purple', 'black']
                for i, fc in enumerate(model.feature_clusters):
                    c = colors[i % len(colors)]
                    centroids = fc.centroids.cpu().numpy()
                    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c=c, edgecolor='black', label=f'WP {i} Centroids')
                
            plt.title("Generated 2D Samples & Trajectories")
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            # Clean up legend to avoid repeating line labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            
            plt.savefig(grid_path, dpi=300)
            plt.close()
            print(f"saved scatter plot → {grid_path}")


if __name__ == "__main__":
    main()
