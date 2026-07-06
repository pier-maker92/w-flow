"""Phase 1 trajectory visualization — ODE paths, curvature, waypoint analysis."""
from pathlib import Path
from typing import Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_trajectories(
    trajectories: torch.Tensor,   # [B, T_steps+1, 2]
    output_path: str,
    t_q: Optional[float] = None,
    title: str = "ODE Trajectories",
    n_show: int = 200,
) -> None:
    traj = trajectories[:n_show].cpu().numpy()   # [n_show, T, 2]
    T = traj.shape[1]
    colors = cm.viridis(np.linspace(0, 1, T))

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(traj)):
        ax.plot(traj[i, :, 0], traj[i, :, 1], alpha=0.3, linewidth=0.8, color="steelblue")
    ax.scatter(traj[:, 0, 0], traj[:, 0, 1], s=4, color="green", label="x0", zorder=3)
    ax.scatter(traj[:, -1, 0], traj[:, -1, 1], s=4, color="red", label="x1", zorder=3)

    if t_q is not None:
        t_q_step = int(t_q * (T - 1))
        ax.scatter(
            traj[:, t_q_step, 0], traj[:, t_q_step, 1],
            s=8, color="orange", label=f"waypoint t_q={t_q}", zorder=4,
        )

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def trajectory_curvature(trajectories: torch.Tensor) -> torch.Tensor:
    """Mean squared curvature: E[||d²x/dt²||²] approximated via finite differences.

    trajectories: [B, T, D] → scalar
    """
    v = trajectories[:, 1:, :] - trajectories[:, :-1, :]   # [B, T-1, D]
    a = v[:, 1:, :] - v[:, :-1, :]                         # [B, T-2, D]
    return a.pow(2).sum(-1).mean()


def plot_codebook_usage(
    indices: torch.LongTensor,   # [N] flattened codebook indices
    num_embeddings: int,
    output_path: str,
) -> None:
    counts = torch.bincount(indices.cpu(), minlength=num_embeddings).float()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(num_embeddings), counts.numpy())
    ax.set_xlabel("Codebook index")
    ax.set_ylabel("Usage count")
    ax.set_title("Codebook usage distribution")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
