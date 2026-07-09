"""Post-hoc waypoint-merge analysis for feature-cluster gravity runs.

Estimates an optimal number of waypoints starting from a large K (e.g. 128
centroids distributed on the hypersphere at t_q). The trained flow velocity is
frozen; centroids and their accumulated mass come from the EMA tracker.

Merge rule (greedy, mass-weighted, angular):
  - order active clusters by ascending mass;
  - for the lowest-mass cluster, look at its angular neighbours on the sphere;
  - if a neighbour has >= mass and the two flow-velocity vectors
    v_theta(centroid, t_q) deviate by less than the threshold angle, merge the
    small cluster into the larger one (mass summed, centroid = mass-weighted mean
    re-projected onto the sphere, velocity recomputed);
  - repeat until no admissible merge remains.

Surviving cluster count K_eff(threshold) is the estimated waypoint budget.
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Optional

import torch


@torch.no_grad()
def _centroid_velocities(model, centroids: torch.Tensor, t_q: float) -> torch.Tensor:
    """v_theta(centroid_i, t_q) for every centroid. Returns [K, D] (flattened).

    Centroids are stored flat [K, D]; image backbones need [K, C, H, W], so we
    reshape in/out based on the model's channel/spatial config.
    """
    t_vec = centroids.new_full((centroids.shape[0],), t_q)
    ic = getattr(model.config, "in_channels", None)
    isz = getattr(model.config, "image_size", None)
    x = centroids
    if ic is not None and isz is not None:
        x = centroids.reshape(centroids.shape[0], ic, isz, isz)
    # _velocity_fn handles any backbone wrapping; x0_cond=False runs ignore x0.
    v = model._velocity_fn(x, t_vec)
    return v.reshape(centroids.shape[0], -1)


def _unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _knn_adjacency(centroids: torch.Tensor, k: int) -> List[List[int]]:
    """k nearest neighbours (Euclidean) for each active centroid."""
    n = centroids.shape[0]
    if n <= 1:
        return [[] for _ in range(n)]
    d = torch.cdist(centroids, centroids)
    d.fill_diagonal_(float("inf"))
    kk = min(k, n - 1)
    idx = d.topk(kk, largest=False).indices  # [n, kk]
    return [row.tolist() for row in idx]


@torch.no_grad()
def _merge_at_threshold(
    model,
    centroids0: torch.Tensor,
    masses0: torch.Tensor,
    t_q: float,
    threshold_deg: float,
    k_neighbors: int,
    sphere_radius: float,
) -> dict:
    """Run the greedy merge for a single threshold. Returns summary dict."""
    cos_thr = math.cos(math.radians(threshold_deg))

    centroids = centroids0.clone()
    masses = masses0.clone()
    vel = _unit(_centroid_velocities(model, centroids, t_q))
    active = [i for i in range(centroids.shape[0]) if masses[i] > 0]

    n_merges = 0
    while True:
        adj = _knn_adjacency(centroids[active], k_neighbors)
        pos = {node: p for p, node in enumerate(active)}
        order = sorted(active, key=lambda i: masses[i].item())  # least massive first

        merged = False
        for i in order:
            neigh = [active[p] for p in adj[pos[i]]]
            # merge only into neighbours that are at least as massive
            cand = sorted(
                (j for j in neigh if masses[j] >= masses[i]),
                key=lambda j: masses[j].item(),
                reverse=True,  # prefer the heaviest neighbour
            )
            for j in cand:
                cos_ij = torch.dot(vel[i], vel[j]).clamp(-1.0, 1.0).item()
                if cos_ij >= cos_thr:  # angle < threshold
                    new_mass = masses[i] + masses[j]
                    new_c = (masses[i] * centroids[i] + masses[j] * centroids[j]) / new_mass
                    norm = new_c.norm().clamp(min=1e-8)
                    centroids[j] = new_c * (sphere_radius / norm)
                    masses[j] = new_mass
                    vel[j] = _unit(_centroid_velocities(model, centroids[j:j + 1], t_q))[0]
                    active.remove(i)
                    n_merges += 1
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break

    surviving_mass = masses[active]
    return {
        "threshold_deg": threshold_deg,
        "k_eff": len(active),
        "n_merges": n_merges,
        "surviving_indices": active,
        "surviving_mass": surviving_mass.tolist(),
        "surviving_centroids": centroids[active].tolist(),
    }


@torch.no_grad()
def run_merge_sweep(
    model,
    out_dir: str,
    t_q: Optional[float] = None,
    thresholds_deg: Optional[List[float]] = None,
    k_neighbors: int = 6,
    waypoint_index: int = 0,
    target_threshold_deg: float = 30.0,
) -> dict:
    """Sweep angular thresholds and report K_eff(threshold).

    Writes JSON + PNG to out_dir. Returns the results dict.
    """
    if getattr(model, "feature_clusters", None) is None:
        raise ValueError("model has no feature_clusters; merge analysis needs a gravity run")

    fc = model.feature_clusters[waypoint_index]
    if t_q is None:
        t_q = float(fc.t_q)
    if thresholds_deg is None:
        thresholds_deg = [10.0, 15.0, 20.0, 30.0, 45.0, 60.0]

    device = fc.centroids.device
    centroids0 = fc.centroids.detach().to(device)
    masses0 = fc.cluster_mass.detach().to(device)

    live = masses0 > 0
    sphere_radius = centroids0[live].norm(dim=-1).mean().item() if live.any() else 1.0
    k_start = int(live.sum().item())

    was_training = model.training
    model.eval()
    sweep = [
        _merge_at_threshold(
            model, centroids0, masses0, t_q, thr, k_neighbors, sphere_radius
        )
        for thr in thresholds_deg
    ]
    if was_training:
        model.train()

    results = {
        "t_q": t_q,
        "k_start": k_start,
        "sphere_radius": sphere_radius,
        "k_neighbors": k_neighbors,
        "sweep": sweep,
        "k_eff_by_threshold": {s["threshold_deg"]: s["k_eff"] for s in sweep},
        "target_threshold_deg": target_threshold_deg,
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "waypoint_merge.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    _plot(results, centroids0, masses0, model, t_q, out_dir, target_threshold_deg)
    return results


@torch.no_grad()
def _plot(results, centroids0, masses0, model, t_q, out_dir, target_threshold_deg=30.0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if centroids0.shape[1] != 2:
        # curve-only plot for non-2D data
        fig, ax = plt.subplots(figsize=(5, 4))
        thr = [s["threshold_deg"] for s in results["sweep"]]
        keff = [s["k_eff"] for s in results["sweep"]]
        ax.plot(thr, keff, "o-")
        ax.set_xlabel("angular threshold (deg)")
        ax.set_ylabel("surviving waypoints K_eff")
        ax.set_title(f"waypoint merge sweep (K_start={results['k_start']})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "waypoint_merge.png"), dpi=130)
        plt.close(fig)
        return

    live = masses0 > 0
    c = centroids0[live].cpu()
    m = masses0[live].cpu()
    vel = _unit(_centroid_velocities(model, centroids0[live], t_q)).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) K_eff vs threshold
    thr = [s["threshold_deg"] for s in results["sweep"]]
    keff = [s["k_eff"] for s in results["sweep"]]
    axes[0].plot(thr, keff, "o-")
    axes[0].set_xlabel("angular threshold (deg)")
    axes[0].set_ylabel("surviving waypoints K_eff")
    axes[0].set_title(f"merge sweep (K_start={results['k_start']})")
    for x, y in zip(thr, keff):
        axes[0].annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 6))

    # (2) all centroids coloured by mass + velocity quivers
    sc = axes[1].scatter(c[:, 0], c[:, 1], c=m, cmap="viridis", s=25)
    axes[1].quiver(c[:, 0], c[:, 1], vel[:, 0], vel[:, 1],
                   color="red", alpha=0.5, scale=25, width=0.004)
    axes[1].set_title(f"K_start={results['k_start']} centroids (mass + velocity)")
    axes[1].set_aspect("equal")
    fig.colorbar(sc, ax=axes[1], fraction=0.046)

    # (3) surviving set at the target threshold (or nearest available)
    target = min(results["sweep"], key=lambda s: abs(s["threshold_deg"] - target_threshold_deg))
    surv_c = torch.tensor(target["surviving_centroids"])
    surv_m = torch.tensor(target["surviving_mass"])
    axes[2].scatter(c[:, 0], c[:, 1], color="lightgray", s=12)
    if surv_c.numel() > 0:
        axes[2].scatter(surv_c[:, 0], surv_c[:, 1], c=surv_m, cmap="plasma", s=60,
                        edgecolors="black")
    axes[2].set_title(f"surviving @ {target['threshold_deg']:.0f} deg -> "
                      f"K_eff={target['k_eff']}")
    axes[2].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "waypoint_merge.png"), dpi=130)
    plt.close(fig)
