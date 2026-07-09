from typing import Callable, List, Optional, Tuple
import torch

from ..output_dataclasses import SolverOutput


class ODESolver:
    """Euler, Midpoint, and RK4 ODE solvers with optional quantization waypoints."""

    METHODS = ("euler", "midpoint", "rk4")

    def __init__(self, method: str) -> None:
        if method not in self.METHODS:
            raise ValueError(f"unknown solver method '{method}'; choose from {self.METHODS}")
        self.method = method

    @torch.no_grad()
    def solve(
        self,
        x0: torch.FloatTensor,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_steps: int,
        # single-waypoint API (backward compat)
        t_q: Optional[float] = None,
        quantize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        # multi-waypoint API
        t_qs: Optional[List[float]] = None,
        quantize_fns: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        return_trajectory: bool = False,
        gravity_g: float = 0.0,
        feature_clusters: Optional[List] = None,
        gravity_clusters: Optional[List[int]] = None,
        t_start: float = 0.0,
    ) -> SolverOutput:
        """Integrate dx/dt = v_θ(x, t) from t=0 to t=1.

        Waypoints are applied once when t first crosses each t_q value.
        Accepts either single (t_q, quantize_fn) or lists (t_qs, quantize_fns).
        """
        # Normalise to a sorted list of (t_q, fn) pairs
        if t_qs is not None and quantize_fns is not None:
            waypoints: List[Tuple[float, Callable]] = sorted(
                zip(t_qs, quantize_fns), key=lambda p: p[0]
            )
        elif t_q is not None and quantize_fn is not None:
            waypoints = [(t_q, quantize_fn)]
        else:
            waypoints = []

        applied = [False] * len(waypoints)
        for i, (tq, _) in enumerate(waypoints):
            if tq <= t_start + 1e-5:
                applied[i] = True
                
        x = x0.clone()
        t = t_start

        traj = [x.clone()] if return_trajectory else None

        while t < 1.0 - 1e-5:
            # Default proposed step
            dt = 1.0 / num_steps
            t_next = min(t + dt, 1.0)

            # Intercept with unapplied waypoints
            for i, (tq, qfn) in enumerate(waypoints):
                if not applied[i] and t < tq <= t_next + 1e-5:
                    t_next = tq
                    break

            dt_step = t_next - t

            if dt_step > 1e-6:
                t_vec = x.new_full((x.shape[0],), t)
                
                # Calcola la forza gravitazionale se abilitata
                gravity_v = 0.0
                if gravity_g > 0 and feature_clusters is not None:
                    # Trova il prossimo waypoint non ancora applicato
                    next_wp_idx = None
                    for i, (tq, _) in enumerate(waypoints):
                        if not applied[i]:
                            next_wp_idx = i
                            break
                    
                    if next_wp_idx is not None:
                        fc = feature_clusters[next_wp_idx]
                        centroids = fc.centroids  # [K, D]
                        # Normalize mass so it sums to 1. This prevents G from needing to be 1e-8 due to unbounded accumulation.
                        raw_masses = fc.cluster_mass
                        masses = raw_masses / (raw_masses.sum() + 1e-8)  # [K]
                        
                        eps = getattr(fc.config, "gravity_softening", 0.01)
                        
                        # Calcola le distanze da tutti i centroidi
                        # x_flat: [B, D], centroids: [K, D]
                        x_flat = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
                        # dist_sq: [B, K]
                        diff = centroids.unsqueeze(0) - x_flat.unsqueeze(1) # [B, K, D]
                        dist_sq = (diff ** 2).sum(dim=-1)
                        dist = torch.sqrt(dist_sq.clamp(min=1e-8))
                        
                        # F = G * m / (d^2 + eps) * (diff / d)
                        # diff / dist shape is [B, K, D]
                        direction = diff / dist.unsqueeze(-1)
                        magnitude = gravity_g * masses.unsqueeze(0) / (dist_sq + eps) # [B, K]
                        
                        if gravity_clusters is not None:
                            mask = torch.zeros(magnitude.shape[1], device=magnitude.device, dtype=torch.bool)
                            mask[gravity_clusters] = True
                            magnitude = magnitude * mask.unsqueeze(0)
                            
                        force = (magnitude.unsqueeze(-1) * direction).sum(dim=1) # [B, D]
                        gravity_v = force.reshape(x.shape)

                def get_v(x_curr, t_curr):
                    v = velocity_fn(x_curr, t_curr)
                    if isinstance(gravity_v, float) and gravity_v == 0.0:
                        return v
                        
                    v_flat = v.reshape(v.shape[0], -1)
                    v_norm = torch.norm(v_flat, p=2, dim=1).view(-1, *([1] * (v.ndim - 1)))
                    
                    v_new = v + gravity_v
                    v_new_flat = v_new.reshape(v_new.shape[0], -1)
                    v_new_norm = torch.norm(v_new_flat, p=2, dim=1).view(-1, *([1] * (v.ndim - 1)))
                    
                    return v_new * (v_norm / v_new_norm.clamp(min=1e-8))

                if self.method == "euler":
                    x = x + dt_step * get_v(x, t_vec)
                elif self.method == "midpoint":
                    v1 = get_v(x, t_vec)
                    x_mid = x + 0.5 * dt_step * v1
                    t_mid = x.new_full((x.shape[0],), t + 0.5 * dt_step)
                    x = x + dt_step * get_v(x_mid, t_mid)
                else:  # rk4
                    v1 = get_v(x, t_vec)
                    t_half = x.new_full((x.shape[0],), t + 0.5 * dt_step)
                    t_full = x.new_full((x.shape[0],), t + dt_step)
                    v2 = get_v(x + 0.5 * dt_step * v1, t_half)
                    v3 = get_v(x + 0.5 * dt_step * v2, t_half)
                    v4 = get_v(x + dt_step * v3, t_full)
                    x = x + (dt_step / 6) * (v1 + 2 * v2 + 2 * v3 + v4)

                if traj is not None:
                    traj.append(x.clone())

            t = t_next

            # Apply waypoint if we landed exactly on it
            for i, (tq, qfn) in enumerate(waypoints):
                if not applied[i] and abs(t - tq) <= 1e-5:
                    is_gravity_mode = feature_clusters is not None and getattr(feature_clusters[i].config, "gravity_mode", False)
                    if is_gravity_mode:
                        # In gravity mode, we don't perform the hard jump. 
                        # We just run qfn(x) in case it's needed for state tracking, but keep x unchanged.
                        _ = qfn(x)
                    else:
                        x = qfn(x)
                        
                    applied[i] = True
                    if traj is not None and dt_step > 1e-6:
                        traj[-1] = x.clone()

        trajectory = torch.stack(traj, dim=1) if traj is not None else None
        return SolverOutput(x1=x, trajectory=trajectory)
