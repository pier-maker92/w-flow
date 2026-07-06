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
        x = x0.clone()
        t = 0.0

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

                if self.method == "euler":
                    x = x + dt_step * velocity_fn(x, t_vec)
                elif self.method == "midpoint":
                    v1 = velocity_fn(x, t_vec)
                    x_mid = x + 0.5 * dt_step * v1
                    t_mid = x.new_full((x.shape[0],), t + 0.5 * dt_step)
                    x = x + dt_step * velocity_fn(x_mid, t_mid)
                else:  # rk4
                    v1 = velocity_fn(x, t_vec)
                    t_half = x.new_full((x.shape[0],), t + 0.5 * dt_step)
                    t_full = x.new_full((x.shape[0],), t + dt_step)
                    v2 = velocity_fn(x + 0.5 * dt_step * v1, t_half)
                    v3 = velocity_fn(x + 0.5 * dt_step * v2, t_half)
                    v4 = velocity_fn(x + dt_step * v3, t_full)
                    x = x + (dt_step / 6) * (v1 + 2 * v2 + 2 * v3 + v4)

                if traj is not None:
                    traj.append(x.clone())

            t = t_next

            # Apply waypoint if we landed exactly on it
            for i, (tq, qfn) in enumerate(waypoints):
                if not applied[i] and abs(t - tq) <= 1e-5:
                    x = qfn(x)
                    applied[i] = True
                    if traj is not None and dt_step > 1e-6:
                        traj[-1] = x.clone()

        trajectory = torch.stack(traj, dim=1) if traj is not None else None
        return SolverOutput(x1=x, trajectory=trajectory)
