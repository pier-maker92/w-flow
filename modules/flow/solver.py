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
        dt = 1.0 / num_steps
        ts = torch.linspace(0.0, 1.0 - dt, num_steps, device=x0.device)

        traj = [x.clone()] if return_trajectory else None

        for t_scalar in ts:
            t_float = float(t_scalar)

            # Apply each waypoint the first time t crosses it
            for i, (tq, qfn) in enumerate(waypoints):
                if not applied[i] and t_float >= tq:
                    x = qfn(x)
                    applied[i] = True

            t_vec = x.new_full((x.shape[0],), t_scalar)

            if self.method == "euler":
                x = x + dt * velocity_fn(x, t_vec)
            elif self.method == "midpoint":
                v1 = velocity_fn(x, t_vec)
                x_mid = x + 0.5 * dt * v1
                t_mid = x.new_full((x.shape[0],), t_float + 0.5 * dt)
                x = x + dt * velocity_fn(x_mid, t_mid)
            else:  # rk4
                v1 = velocity_fn(x, t_vec)
                t_half = x.new_full((x.shape[0],), t_float + 0.5 * dt)
                t_full = x.new_full((x.shape[0],), t_float + dt)
                v2 = velocity_fn(x + 0.5 * dt * v1, t_half)
                v3 = velocity_fn(x + 0.5 * dt * v2, t_half)
                v4 = velocity_fn(x + dt * v3, t_full)
                x = x + (dt / 6) * (v1 + 2 * v2 + 2 * v3 + v4)

            if traj is not None:
                traj.append(x.clone())

        trajectory = torch.stack(traj, dim=1) if traj is not None else None
        return SolverOutput(x1=x, trajectory=trajectory)
