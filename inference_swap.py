#!/usr/bin/env python3
"""
inference_swap.py — Visualize FlowQuant generation + token swap editing.

Usage:
  python inference_swap.py \
    --exp_dir /scratch/piermel/agente/reports/flow_quant_phase1/phase1-gaussians-ema-8codes \
    --n_samples 512 \
    --sigma 0.0 \
    --out /scratch/piermel/agente/plots/token_swap.png

Produces a 4-panel figure:
  [A] Generated x1 colored by VQ code (+ codebook vectors)
  [B] Trajectories: x0 → x_tq_raw → z_q → x1 for a few samples per code
  [C] Token swap grid: row = original code, col = forced code, scatter of x1
  [D] Print of codebook vectors and per-code stats
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from modules.builder import build_model
from modules.quantizer.vq import VectorQuantizer
from modules.dequantizer.stochastic import StochasticDequantizer
from modules.dequantizer.residual import ResidualDequantizer
from data.toy_2d import GaussiansDataset


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(exp_dir: str, device: torch.device) -> tuple:
    import json
    exp_dir = Path(exp_dir)

    cfg_path = exp_dir / "config.json"
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    # Build model skeleton
    model = build_model(cfg_dict)
    model.eval()

    ckpt = exp_dir / "model.safetensors"
    from safetensors.torch import load_file
    state = load_file(ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)

    return model, cfg_dict


# ---------------------------------------------------------------------------
# Custom sampling with waypoint capture
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_with_info(model, x0: torch.Tensor, sigma: float = 0.0):
    """
    Run ODE with bottleneck. Returns:
      x_tq_raw  — state just before quantize_fn (pre-VQ)
      codes     — assigned VQ code indices [B]
      z_q       — codebook vectors assigned to each sample [B, D]
      x_tq_hat  — dequantized state (post-VQ) [B, D]
      x1        — final generated sample [B, D]
    """
    cfg = model.config.flow_config
    t_q = cfg.t_q

    captured = {}

    def quantize_fn(x):
        captured["x_tq_raw"] = x.clone()
        q_out = model.quantizer(x)
        captured["codes"] = q_out.indices.clone()
        captured["z_q"] = model.quantizer.embedding(q_out.indices).clone()

        # Dequantize — override sigma if requested
        if isinstance(model.dequantizer, StochasticDequantizer):
            noise = torch.randn_like(q_out.z_q) * sigma
            x_hat = q_out.z_q + noise
        elif isinstance(model.dequantizer, ResidualDequantizer):
            alpha = model.dequantizer.config.alpha
            x_hat = (1 - alpha) * q_out.z_q + alpha * x
        else:
            x_hat = q_out.z_q

        captured["x_tq_hat"] = x_hat.clone()
        return x_hat

    num_steps = cfg.num_steps
    velocity_fn = model._make_velocity_fn(x0)
    sol = model.solver.solve(
        x0=x0,
        velocity_fn=velocity_fn,
        num_steps=num_steps,
        t_q=t_q,
        quantize_fn=quantize_fn if t_q is not None else None,
    )

    return {
        "x0": x0,
        "x_tq_raw": captured.get("x_tq_raw"),
        "codes": captured.get("codes"),
        "z_q": captured.get("z_q"),
        "x_tq_hat": captured.get("x_tq_hat"),
        "x1": sol.x1,
    }


@torch.no_grad()
def sample_forced_code(model, x0: torch.Tensor, target_code: int, sigma: float = 0.0):
    """
    Like sample_with_info but forces the VQ code to target_code for all samples.
    Demonstrates token swap editing.
    """
    cfg = model.config.flow_config
    t_q = cfg.t_q
    quant_target = cfg.quant_target

    def quantize_fn(x):
        B = x.shape[0]
        z_q_vec = model.quantizer.embedding.weight[target_code].unsqueeze(0).expand(B, -1)

        if isinstance(model.dequantizer, StochasticDequantizer):
            noise = torch.randn_like(z_q_vec) * sigma
            z_q_vec = z_q_vec + noise
        elif isinstance(model.dequantizer, ResidualDequantizer):
            alpha = model.dequantizer.config.alpha
            # Approximation for forced code
            z_q_vec = (1 - alpha) * z_q_vec + alpha * z_q_vec

        if quant_target == "velocity_ae":
            v_tq_hat = model.velocity_ae.decode(z_q_vec)
            v_tq_hat = v_tq_hat.reshape(x.shape)
            x_hat = x0 + t_q * v_tq_hat
        elif quant_target == "feature":
            if model.dequantizer is not None:
                # Stochastic/Residual applied above
                pass
            # Just simple reshape for demo
            v_tq_hat = z_q_vec.reshape(x.shape)
            x_hat = x0 + t_q * v_tq_hat
        else:
            x_hat = z_q_vec.reshape(x.shape)

        return x_hat

    velocity_fn = model._make_velocity_fn(x0)
    sol = model.solver.solve(
        x0=x0,
        velocity_fn=velocity_fn,
        num_steps=cfg.num_steps,
        t_q=t_q,
        quantize_fn=quantize_fn if t_q is not None else None,
    )
    return sol.x1


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

try:
    PALETTE = plt.cm.get_cmap("tab10")
except AttributeError:
    PALETTE = plt.colormaps.get_cmap("tab10")


def code_colors(codes: torch.Tensor, K: int) -> list:
    return [PALETTE(int(c) / K) for c in codes.cpu().numpy()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Dequantizer noise at inference (0 = deterministic)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, cfg_dict = load_model(args.exp_dir, device)
    t_q = model.config.flow_config.t_q
    K = model.config.vq_config.num_embeddings  # number of codes

    print(f"t_q={t_q}, K={K}, sigma={args.sigma}")

    # Sample noise
    torch.manual_seed(0)
    
    # 4 samples if image grid is requested, otherwise use args.n_samples
    n_samples = 4 if model.config.image_size is not None else args.n_samples

    if model.config.image_size is not None and model.config.in_channels is not None:
        shape = (n_samples, model.config.in_channels, model.config.image_size, model.config.image_size)
    else:
        shape = (n_samples, model.config.data_dim)
        
    x0 = torch.randn(*shape, device=device) * model.config.flow_config.source_std

    # --- Normal generation with waypoint capture ---
    info = sample_with_info(model, x0, sigma=args.sigma)
    x1 = info["x1"].cpu().numpy() if model.config.image_size is None else info["x1"].cpu()
    codes = info["codes"].cpu().numpy()
    if info["x_tq_raw"] is not None:
        x_tq_raw = info["x_tq_raw"].cpu().numpy() if model.config.image_size is None else None
    if info["z_q"] is not None:
        z_q = info["z_q"].cpu().numpy() if model.config.image_size is None else None
    if info["x_tq_hat"] is not None:
        x_tq_hat = info["x_tq_hat"].cpu().numpy() if model.config.image_size is None else None

    # Codebook vectors
    codebook = model.quantizer.embedding.weight.detach().cpu().numpy()

    # --- Unique codes actually used ---
    used_codes = np.unique(codes)
    print(f"Codes used: {sorted(used_codes.tolist())} ({len(used_codes)}/{K})")
    if model.config.image_size is None:
        for k in used_codes:
            mask = codes == k
            cx, cy = x1[mask, 0].mean(), x1[mask, 1].mean()
            print(f"  code {k:2d}: n={mask.sum():4d}  mean_x1=({cx:.2f},{cy:.2f})"
                  f"  codebook=({codebook[k,0]:.2f},{codebook[k,1]:.2f})")

    # --- Token swap: for each target code, generate from all x0 ---
    swap_results = {}
    for k in range(K):
        swap_results[k] = sample_forced_code(model, x0, target_code=k,
                                             sigma=args.sigma).cpu()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if model.config.image_size is not None:
        from torchvision.utils import save_image
        
        # Grid layout: 10 rows (codes), 4 columns (samples)
        # We will collect all 40 generated images in a list
        grid_images = []
        for k in range(K):
            # For each code k, append the 4 images
            grid_images.append((swap_results[k].clamp(-1, 1) + 1) / 2)
            
        grid_tensor = torch.cat(grid_images, dim=0) # [40, C, H, W]
        save_image(grid_tensor, args.out, nrow=n_samples)
        print(f"Saved image grid ({K}x{n_samples}) → {args.out}")
        return

    # -----------------------------------------------------------------------
    # Figure layout: 2 rows × 3 cols (Only for 2D Toy Data)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f"FlowQuant inference — {Path(args.exp_dir).name}\n"
        f"t_q={t_q}, K={K}, σ={args.sigma}",
        fontsize=13,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # --- Panel A: Generated x1 colored by code ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title("A — x₁ generated (colored by VQ code)")
    for k in range(K):
        mask = codes == k
        if mask.sum() > 0:
            ax_a.scatter(x1[mask, 0], x1[mask, 1], s=6, alpha=0.5,
                         color=PALETTE(k / K), label=f"code {k}")
    # Codebook vectors as large stars
    ax_a.scatter(codebook[:, 0], codebook[:, 1], s=200, marker="*",
                 c=[PALETTE(k / K) for k in range(K)], edgecolors="black",
                 linewidths=0.8, zorder=5, label="codebook")
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("x"); ax_a.set_ylabel("y")

    # --- Panel B: Waypoint anatomy (x0 → x_tq → z_q → x1) for subset ---
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title(f"B — Waypoints (x₀→x_tq→z_q→x₁) at t_q={t_q}")
    n_show = min(5, args.n_samples // K)
    for k in used_codes[:8]:
        idx = np.where(codes == k)[0][:n_show]
        col = PALETTE(k / K)
        for i in idx:
            pts = np.stack([x0[i].numpy(), x_tq_raw[i], z_q[i], x1[i]])
            ax_b.plot(pts[:, 0], pts[:, 1], "-o", color=col, alpha=0.6,
                      markersize=3, linewidth=0.8)
    ax_b.scatter(codebook[:, 0], codebook[:, 1], s=200, marker="*",
                 c=[PALETTE(k / K) for k in range(K)], edgecolors="black",
                 linewidths=0.8, zorder=5)
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("x"); ax_b.set_ylabel("y")
    labels = ["x₀", f"x_tq raw\n(pre-VQ)", "z_q (post-VQ)", "x₁"]
    for lbl, marker in zip(labels, [".", "^", "D", "s"]):
        ax_b.plot([], [], marker=marker, ls="none", color="gray",
                  label=lbl, markersize=5)
    ax_b.legend(fontsize=7, loc="upper right")

    # --- Panel C: Token swap — x1 when forcing each code ---
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_title("C — Token swap: x₁ with forced code k")
    for k in range(K):
        x1_swap = swap_results[k].numpy()
        ax_c.scatter(x1_swap[:, 0], x1_swap[:, 1], s=3, alpha=0.3,
                     color=PALETTE(k / K))
    ax_c.scatter(codebook[:, 0], codebook[:, 1], s=200, marker="*",
                 c=[PALETTE(k / K) for k in range(K)], edgecolors="black",
                 linewidths=0.8, zorder=5)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("x"); ax_c.set_ylabel("y")
    ax_c.set_title(f"C — Swap: all x0 → forced code k")

    # --- Panel D: Per-code swap comparison (4 codes, 2x2 subgrid) ---
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, :])
    show_codes = used_codes[:4] if len(used_codes) >= 4 else used_codes

    for idx_panel, k in enumerate(show_codes):
        ax_d = fig.add_subplot(inner_gs[idx_panel // 2, idx_panel % 2])
        # Original x1 for this code's samples
        mask_orig = codes == k
        x1_orig = x1[mask_orig]
        ax_d.scatter(x1_orig[:, 0], x1_orig[:, 1], s=8, alpha=0.6,
                     color=PALETTE(k / K), label=f"natural code {k}")
        # Swapped outputs: all samples forced to each OTHER code
        for k2 in range(K):
            if k2 == k:
                continue
            x1_swap = swap_results[k2][mask_orig].numpy()
            ax_d.scatter(x1_swap[:, 0], x1_swap[:, 1], s=4, alpha=0.3,
                         color=PALETTE(k2 / K), marker="x")
        ax_d.scatter(codebook[:, 0], codebook[:, 1], s=80, marker="*",
                     c=[PALETTE(j / K) for j in range(K)], edgecolors="black",
                     linewidths=0.5, zorder=5)
        ax_d.set_aspect("equal")
        ax_d.set_title(
            f"D{idx_panel+1} — src code {k} (●) → swap to k' (×)",
            fontsize=9,
        )
        ax_d.set_xlabel("x"); ax_d.set_ylabel("y")

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
