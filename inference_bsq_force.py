import argparse
import os
from pathlib import Path
import torch
from torchvision.utils import save_image

from modules.builder import build_model
from safetensors.torch import load_file

def load_model(checkpoint_dir: str, device: torch.device):
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    import json
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    model = build_model(cfg_dict)
    ckpt = os.path.join(checkpoint_dir, "model.safetensors")
    state = load_file(ckpt)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="bsq_force.png")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_code", type=int, default=-1, help="-1 for all neg, 1 for all pos, 0 for random")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, device)
    cfg = model.config
    
    # Check if BSQ
    if cfg.bsq_config is None:
        raise ValueError("Model does not use BSQ. Use inference_swap.py instead.")
    
    dim = cfg.bsq_config.dim
    assert dim == 64, "This script is designed for a 64-dim BSQ to make an 8x8 grid."
    
    # 1. Base x0
    shape = (1, cfg.in_channels, cfg.image_size, cfg.image_size)
    x0_base = torch.randn(*shape, device=device) * cfg.flow_config.source_std
    x0 = x0_base.repeat(64, 1, 1, 1)

    # 2. Base BSQ code
    if args.base_code == 0:
        # Random binary vector
        base_b = torch.randint(0, 2, (dim,), device=device).float() * 2 - 1
    else:
        # All -1 or all +1
        base_b = torch.ones((dim,), device=device) * args.base_code
        
    # 3. Create 64 variations
    # b_batch shape [64, 64]
    b_batch = base_b.unsqueeze(0).repeat(64, 1)
    
    # For i-th image, randomly flip 8 bits
    for i in range(64):
        flip_indices = torch.randperm(dim, device=device)[:8]
        b_batch[i, flip_indices] *= -1
        
    # Scale to unit sphere exactly how bsq.py does it
    z_q_forced = b_batch * (dim ** -0.5)

    # 4. Custom quantize_fn
    t_q = cfg.flow_config.t_q
    
    def quantize_fn(x: torch.Tensor) -> torch.Tensor:
        # Ignore whatever the encoder predicts and force our precomputed z_q
        v_tq_hat = model.velocity_ae.decode(z_q_forced)
        v_tq_hat = v_tq_hat.reshape(x.shape)
        return x0 + t_q * v_tq_hat
        
    velocity_fn = model._make_velocity_fn(x0)
    
    print("Running ODE solver with forced BSQ bits...")
    with torch.no_grad():
        out = model.solver.solve(
            x0=x0,
            velocity_fn=velocity_fn,
            num_steps=args.num_steps,
            t_q=t_q,
            quantize_fn=quantize_fn
        )
        
    # Save grid
    save_image((out.x1.clamp(-1, 1) + 1) / 2, args.out, nrow=8)
    print(f"Saved 8x8 grid showing effect of flipping each of the 64 bits to {args.out}")

if __name__ == "__main__":
    main()
