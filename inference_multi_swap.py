#!/usr/bin/env python3
import argparse
import itertools
import os
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

# Aggiungo la root directory per importare i moduli correttamente
sys.path.insert(0, str(Path(__file__).parent))
from modules.builder import build_model


def load_model(checkpoint_dir: str, device: torch.device):
    import json
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    model = build_model(cfg_dict)
    ckpt = os.path.join(checkpoint_dir, "model.safetensors")
    from safetensors.torch import load_file
    state = load_file(ckpt)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


@torch.no_grad()
def sample_forced_multi(model, x0: torch.Tensor, target_codes: list[int]):
    """Force specific VQ codes at each waypoint."""
    cfg = model.config.flow_config
    t_qs = cfg.waypoints()
    quant_target = cfg.quant_target
    
    prev_state = {"x": x0, "t": 0.0}
    
    def make_qfn(i: int, tq: float):
        def qfn(x: torch.Tensor) -> torch.Tensor:
            B = x.shape[0]
            # Estrae l'embedding corrispondente al codice forzato per il waypoint `i`
            z_q_vec = model.quantizers[i].embedding.weight[target_codes[i]].unsqueeze(0).expand(B, -1)
            
            if quant_target == "velocity_ae":
                v_tq_hat = model.velocity_ae.decode(z_q_vec)
                v_tq_hat = v_tq_hat.reshape(x.shape)
            elif quant_target == "feature":
                d_out = model.dequantizers[i](z_q_vec)
                v_tq_hat = d_out.x_tq_hat.reshape(x.shape)
            else:
                v_tq_hat = z_q_vec.reshape(x.shape)
                
            dt = tq - prev_state["t"]
            x_hat = prev_state["x"] + dt * v_tq_hat
            prev_state["x"] = x_hat
            prev_state["t"] = tq
            return x_hat
        return qfn

    quantize_fns = [make_qfn(i, tq) for i, tq in enumerate(t_qs)]
    velocity_fn = model._make_velocity_fn(x0)

    sol = model.solver.solve(
        x0=x0,
        velocity_fn=velocity_fn,
        num_steps=cfg.num_steps,
        t_qs=t_qs,
        quantize_fns=quantize_fns,
        return_trajectory=False,
    )
    return sol.x1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="multi_swap.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    torch.manual_seed(args.seed)
    
    # 1. Genera un singolo sample rumore di partenza (x0)
    cfg = model.config
    if cfg.image_size is not None and cfg.in_channels is not None:
        shape = (1, cfg.in_channels, cfg.image_size, cfg.image_size)
    else:
        shape = (1, cfg.data_dim)
    
    x0 = torch.randn(*shape, device=device) * cfg.flow_config.source_std

    # Genera anche il sample "naturale" in modo da poterlo confrontare (senza forzare i token)
    natural_sample = model.sample(x0=x0).x1

    # 2. Ottieni tutte le combinazioni possibili
    num_embeddings = cfg.vq_config.num_embeddings
    t_qs = cfg.flow_config.waypoints()
    num_waypoints = len(t_qs)
    
    combinations = list(itertools.product(range(num_embeddings), repeat=num_waypoints))
    print(f"Total combinations (multi-waypoint codes): {len(combinations)}")
    
    results = []
    
    # 3. Per ogni combinazione, forza la generazione con quella tupla di codici
    for combo in combinations:
        x1 = sample_forced_multi(model, x0, combo)
        results.append(x1)
    
    results_tensor = torch.cat(results, dim=0)
    
    if cfg.image_size is not None:
        # Salva la griglia
        # Organizziamo la griglia con 9 colonne (così abbiamo 3 righe da 9 per le 27 combinazioni)
        save_image((results_tensor.clamp(-1, 1) + 1) / 2, args.out, nrow=9)
        print(f"Saved {len(combinations)} variants to {args.out}")

        # Salva a parte il campione naturale
        natural_out = args.out.replace(".png", "_natural.png")
        save_image((natural_sample.clamp(-1, 1) + 1) / 2, natural_out)
        print(f"Saved natural non-forced sample to {natural_out}")


if __name__ == "__main__":
    main()
