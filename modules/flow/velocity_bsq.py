import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityBSQWrapper(nn.Module):
    """Applies BSQ to the backbone's hidden representation (pre-projection) for t < threshold.

    If the backbone exposes forward_hidden() / out_proj (MLPBackbone), BSQ is applied
    in hidden_dim space (e.g. 256D → 2^256 possible directions) before the final
    linear projection to velocity space.

    Fallback (no forward_hidden): BSQ is applied to the 2D velocity output — kept
    for non-MLP backbones.

    BSQ (binary spherical quantization):
      h → normalize(h) → sign(·) * (1/√D)   with STE gradient
    """

    def __init__(self, backbone: nn.Module, threshold: float = 0.75) -> None:
        super().__init__()
        self.backbone = backbone
        self.threshold = threshold
        
        # Estraiamo le dimensioni per le proiezioni
        hidden_dim = backbone.config.hidden_dim if hasattr(backbone, 'config') and hasattr(backbone.config, 'hidden_dim') else 256
        data_dim = backbone.out_proj.out_features if hasattr(backbone, 'out_proj') else 2
        
        # Projection to 128D BSQ space, then to velocity
        self.proj_128 = nn.Linear(hidden_dim, 128)
        self.norm_128 = nn.LayerNorm(128)   # stabilizes scale → clean gradient flow through BSQ
        self.proj_out = nn.Linear(128, data_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_cond: torch.Tensor | None = None,
        force_continuous: bool = False,
    ) -> torch.Tensor:
        if force_continuous:
            apply = torch.zeros_like(t, dtype=torch.bool)
        else:
            apply = t < self.threshold   # [B] bool

        if hasattr(self.backbone, 'forward_hidden'):
            # 1. Estrai lo stato nascosto (256D)
            h = self.backbone.forward_hidden(x, t, x_cond=x_cond)
            
            # 2. Project to 128D, normalize scale so BSQ gets a stable gradient signal
            h_128 = self.norm_128(self.proj_128(h))

            if apply.any():
                h_f = h_128.float()
                D = h_f.shape[-1]
                h_norm = h_f / h_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                h_q = h_norm.sign() * (D ** -0.5)     # pure unit BSQ direction
                h_bsq = h_f + (h_q - h_f).detach()   # STE

                if apply.all():
                    h_128 = h_bsq.to(h_128.dtype)
                else:
                    mask = apply.view(-1, *([1] * (h_128.ndim - 1)))
                    h_128 = torch.where(mask, h_bsq.to(h_128.dtype), h_128)

            # 4. Riproietta verso la velocity finale (2D) bypassando l'out_proj originale
            return self.proj_out(h_128)

        else:
            # ── Fallback: magnitude-preserving BSQ on 2D velocity output ────
            v = self.backbone(x, t, x_cond=x_cond)
            if not apply.any():
                return v
            norm = v.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
            direction = v.float() / norm
            direction_q = direction + (direction.sign() - direction).detach()
            v_bsq = (direction_q * norm).to(v.dtype)
            if apply.all():
                return v_bsq
            mask = apply.view(-1, *([1] * (v.ndim - 1)))
            return torch.where(mask, v_bsq, v)
