import math
import torch
import torch.nn as nn

from ..configs import MLPBackboneConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        half = embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / (half - 1))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] scalar times in [0, 1]
        args = t[:, None] * self.freqs[None, :]          # [B, half]
        emb = torch.cat([args.sin(), args.cos()], dim=-1) # [B, embed_dim]
        return self.proj(emb)                              # [B, embed_dim]


class MLPBackbone(nn.Module):
    """Lightweight MLP velocity network for Phase 1 (2D toy datasets)."""

    def __init__(self, config: MLPBackboneConfig, data_dim: int) -> None:
        super().__init__()
        self.config = config
        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim)
        # x_cond (x0) doubles the spatial input when enabled
        in_dim = data_dim * (2 if config.use_x0_cond else 1) + config.time_embed_dim
        layers: list[nn.Module] = []
        for i in range(config.num_layers):
            out_dim = config.hidden_dim if i < config.num_layers - 1 else data_dim
            layers.append(nn.Linear(in_dim if i == 0 else config.hidden_dim, out_dim))
            if i < config.num_layers - 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                x_cond: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, D], t: [B], x_cond: [B, D] optional (x0 memory)
        t_emb = self.time_embed(t)
        parts = [x, x_cond, t_emb] if x_cond is not None else [x, t_emb]
        return self.net(torch.cat(parts, dim=-1))

    def forward_hidden(self, x: torch.Tensor, t: torch.Tensor,
                       x_cond: torch.Tensor | None = None) -> torch.Tensor:
        """Hidden state before the final linear projection (hidden_dim)."""
        t_emb = self.time_embed(t)
        parts = [x, x_cond, t_emb] if x_cond is not None else [x, t_emb]
        h = torch.cat(parts, dim=-1)
        for layer in list(self.net)[:-1]:
            h = layer(h)
        return h

    @property
    def out_proj(self) -> nn.Linear:
        """Final linear projection: hidden_dim → data_dim."""
        return self.net[-1]
