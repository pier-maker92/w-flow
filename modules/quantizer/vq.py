import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import VQConfig
from ..output_dataclasses import QuantizerOutput


class VectorQuantizer(nn.Module):
    """Standard VQ with optional EMA codebook updates."""

    def __init__(self, config: VQConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.num_embeddings, config.embedding_dim)
        scale = config.embedding_init_scale if config.embedding_init_scale is not None \
            else 1.0 / config.num_embeddings
        nn.init.uniform_(self.embedding.weight, -scale, scale)
        if config.use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(config.num_embeddings))
            self.register_buffer("ema_embed_avg", self.embedding.weight.data.clone())

    def forward(self, z: torch.FloatTensor) -> QuantizerOutput:
        # z: [B, D] or [B, T, D]
        original_shape = z.shape
        flat = z.reshape(-1, self.config.embedding_dim)   # [N, D]
        flat_f32 = flat.float()                           # codebook always fp32

        # Compute L2 distances to all codebook entries
        distances = (
            flat_f32.pow(2).sum(1, keepdim=True)         # [N, 1]
            - 2 * flat_f32 @ self.embedding.weight.T      # [N, K]
            + self.embedding.weight.pow(2).sum(1)         # [K]
        )                                                  # [N, K]
        indices = distances.argmin(1)                      # [N]
        z_q_flat = self.embedding(indices)                 # [N, D] fp32

        if self.training and self.config.use_ema:
            self._ema_update(flat_f32, indices)
            commitment_loss = F.mse_loss(z_q_flat.detach(), flat_f32)
            codebook_loss = None
        else:
            commitment_loss = F.mse_loss(z_q_flat.detach(), flat_f32)
            codebook_loss = F.mse_loss(z_q_flat, flat_f32.detach())

        # Straight-through estimator: gradients flow through z (original dtype)
        z_q = flat + (z_q_flat.to(flat.dtype) - flat).detach()
        z_q = z_q.reshape(original_shape)
        indices = indices.reshape(original_shape[:-1])

        encodings = F.one_hot(indices.reshape(-1), self.config.num_embeddings).float()
        avg_probs = encodings.mean(0)
        perplexity = (-avg_probs * (avg_probs + 1e-10).log()).sum().exp()

        return QuantizerOutput(
            z_q=z_q,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            indices=indices,
            perplexity=perplexity,
        )

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.LongTensor) -> None:
        decay = 0.99
        K = self.config.num_embeddings
        encodings = F.one_hot(indices, K).float()          # [N, K]
        cluster_size = encodings.sum(0)                    # [K]
        embed_sum = encodings.T @ flat                     # [K, D]
        self.ema_cluster_size = decay * self.ema_cluster_size + (1 - decay) * cluster_size
        self.ema_embed_avg = decay * self.ema_embed_avg + (1 - decay) * embed_sum
        # Laplace smoothing
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + 1e-5) / (n + K * 1e-5) * n
        self.embedding.weight.data = self.ema_embed_avg / smoothed.unsqueeze(1)
        # Restart dead codes: replace with random encoder outputs from current batch
        dead = self.ema_cluster_size < 1.0
        num_dead = int(dead.sum().item())
        if num_dead > 0:
            rand_idx = torch.randint(0, flat.shape[0], (num_dead,), device=flat.device)
            self.embedding.weight.data[dead] = flat[rand_idx]
            self.ema_cluster_size[dead] = 1.0
            self.ema_embed_avg[dead] = flat[rand_idx]
