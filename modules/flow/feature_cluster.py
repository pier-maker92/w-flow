"""Feature-space clustering waypoint.

Clusters the interpolation x_tq = (1-t_q)*x0 + t_q*x1 via K-Means,
then creates waypoint targets x_tq_hat = centroid_k + ε, ε ~ N(0, noise_std).

Per-cluster running statistics (mean, var) of x1 are tracked via EMA and
stored in register_buffers so that at inference we can denormalize the ODE
output back to the true data distribution.
"""

import torch
import torch.nn as nn

from ..configs import FeatureClusterConfig


class FeatureClusterWaypoint(nn.Module):
    """K-Means clustering in interpolation space with per-cluster x1 normalization."""

    def __init__(self, config: FeatureClusterConfig, data_dim: int, t_q: float) -> None:
        super().__init__()
        self.config = config
        self.t_q = t_q
        K = config.n_clusters
        D = data_dim

        # Gestione varianza apprendibile
        if getattr(config, "learnable_noise", False):
            import math
            init_val = math.log(config.noise_std) if config.noise_std > 0 else -10.0
            self.log_noise_std = nn.Parameter(torch.tensor(init_val))
        else:
            self.log_noise_std = None

        # Cluster centroids (updated via EMA)
        self.register_buffer("centroids", torch.zeros(K, D))
        self.register_buffer("initialized", torch.tensor(False))
        # Accumulate absolute mass (number of points assigned to each centroid)
        self.register_buffer("cluster_mass", torch.zeros(K))

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _init_centroids_kmeans_pp(self, x: torch.Tensor) -> None:
        """K-Means++ initialization from a batch of x_tq points."""
        K = self.config.n_clusters
        N = x.shape[0]
        device = x.device

        # Pick first centroid uniformly at random
        idx = torch.randint(0, N, (1,), device=device).item()
        centroids = [x[idx]]

        for _ in range(1, K):
            stacked = torch.stack(centroids, dim=0)  # [k, D]
            dists = torch.cdist(x.unsqueeze(0), stacked.unsqueeze(0)).squeeze(0)  # [N, k]
            min_dists = dists.min(dim=1).values  # [N]
            probs = min_dists / (min_dists.sum() + 1e-8)
            idx = torch.multinomial(probs, 1).item()
            centroids.append(x[idx])

        centroids_tensor = torch.stack(centroids, dim=0)
        
        # Calcola il raggio medio dei dati correnti per approssimare la circonferenza a t_q
        target_radius = torch.norm(x, p=2, dim=1).mean()
        
        # Proietta i centroidi iniziali sulla ipercirconferenza calcolata
        if getattr(self.config, "project_centroids", True):
            norms = torch.norm(centroids_tensor, p=2, dim=1, keepdim=True)
            centroids_tensor = torch.where(norms > 1e-8, centroids_tensor * (target_radius / norms), centroids_tensor)
        
        self.centroids.copy_(centroids_tensor)
        self.initialized.fill_(True)

    # ------------------------------------------------------------------
    # Assignment & EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _assign(self, x_tq: torch.Tensor) -> torch.LongTensor:
        """Assign each x_tq to nearest centroid. Returns [B] indices."""
        # [B, K]
        dists = torch.cdist(x_tq.unsqueeze(0), self.centroids.unsqueeze(0)).squeeze(0)
        return dists.argmin(dim=1)

    @torch.no_grad()
    def _ema_update_centroids(self, x_tq: torch.Tensor, assignments: torch.LongTensor) -> None:
        """EMA update of centroids based on current batch assignments."""
        decay = self.config.ema_decay
        K = self.config.n_clusters
        
        # Calcola il raggio medio globale per questo batch al tempo t_q
        target_radius = torch.norm(x_tq, p=2, dim=1).mean()

        for k in range(K):
            mask = assignments == k
            count_in_batch = mask.sum()
            self.cluster_mass[k] += count_in_batch
            
            if not mask.any():
                continue
            cluster_mean = x_tq[mask].mean(dim=0)
            new_centroid = decay * self.centroids[k] + (1 - decay) * cluster_mean
            
            # Proietta il centroide aggiornato sulla ipercirconferenza
            if getattr(self.config, "project_centroids", True):
                norm = torch.norm(new_centroid, p=2)
                if norm > 1e-8:
                    new_centroid = new_centroid * (target_radius / norm)
                
            self.centroids[k] = new_centroid

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        x_tq: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        """Cluster x_tq and produce noisy waypoint x_tq_hat.

        Args:
            x_tq: interpolation at t_q, shape [B, D].
            x1:   target data, shape [B, D].

        Returns:
            x_tq_hat:    waypoint = centroid_k + ε, shape [B, D].
            assignments: cluster indices, shape [B].
            commit_loss: MSE between x_tq and centroid, scalar tensor.
        """
        if not self.initialized:
            self._init_centroids_kmeans_pp(x_tq)

        assignments = self._assign(x_tq)

        if self.training:
            self._ema_update_centroids(x_tq, assignments)

        # x_tq_hat = centroid_k + N(0, noise_std)
        selected_centroids = self.centroids[assignments]  # [B, D]
        
        if getattr(self.config, "gravity_mode", False):
            # In gravity mode, we don't apply noise during forward clustering 
            # because the model learns the unperturbed path anyway.
            x_tq_hat = selected_centroids
        else:
            # Applica il rumore per generare il waypoint rumoroso
            if self.log_noise_std is not None:
                current_noise_std = torch.exp(self.log_noise_std)
            else:
                current_noise_std = self.config.noise_std

            if current_noise_std > 0:
                noise = torch.randn_like(selected_centroids) * current_noise_std
                x_tq_hat = selected_centroids + noise
            else:
                x_tq_hat = selected_centroids

        import torch.nn.functional as F
        commit_loss = F.mse_loss(x_tq, selected_centroids)

        return x_tq_hat, assignments, commit_loss

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def normalize_x1(self, x1: torch.Tensor, assignments: torch.LongTensor) -> torch.Tensor:
        """No-op: normalization removed."""
        return x1

    def denormalize(self, x1_norm: torch.Tensor, assignments: torch.LongTensor) -> torch.Tensor:
        """No-op: denormalization removed."""
        return x1_norm

    @torch.no_grad()
    def assign_and_sample(self, x_tq: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:
        """Inference-time: assign to nearest centroid and sample noisy waypoint.

        Args:
            x_tq: current ODE state at t_q, shape [B, D].

        Returns:
            x_tq_hat:    centroid_k + N(0, noise_std), shape [B, D].
            assignments: cluster indices, shape [B].
        """
        assignments = self._assign(x_tq)
        selected_centroids = self.centroids[assignments]
        noise = torch.randn_like(selected_centroids) * self.config.noise_std
        x_tq_hat = selected_centroids + noise
        return x_tq_hat, assignments
