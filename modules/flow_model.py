from typing import Optional, Callable, List
import torch
import torch.nn as nn

from .configs import FlowQuantConfig
from .output_dataclasses import FlowOutput, SolverOutput, QuantizerOutput, DequantizerOutput
from .backbone.mlp import MLPBackbone
from .backbone.unet import UNetBackbone
from .backbone.dit import DiTBackbone
from .quantizer.vq import VectorQuantizer
from .quantizer.fsq import FSQ
from .quantizer.bsq import BinarySphericalQuantizer
from .dequantizer.stochastic import StochasticDequantizer
from .dequantizer.residual import ResidualDequantizer
from .dequantizer.linear import LinearDequantizer
from .flow.rectified_flow import RectifiedFlow
from .flow.solver import ODESolver
from .flow.velocity_quant import VelocityQuantizedFlow
from .flow.velocity_bsq import VelocityBSQWrapper
from .flow.feature_cluster import FeatureClusterWaypoint
from .autoencoder.velocity_ae import VelocityAutoencoder


class FlowQuant(nn.Module):
    def __init__(self, config: FlowQuantConfig) -> None:
        super().__init__()
        self.config = config

        # Backbone (exactly one non-null)
        if config.mlp_backbone_config is not None:
            self.backbone: nn.Module = MLPBackbone(config.mlp_backbone_config, config.data_dim)
        elif config.unet_backbone_config is not None:
            self.backbone = UNetBackbone(config.unet_backbone_config, config.in_channels)
        else:
            self.backbone = DiTBackbone(config.dit_backbone_config, config.in_channels, config.image_size)

        # Feature-cluster waypoints (new approach, replaces VQ when set)
        fc_cfgs = config.feature_cluster_configs
        self._uses_feature_cluster = False
        if fc_cfgs is not None and len(fc_cfgs) > 0:
            self._uses_feature_cluster = True
            t_qs = config.flow_config.waypoints()
            self.feature_clusters = nn.ModuleList([
                FeatureClusterWaypoint(fc, config.data_dim, t_q=t_qs[i]) for i, fc in enumerate(fc_cfgs)
            ])
        else:
            self.feature_clusters: Optional[nn.ModuleList] = None

        # Quantizer / dequantizer
        n_waypoints = len(config.flow_config.waypoints())
        self._is_multi = n_waypoints > 1

        if self._uses_feature_cluster:
            # Feature-cluster mode: no VQ / dequantizer needed
            self.quantizer: Optional[nn.Module] = None
            self.dequantizer: Optional[nn.Module] = None
            self.quantizers: Optional[nn.ModuleList] = None
            self.dequantizers: Optional[nn.ModuleList] = None
        elif self._is_multi:
            # Multi-waypoint (VQ-based)
            quant_target = config.flow_config.quant_target
            self.quantizer = None
            self.dequantizer = None
            
            if quant_target == "feature":
                assert config.vq_config is not None, "vq_config required for multi-waypoint"
                assert config.linear_dequantizer_config is not None, "linear_dequantizer required for multi-waypoint feature"
                self.quantizers = nn.ModuleList([
                    VectorQuantizer(config.vq_config) for _ in range(n_waypoints)
                ])
                self.dequantizers = nn.ModuleList([
                    LinearDequantizer(config.linear_dequantizer_config)
                    for _ in range(n_waypoints)
                ])
                self.velocity_aes = None
            elif quant_target == "velocity_ae":
                assert config.vq_config is not None, "vq_config required for multi-waypoint"
                assert config.velocity_ae_config is not None, "velocity_ae_config required for multi-waypoint velocity_ae"
                self.quantizers = nn.ModuleList([
                    VectorQuantizer(config.vq_config) for _ in range(n_waypoints)
                ])
                self.velocity_ae = VelocityAutoencoder(config.velocity_ae_config, config.in_channels, config.image_size)
                self.dequantizers = None
            else:
                raise ValueError(f"Unsupported multi-waypoint target: {quant_target}")
        else:
            self.quantizers: Optional[nn.ModuleList] = None
            self.dequantizers: Optional[nn.ModuleList] = None
            self.quantizer = None
            if config.vq_config is not None:
                self.quantizer = VectorQuantizer(config.vq_config)
            elif config.fsq_config is not None:
                self.quantizer = FSQ(config.fsq_config)
            elif config.bsq_config is not None:
                self.quantizer = BinarySphericalQuantizer(config.bsq_config)

            self.dequantizer = None
            if config.stochastic_dequantizer_config is not None:
                self.dequantizer = StochasticDequantizer(config.stochastic_dequantizer_config)
            elif config.residual_dequantizer_config is not None:
                self.dequantizer = ResidualDequantizer(config.residual_dequantizer_config)
            elif config.linear_dequantizer_config is not None:
                self.dequantizer = LinearDequantizer(config.linear_dequantizer_config)

        # Velocity Autoencoder
        self.velocity_ae: Optional[nn.Module] = None
        if config.velocity_ae_config is not None:
            self.velocity_ae = VelocityAutoencoder(
                config.velocity_ae_config,
                in_channels=config.in_channels if config.in_channels is not None else 1,
                image_size=config.image_size if config.image_size is not None else 28
            )

        # Flow loss and solver
        self.flow = RectifiedFlow()
        self.solver = ODESolver(config.flow_config.solver)

        # Velocity quantization wrapper (§4) — wraps backbone
        if config.use_velocity_quant:
            self.backbone = VelocityQuantizedFlow(self.backbone, self.quantizer)
            self.quantizer = None
            self._last_quant_output = None

        # Inline velocity BSQ (C1-style) — wraps backbone after any other wrapping
        if config.use_velocity_bsq:
            self.backbone = VelocityBSQWrapper(self.backbone, config.velocity_bsq_threshold)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, x1: torch.FloatTensor) -> FlowOutput:
        cfg = self.config.flow_config
        if self.config.mlp_backbone_config is not None and x1.ndim > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        x0 = torch.randn_like(x1) * cfg.source_std
        t = torch.rand(x1.shape[0], device=x1.device)

        if self._uses_feature_cluster:
            return self._forward_with_feature_cluster(x0, x1, t, x0_cond=x0)
        elif self._is_multi:
            return self._forward_with_multi_bottleneck(x0, x1, t, x0_cond=x0)
        elif self.config.use_velocity_quant or cfg.t_q is None:
            return self._forward_no_bottleneck(x0, x1, t)
        else:
            return self._forward_with_bottleneck(x0, x1, t, x0_cond=x0)

    def _forward_no_bottleneck(
        self,
        x0: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> FlowOutput:
        velocity_fn = self._make_velocity_fn(x0)
        fm_loss, quant_out = self.flow.loss(x0, x1, t, velocity_fn)
        if quant_out is None and hasattr(self.backbone, "last_quant_output"):
            quant_out = self.backbone.last_quant_output
        commitment_loss = None
        codebook_loss = None
        if quant_out is not None:
            commitment_loss = quant_out.commitment_loss
            codebook_loss = quant_out.codebook_loss

        total = fm_loss
        if commitment_loss is not None:
            total = total + self.config.flow_config.commitment_weight * commitment_loss
        if codebook_loss is not None:
            total = total + codebook_loss

        return FlowOutput(
            loss=total,
            fm_loss=fm_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            quantizer_output=quant_out,
        )

    def _forward_with_bottleneck(
        self,
        x0: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
        x0_cond: torch.FloatTensor | None = None,
    ) -> FlowOutput:
        t_q = self.config.flow_config.t_q
        pre_mask = t < t_q
        post_mask = ~pre_mask

        x_tq = (1 - t_q) * x0 + t_q * x1
        quant_target = self.config.flow_config.quant_target
        t_q_vec = x0.new_full((x0.shape[0],), t_q)
        
        ae_loss = None

        if quant_target == "velocity":
            cond = x0_cond if (self._use_x0_cond and x0_cond is not None) else None
            v_tq = self.backbone(x_tq, t_q_vec, x_cond=cond)
            v_tq_flat = v_tq.reshape(v_tq.shape[0], -1) if v_tq.ndim > 2 else v_tq
            quant_out = self.quantizer(v_tq_flat)
            if self.dequantizer is not None:
                d_out = self.dequantizer(quant_out.z_q)
                v_tq_hat = d_out.x_tq_hat.reshape(x_tq.shape)
            else:
                v_tq_hat = quant_out.z_q.reshape(x_tq.shape)
            x_tq_hat = x0 + t_q * v_tq_hat

        elif quant_target == "velocity_ae":
            assert self.velocity_ae is not None
            v_gt = x1 - x0
            # Ensure v_gt has image shape [B, C, H, W]
            v_gt_spatial = v_gt.reshape(x_tq.shape) if v_gt.ndim <= 2 else v_gt
            z = self.velocity_ae.encode(v_gt_spatial)
            quant_out = self.quantizer(z)
            v_tq_hat = self.velocity_ae.decode(quant_out.z_q)
            v_tq_hat = v_tq_hat.reshape(x_tq.shape)
            x_tq_hat = x0 + t_q * v_tq_hat
            # Compute AE reconstruction loss
            ae_loss = torch.nn.functional.mse_loss(v_tq_hat, v_gt_spatial)

        elif quant_target == "feature":
            t_emb = self.backbone.time_embed(t_q_vec)
            x_aug = torch.cat([x_tq, x0_cond], dim=1) \
                if (self._use_x0_cond and x0_cond is not None) else x_tq
            z, _skips = self.backbone._encode(x_aug, t_emb)
            z_pool = z.mean(dim=[2, 3])
            quant_out = self.quantizer(z_pool)
            dequant_out = self.dequantizer(quant_out.z_q)
            v_tq_hat = dequant_out.x_tq_hat.reshape(x_tq.shape)
            x_tq_hat = x0 + t_q * v_tq_hat

        else:  # "x_tq" — VQ in data space
            x_tq_vq = x_tq.reshape(x_tq.shape[0], -1) if x_tq.ndim > 2 else x_tq
            quant_out = self.quantizer(x_tq_vq)
            z_q = quant_out.z_q.reshape(x_tq.shape)
            if self.dequantizer is not None:
                dequant_out = self._dequantize(z_q, x_tq)
                x_tq_hat = dequant_out.x_tq_hat
            else:
                x_tq_hat = z_q   # codebook vector used directly as waypoint

        total_fm_loss = x0.new_zeros(())

        if pre_mask.any():
            t_pre = t[pre_mask]
            x0_pre, x1_pre = x0[pre_mask], x1[pre_mask]
            x_t_pre = (1 - t_pre.view(-1, *([1] * (x0.ndim - 1)))) * x0_pre \
                      + t_pre.view(-1, *([1] * (x0.ndim - 1))) * x1_pre
            cond_pre = x0_cond[pre_mask] if (self._use_x0_cond and x0_cond is not None) else None
            v_pred_pre = self.backbone(x_t_pre, t_pre, x_cond=cond_pre)
            v_target_pre = x1_pre - x0_pre
            total_fm_loss = total_fm_loss + (v_pred_pre - v_target_pre).pow(2).mean()

        if post_mask.any():
            t_post = t[post_mask]
            x1_post = x1[post_mask]
            x_tq_hat_post = x_tq_hat[post_mask]
            s = (t_post - t_q) / (1 - t_q)
            x_t_post = (1 - s.view(-1, *([1] * (x0.ndim - 1)))) * x_tq_hat_post \
                       + s.view(-1, *([1] * (x0.ndim - 1))) * x1_post
            cond_post = x0_cond[post_mask] if (self._use_x0_cond and x0_cond is not None) else None
            v_pred_post = self.backbone(x_t_post, t_post, x_cond=cond_post)
            v_target_post = (x1_post - x_tq_hat_post) / (1 - t_q)
            total_fm_loss = total_fm_loss + (v_pred_post - v_target_post).pow(2).mean()

        commitment_loss = quant_out.commitment_loss
        codebook_loss = quant_out.codebook_loss
        total = total_fm_loss
        if ae_loss is not None:
            total = total + ae_loss
        if commitment_loss is not None:
            total = total + self.config.flow_config.commitment_weight * commitment_loss
        if codebook_loss is not None:
            total = total + codebook_loss

        return FlowOutput(
            loss=total,
            fm_loss=total_fm_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            ae_loss=ae_loss,
            quantizer_output=quant_out,
        )

    def _forward_with_multi_bottleneck(
        self,
        x0: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
        x0_cond: torch.FloatTensor | None = None,
    ) -> FlowOutput:
        """Training forward for multi-waypoint feature-VQ (quant_target='feature').

        Computes ground-truth x̂ at each waypoint, then computes FM loss for
        whichever trajectory segment each training sample t falls into.

        Segments (N=3 waypoints example):
          [0, t_q0]   → source = x0
          [t_q0, t_q1] → source = x̂_tq0
          [t_q1, t_q2] → source = x̂_tq1
          [t_q2, 1]   → source = x̂_tq2
        """
        t_qs: List[float] = self.config.flow_config.waypoints()
        ndim_m1 = x0.ndim - 1

        # --- compute x_hat at each waypoint accumulatively ---
        x_hats: List[torch.Tensor] = []
        all_commit: List[torch.Tensor] = []
        all_codebook: List[torch.Tensor] = []
        all_ae: List[torch.Tensor] = []

        current_x = x0
        current_t = 0.0

        for i, tq in enumerate(t_qs):
            quant_target = self.config.flow_config.quant_target
            
            # Ground truth velocity from current_x to x1
            v_gt_remaining = (x1 - current_x) / (1.0 - current_t)

            if quant_target == "feature":
                t_q_vec = x0.new_full((x0.shape[0],), tq)
                t_emb = self.backbone.time_embed(t_q_vec)
                x_tq = current_x + (tq - current_t) * v_gt_remaining
                x_aug = (torch.cat([x_tq, x0_cond], dim=1)
                         if (self._use_x0_cond and x0_cond is not None) else x_tq)
                z, _ = self.backbone._encode(x_aug, t_emb)
                z_pool = z.mean(dim=[2, 3])
                q_out = self.quantizers[i](z_pool)
                d_out = self.dequantizers[i](q_out.z_q)
                v_tq_hat = d_out.x_tq_hat.reshape(x_tq.shape)
                
                current_x = current_x + (tq - current_t) * v_tq_hat
                x_hats.append(current_x)
                current_t = tq
            elif quant_target == "velocity_ae":
                v_gt_spatial = v_gt_remaining.reshape(current_x.shape) if v_gt_remaining.ndim <= 2 else v_gt_remaining
                ae = self.velocity_ae
                z = ae.encode(v_gt_spatial)
                q_out = self.quantizers[i](z)
                v_tq_hat = ae.decode(q_out.z_q)
                v_tq_hat = v_tq_hat.reshape(current_x.shape)
                
                current_x = current_x + (tq - current_t) * v_tq_hat
                x_hats.append(current_x)
                current_t = tq
                
                v_pred_ae = ae.decode(z)
                v_pred_ae = v_pred_ae.reshape(v_gt_remaining.shape)
                ae_loss = (v_pred_ae - v_gt_remaining).pow(2).mean()
                all_ae.append(ae_loss)
                
            if q_out.commitment_loss is not None:
                all_commit.append(q_out.commitment_loss)
            if q_out.codebook_loss is not None:
                all_codebook.append(q_out.codebook_loss)

        # --- FM loss per segment ---
        # N waypoints → N+1 segments
        # boundaries: [0, t_q0, ..., t_qN-1, 1]
        # sources:    [x0,  x̂0, ..., x̂N-2,   x̂N-1]
        t_lows  = [0.0] + list(t_qs)
        t_highs = list(t_qs) + [1.0]
        sources = [x0] + x_hats          # length N+1

        total_fm_loss = x0.new_zeros(())

        for seg_idx, (t_lo, t_hi, x_src) in enumerate(zip(t_lows, t_highs, sources)):
            is_last = (seg_idx == len(t_lows) - 1)
            mask = (t >= t_lo) & (t <= t_hi if is_last else t < t_hi)
            if not mask.any():
                continue

            t_seg  = t[mask]
            x1_seg = x1[mask]
            x_src_seg = x_src[mask]
            cond_seg = (x0_cond[mask]
                        if (self._use_x0_cond and x0_cond is not None) else None)

            if t_lo == 0.0:
                # First segment: standard rectified flow x0 → x1
                t_v = t_seg.view(-1, *([1] * ndim_m1))
                x_t = (1 - t_v) * x_src_seg + t_v * x1_seg
                v_pred   = self.backbone(x_t, t_seg, x_cond=cond_seg)
                v_target = x1_seg - x_src_seg          # = x1 - x0
            else:
                # Post-waypoint segment: flow from x̂_tq_lo → x1
                s = (t_seg - t_lo) / (1.0 - t_lo)
                s_v = s.view(-1, *([1] * ndim_m1))
                x_t = (1 - s_v) * x_src_seg + s_v * x1_seg
                v_pred   = self.backbone(x_t, t_seg, x_cond=cond_seg)
                v_target = (x1_seg - x_src_seg) / (1.0 - t_lo)

            total_fm_loss = total_fm_loss + (v_pred - v_target).pow(2).mean()

        commitment_loss = sum(all_commit) if all_commit else None
        codebook_loss   = sum(all_codebook) if all_codebook else None
        ae_loss         = sum(all_ae) if all_ae else None

        total = total_fm_loss
        if ae_loss is not None:
            total = total + ae_loss
        if commitment_loss is not None:
            total = total + self.config.flow_config.commitment_weight * commitment_loss
        if codebook_loss is not None:
            total = total + codebook_loss

        return FlowOutput(
            loss=total,
            fm_loss=total_fm_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            ae_loss=ae_loss,
            quantizer_output=None,
        )

    # ------------------------------------------------------------------
    # Feature-cluster training forward
    # ------------------------------------------------------------------

    def _forward_with_feature_cluster(
        self,
        x0: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
        x0_cond: torch.FloatTensor | None = None,
    ) -> FlowOutput:
        """Training forward for feature-space clustering waypoints.

        For each waypoint t_q[i]:
          1. Compute x_tq = (1-t_q)*x_src + t_q*x1  (interpolation)
          2. Cluster via FeatureClusterWaypoint → assignments, x_tq_hat
          3. Normalize x1 per-cluster

        FM loss is computed per trajectory segment:
          Pre-first-waypoint:  x0 → x_tq_hat_0  (straight line)
          Between waypoints:   x_tq_hat_i → x_tq_hat_{i+1}
          Post-last-waypoint:  x_tq_hat_last → x1_norm
        """
        t_qs: List[float] = self.config.flow_config.waypoints()
        ndim_m1 = x0.ndim - 1

        # --- compute waypoint targets at each t_q ---
        x_hats: List[torch.Tensor] = []     # waypoint targets
        all_assignments: List[torch.LongTensor] = []
        all_commit: List[torch.Tensor] = []
        last_assignments: torch.LongTensor = None  # assignments from last waypoint (for x1 norm)

        for i, tq in enumerate(t_qs):
            # Interpolation at t_q. Centroids live in flattened data space so image
            # inputs ([B,C,H,W]) are flattened for clustering, then reshaped back.
            x_tq = (1 - tq) * x0 + tq * x1  # [B, ...]
            x_tq_flat = x_tq.reshape(x_tq.shape[0], -1)
            x1_flat = x1.reshape(x1.shape[0], -1)
            x_tq_hat_flat, assignments, commit_loss = self.feature_clusters[i](x_tq_flat, x1_flat)
            x_hats.append(x_tq_hat_flat.reshape(x_tq.shape))
            all_assignments.append(assignments)
            last_assignments = assignments
            all_commit.append(commit_loss)

        # Normalize x1 using the last waypoint's cluster assignments
        x1_norm = self.feature_clusters[-1].normalize_x1(
            x1.reshape(x1.shape[0], -1), last_assignments
        ).reshape(x1.shape)

        # --- FM loss per segment ---
        gravity_mode = any(getattr(fc.config, "gravity_mode", False) for fc in self.feature_clusters)
        
        total_fm_loss = x0.new_zeros(())
        
        if gravity_mode:
            # Standard continuous FM: rete impara le traiettorie naturali (x0 -> x1)
            velocity_fn = self._make_velocity_fn(x0)
            total_fm_loss, _ = self.flow.loss(x0, x1, t, velocity_fn)
        else:
            # Segments: [0, t_q0], [t_q0, t_q1], ..., [t_qN-1, 1]
            # Sources:  [x0,      x_hat_0,       ..., x_hat_{N-1}]
            # Targets at segment end: [x_hat_0, x_hat_1, ..., x1_norm]
            t_lows  = [0.0] + list(t_qs)
            t_highs = list(t_qs) + [1.0]
            sources = [x0] + x_hats               # length N+1
            targets = x_hats + [x1_norm]           # length N+1

            for seg_idx, (t_lo, t_hi, x_src, x_tgt) in enumerate(
                zip(t_lows, t_highs, sources, targets)
            ):
                is_last = (seg_idx == len(t_lows) - 1)
                mask = (t >= t_lo) & (t <= t_hi if is_last else t < t_hi)
                if not mask.any():
                    continue

                t_seg = t[mask]
                x_src_seg = x_src[mask]
                x_tgt_seg = x_tgt[mask]
                dt = t_hi - t_lo

                # Remap t to local [0, 1] within segment
                s = (t_seg - t_lo) / dt
                s_v = s.view(-1, *([1] * ndim_m1))

                # Interpolation within segment
                x_t = (1 - s_v) * x_src_seg + s_v * x_tgt_seg

                # Velocity target = (x_tgt - x_src) / dt
                v_target = (x_tgt_seg - x_src_seg) / dt

                cond_seg = (x0_cond[mask]
                            if (self._use_x0_cond and x0_cond is not None) else None)
                v_pred = self.backbone(x_t, t_seg, x_cond=cond_seg)

                total_fm_loss = total_fm_loss + (v_pred - v_target).pow(2).mean()

        # Compute cluster perplexity from last waypoint assignments
        import torch.nn.functional as F
        total_K = sum(fc.config.n_clusters for fc in self.feature_clusters)
        # Concatenate all assignments across waypoints for overall usage
        all_indices = torch.cat(all_assignments, dim=0)
        # Offset assignments for each waypoint so they map to unique bins
        offset = 0
        offset_assignments = []
        for i, asgn in enumerate(all_assignments):
            offset_assignments.append(asgn + offset)
            offset += self.feature_clusters[i].config.n_clusters
        all_offset = torch.cat(offset_assignments, dim=0)
        encodings = F.one_hot(all_offset, total_K).float()
        avg_probs = encodings.mean(0)
        perplexity = (-avg_probs * (avg_probs + 1e-10).log()).sum().exp()

        total_commit_loss = sum(all_commit) / len(all_commit) if all_commit else None

        cluster_quant_out = QuantizerOutput(
            z_q=None,
            commitment_loss=total_commit_loss,
            codebook_loss=None,
            indices=last_assignments,
            perplexity=perplexity,
        )

        total_loss = total_fm_loss
        if total_commit_loss is not None and not gravity_mode:
            total_loss = total_loss + self.config.flow_config.commitment_weight * total_commit_loss

        return FlowOutput(
            loss=total_loss,
            fm_loss=total_fm_loss,
            commitment_loss=total_commit_loss,
            codebook_loss=None,
            ae_loss=None,
            quantizer_output=cluster_quant_out,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def sample(
        self,
        x0: torch.FloatTensor,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
        disable_quantization: bool = False,
        force_waypoints: Optional[list[int]] = None,
        gravity_g: float = 0.0,
        gravity_clusters: Optional[list[int]] = None,
        t_start: float = 0.0,
    ) -> SolverOutput:
        steps = num_steps if num_steps is not None else self.config.flow_config.num_steps
        velocity_fn = self._make_velocity_fn(x0)

        if self._uses_feature_cluster and not disable_quantization:
            return self._sample_with_feature_cluster(
                x0, velocity_fn, steps, return_trajectory, force_waypoints, gravity_g, gravity_clusters, t_start
            )

        if self._is_multi:
            return self._sample_multi(x0, velocity_fn, steps, return_trajectory, disable_quantization)

        t_q = self.config.flow_config.t_q if not self.config.use_velocity_quant else None
        quant_target = self.config.flow_config.quant_target

        def quantize_fn(x: torch.Tensor) -> torch.Tensor:
            if quant_target == "velocity":
                t_vec = x.new_full((x.shape[0],), t_q)
                v = velocity_fn(x, t_vec)
                v_flat = v.reshape(v.shape[0], -1) if v.ndim > 2 else v
                q_out = self.quantizer(v_flat)
                if self.dequantizer is not None:
                    d_out = self.dequantizer(q_out.z_q)
                    v_tq_hat = d_out.x_tq_hat.reshape(x.shape)
                else:
                    v_tq_hat = q_out.z_q.reshape(x.shape)
                return x0 + t_q * v_tq_hat
            elif quant_target == "velocity_ae":
                t_vec = x.new_full((x.shape[0],), t_q)
                v_pred = velocity_fn(x, t_vec)
                v_pred_spatial = v_pred.reshape(x.shape) if v_pred.ndim <= 2 else v_pred
                z = self.velocity_ae.encode(v_pred_spatial)
                q_out = self.quantizer(z)
                v_tq_hat = self.velocity_ae.decode(q_out.z_q)
                v_tq_hat = v_tq_hat.reshape(x.shape)
                return x0 + t_q * v_tq_hat
            elif quant_target == "feature":
                t_vec = x.new_full((x.shape[0],), t_q)
                t_emb = self.backbone.time_embed(t_vec)
                x_aug = torch.cat([x, x0], dim=1) \
                    if (self._use_x0_cond and x0 is not None) else x
                z, _skips = self.backbone._encode(x_aug, t_emb)
                z_pool = z.mean(dim=[2, 3])
                q_out = self.quantizer(z_pool)
                d_out = self.dequantizer(q_out.z_q)
                v_tq_hat = d_out.x_tq_hat.reshape(x.shape)
                return x0 + t_q * v_tq_hat
            else:  # "x_tq"
                x_flat = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
                q_out = self.quantizer(x_flat)
                z_q = q_out.z_q.reshape(x.shape)
                if self.dequantizer is not None:
                    d_out = self._dequantize(z_q, x)
                    return d_out.x_tq_hat
                return z_q

        return self.solver.solve(
            x0=x0,
            velocity_fn=velocity_fn,
            num_steps=steps,
            t_q=t_q if not disable_quantization else None,
            quantize_fn=quantize_fn if (t_q is not None and not disable_quantization) else None,
            return_trajectory=return_trajectory,
        )

    def _sample_multi(
        self,
        x0: torch.FloatTensor,
        velocity_fn: Callable,
        num_steps: int,
        return_trajectory: bool,
        disable_quantization: bool = False,
    ) -> SolverOutput:
        """Inference with multiple VQ waypoints applied in sequence."""
        t_qs = self.config.flow_config.waypoints()
        if disable_quantization:
            t_qs = []
        quant_target = self.config.flow_config.quant_target

        prev_state = {"x": x0, "t": 0.0}

        def make_qfn(i: int, tq: float) -> Callable:
            def qfn(x: torch.Tensor) -> torch.Tensor:
                if quant_target == "feature":
                    t_vec = x.new_full((x.shape[0],), tq)
                    t_emb = self.backbone.time_embed(t_vec)
                    x_aug = torch.cat([x, x0], dim=1) if (self._use_x0_cond and x0 is not None) else x
                    z, _ = self.backbone._encode(x_aug, t_emb)
                    z_pool = z.mean(dim=[2, 3])
                    q_out = self.quantizers[i](z_pool)
                    d_out = self.dequantizers[i](q_out.z_q)
                    v_tq_hat = d_out.x_tq_hat.reshape(x.shape)
                    
                    dt = tq - prev_state["t"]
                    x_hat = prev_state["x"] + dt * v_tq_hat
                    prev_state["x"] = x_hat
                    prev_state["t"] = tq
                    return x_hat
                elif quant_target == "velocity_ae":
                    t_vec = x.new_full((x.shape[0],), tq)
                    v_pred = velocity_fn(x, t_vec)
                    v_pred_spatial = v_pred.reshape(x.shape) if v_pred.ndim <= 2 else v_pred
                    ae = self.velocity_ae
                    z = ae.encode(v_pred_spatial)
                    q_out = self.quantizers[i](z)
                    v_tq_hat = ae.decode(q_out.z_q)
                    v_tq_hat = v_tq_hat.reshape(x.shape)
                    
                    dt = tq - prev_state["t"]
                    x_hat = prev_state["x"] + dt * v_tq_hat
                    prev_state["x"] = x_hat
                    prev_state["t"] = tq
                    return x_hat
                else:
                    return x
            return qfn

        quantize_fns = [make_qfn(i, tq) for i, tq in enumerate(t_qs)]

        return self.solver.solve(
            x0=x0,
            velocity_fn=velocity_fn,
            num_steps=num_steps,
            t_qs=t_qs,
            quantize_fns=quantize_fns,
            return_trajectory=return_trajectory,
        )

    # ------------------------------------------------------------------
    # Feature-cluster inference
    # ------------------------------------------------------------------

    def _sample_with_feature_cluster(
        self,
        x0: torch.FloatTensor,
        velocity_fn: Callable,
        num_steps: int,
        return_trajectory: bool,
        force_waypoints: Optional[list[int]] = None,
        gravity_g: float = 0.0,
        gravity_clusters: Optional[list[int]] = None,
        t_start: float = 0.0,
    ) -> SolverOutput:
        """Inference with feature-cluster waypoints.

        At each t_q: assign to nearest centroid, sample noisy waypoint,
        then continue ODE. After final waypoint, denormalize output.
        """
        t_qs = self.config.flow_config.waypoints()
        fc_modules = self.feature_clusters

        if force_waypoints is not None:
            # Normalizziamo force_waypoints in una lista di liste.
            if isinstance(force_waypoints[0], int):
                if len(t_qs) == 1:
                    force_waypoints = [force_waypoints]
                else:
                    assert len(force_waypoints) == len(t_qs), "Must provide a cluster index for each waypoint"
                    force_waypoints = [[idx] for idx in force_waypoints]
            else:
                assert len(force_waypoints) == len(t_qs), "Must provide choices for each waypoint"

            original_velocity_fn = velocity_fn
            
            def forced_velocity_fn(x, t):
                t_val = t[0].item()
                # Find the next waypoint
                for i, tq in enumerate(t_qs):
                    if t_val < tq - 1e-5:
                        choices = force_waypoints[i]
                        B = x.shape[0]
                        # Distribuiamo il batch in modo ciclico tra le scelte
                        idx_tensor = torch.arange(B, device=x.device) % len(choices)
                        choices_tensor = torch.tensor(choices, device=x.device)
                        cluster_indices = choices_tensor[idx_tensor]
                        
                        W = fc_modules[i].centroids[cluster_indices] # Shape: [B, D]
                        if x.ndim > 2:
                            W = W.view(B, *x.shape[1:])
                        return (W - x) / (tq - t_val)
                # After all waypoints, use the original velocity_fn
                return original_velocity_fn(x, t)
            
            velocity_fn = forced_velocity_fn

        # Track last assignments for denormalization
        last_assignments: torch.LongTensor = None

        def make_cluster_qfn(i: int, tq: float) -> Callable:
            def qfn(x: torch.Tensor) -> torch.Tensor:
                nonlocal last_assignments
                x_flat = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
                x_hat, assignments = fc_modules[i].assign_and_sample(x_flat)
                last_assignments = assignments
                return x_hat.reshape(x.shape)
            return qfn

        quantize_fns = [make_cluster_qfn(i, tq) for i, tq in enumerate(t_qs)]

        result = self.solver.solve(
            x0=x0,
            velocity_fn=velocity_fn,
            num_steps=num_steps,
            t_qs=t_qs,
            quantize_fns=quantize_fns,
            return_trajectory=return_trajectory,
            gravity_g=gravity_g,
            feature_clusters=fc_modules,
            gravity_clusters=gravity_clusters,
            t_start=t_start,
        )

        # Denormalize final output using last waypoint's cluster stats
        if last_assignments is not None:
            x1_norm = result.x1.reshape(result.x1.shape[0], -1)
            x1_denorm = fc_modules[-1].denormalize(x1_norm, last_assignments)
            result = SolverOutput(
                x1=x1_denorm.reshape(result.x1.shape),
                trajectory=result.trajectory,
            )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _use_x0_cond(self) -> bool:
        if self.config.mlp_backbone_config is not None:
            return self.config.mlp_backbone_config.use_x0_cond
        if self.config.unet_backbone_config is not None:
            return self.config.unet_backbone_config.use_x0_cond
        return False

    def _velocity_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x, t)
        if isinstance(self.backbone, VelocityQuantizedFlow):
            self._last_quant_output = self.backbone._last_quant_output
        return out

    def _make_velocity_fn(self, x0: torch.Tensor, force_continuous: bool = False):
        if not self._use_x0_cond:
            if force_continuous and isinstance(self.backbone, VelocityBSQWrapper):
                return lambda x, t: self.backbone(x, t, force_continuous=True)
            return self._velocity_fn
        backbone = self.backbone
        def _fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if isinstance(backbone, VelocityBSQWrapper):
                return backbone(x, t, x_cond=x0, force_continuous=force_continuous)
            return backbone(x, t, x_cond=x0)
        return _fn

    def _dequantize(self, z_q: torch.Tensor, x_tq: torch.Tensor) -> DequantizerOutput:
        if isinstance(self.dequantizer, ResidualDequantizer):
            return self.dequantizer(z_q, x_tq)
        return self.dequantizer(z_q)
