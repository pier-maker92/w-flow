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

        # Quantizer / dequantizer
        n_waypoints = len(config.flow_config.waypoints())
        self._is_multi = n_waypoints > 1

        if self._is_multi:
            # Multi-waypoint: one VQ + one LinearDequantizer per waypoint
            assert config.vq_config is not None, "vq_config required for multi-waypoint"
            assert config.linear_dequantizer_config is not None, \
                "linear_dequantizer required for multi-waypoint"
            self.quantizer: Optional[nn.Module] = None
            self.dequantizer: Optional[nn.Module] = None
            self.quantizers = nn.ModuleList([
                VectorQuantizer(config.vq_config) for _ in range(n_waypoints)
            ])
            self.dequantizers = nn.ModuleList([
                LinearDequantizer(config.linear_dequantizer_config)
                for _ in range(n_waypoints)
            ])
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

        if self._is_multi:
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

        if quant_target == "velocity":
            cond = x0_cond if (self._use_x0_cond and x0_cond is not None) else None
            v_tq = self.backbone(x_tq, t_q_vec, x_cond=cond)
            v_tq_flat = v_tq.reshape(v_tq.shape[0], -1) if v_tq.ndim > 2 else v_tq
            quant_out = self.quantizer(v_tq_flat)
            if self.dequantizer is not None:
                d_out = self.dequantizer(quant_out.z_q)
                x_tq_hat = d_out.x_tq_hat.reshape(x_tq.shape)
            else:
                x_tq_hat = quant_out.z_q.reshape(x_tq.shape)

        elif quant_target == "feature":
            t_emb = self.backbone.time_embed(t_q_vec)
            x_aug = torch.cat([x_tq, x0_cond], dim=1) \
                if (self._use_x0_cond and x0_cond is not None) else x_tq
            z, _skips = self.backbone._encode(x_aug, t_emb)
            z_pool = z.mean(dim=[2, 3])
            quant_out = self.quantizer(z_pool)
            dequant_out = self.dequantizer(quant_out.z_q)
            x_tq_hat = dequant_out.x_tq_hat.reshape(x_tq.shape)

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
        if commitment_loss is not None:
            total = total + self.config.flow_config.commitment_weight * commitment_loss
        if codebook_loss is not None:
            total = total + codebook_loss

        return FlowOutput(
            loss=total,
            fm_loss=total_fm_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
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

        # --- compute x̂ at each waypoint via ground-truth interpolation ---
        x_hats: List[torch.Tensor] = []
        all_commit: List[torch.Tensor] = []
        all_codebook: List[torch.Tensor] = []

        for i, tq in enumerate(t_qs):
            x_tq = (1 - tq) * x0 + tq * x1
            t_q_vec = x0.new_full((x0.shape[0],), tq)
            t_emb = self.backbone.time_embed(t_q_vec)
            x_aug = (torch.cat([x_tq, x0_cond], dim=1)
                     if (self._use_x0_cond and x0_cond is not None) else x_tq)
            z, _ = self.backbone._encode(x_aug, t_emb)
            z_pool = z.mean(dim=[2, 3])
            q_out = self.quantizers[i](z_pool)
            d_out = self.dequantizers[i](q_out.z_q)
            x_hats.append(d_out.x_tq_hat.reshape(x_tq.shape))
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

        total = total_fm_loss
        if commitment_loss is not None:
            total = total + self.config.flow_config.commitment_weight * commitment_loss
        if codebook_loss is not None:
            total = total + codebook_loss

        return FlowOutput(
            loss=total,
            fm_loss=total_fm_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            quantizer_output=None,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def sample(
        self,
        x0: torch.FloatTensor,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> SolverOutput:
        steps = num_steps if num_steps is not None else self.config.flow_config.num_steps
        velocity_fn = self._make_velocity_fn(x0)

        if self._is_multi:
            return self._sample_multi(x0, velocity_fn, steps, return_trajectory)

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
                    return d_out.x_tq_hat.reshape(x.shape)
                return q_out.z_q.reshape(x.shape)
            elif quant_target == "feature":
                t_vec = x.new_full((x.shape[0],), t_q)
                t_emb = self.backbone.time_embed(t_vec)
                x_aug = torch.cat([x, x0], dim=1) \
                    if (self._use_x0_cond and x0 is not None) else x
                z, _skips = self.backbone._encode(x_aug, t_emb)
                z_pool = z.mean(dim=[2, 3])
                q_out = self.quantizer(z_pool)
                d_out = self.dequantizer(q_out.z_q)
                return d_out.x_tq_hat.reshape(x.shape)
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
            t_q=t_q,
            quantize_fn=quantize_fn if t_q is not None else None,
            return_trajectory=return_trajectory,
        )

    def _sample_multi(
        self,
        x0: torch.FloatTensor,
        velocity_fn: Callable,
        num_steps: int,
        return_trajectory: bool,
    ) -> SolverOutput:
        """Inference with multiple VQ waypoints applied in sequence."""
        t_qs = self.config.flow_config.waypoints()

        def make_qfn(i: int, tq: float) -> Callable:
            def qfn(x: torch.Tensor) -> torch.Tensor:
                t_vec = x.new_full((x.shape[0],), tq)
                t_emb = self.backbone.time_embed(t_vec)
                x_aug = torch.cat([x, x0], dim=1) if self._use_x0_cond else x
                z, _ = self.backbone._encode(x_aug, t_emb)
                z_pool = z.mean(dim=[2, 3])
                q_out = self.quantizers[i](z_pool)
                d_out = self.dequantizers[i](q_out.z_q)
                return d_out.x_tq_hat.reshape(x.shape)
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
