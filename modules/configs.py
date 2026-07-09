from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass(kw_only=True)
class VelocityAEConfig:
    hidden_dim: int = 64
    bottleneck_dim: int = 256

@dataclass(kw_only=True)
class MLPBackboneConfig:
    hidden_dim: int
    num_layers: int
    time_embed_dim: int
    use_x0_cond: bool = False  # concatenate x0 to backbone input at every ODE step


@dataclass(kw_only=True)
class UNetBackboneConfig:
    base_channels: int
    channel_multipliers: list[int]
    num_res_blocks: int
    time_embed_dim: int
    attention_resolutions: list[int]
    bottleneck_channels: int
    use_x0_cond: bool = False  # concatenate x0 to x_t on channel dim at every ODE step
    # Optional VQ at bottleneck: GAP → VQ (1 global code/image) → broadcast back
    use_bottleneck_vq: bool = False
    bottleneck_vq_num_embeddings: Optional[int] = None
    bottleneck_vq_use_ema: bool = True
    bottleneck_vq_init_scale: Optional[float] = None


@dataclass(kw_only=True)
class DiTBackboneConfig:
    hidden_dim: int
    num_heads: int
    depth: int
    patch_size: int
    time_embed_dim: int


@dataclass(kw_only=True)
class VQConfig:
    num_embeddings: int
    embedding_dim: int
    commitment_weight: float
    use_ema: bool
    embedding_init_scale: Optional[float] = None  # None → default 1/num_embeddings


@dataclass(kw_only=True)
class FSQConfig:
    levels: list[int]    # e.g. [8, 5, 5, 5] → codebook size = prod(levels)


@dataclass(kw_only=True)
class BSQConfig:
    dim: int      # number of bits (BSQ lives on S^{dim-1}, 2^dim implicit codes)
    in_dim: int   # input dimension; Linear(in_dim→dim) inserted if in_dim ≠ dim


@dataclass(kw_only=True)
class StochasticDequantizerConfig:
    sigma: float    # noise std: z̃ = z_q + σε (Metodo A)


@dataclass(kw_only=True)
class ResidualDequantizerConfig:
    alpha: float    # z̃ = (1-α)z_q + αx_tq, residual R = x_tq - z_q (Metodo B)


@dataclass(kw_only=True)
class LinearDequantizerConfig:
    in_dim: int   # feature/velocity dim (e.g. 256)
    out_dim: int  # pixel dim (e.g. 784)


@dataclass(kw_only=True)
class FeatureClusterConfig:
    n_clusters: int          # K waypoints in feature space
    noise_std: float = 1.0   # std of noise N(0, noise_std) around centroid
    gravity_mode: bool = False # se True, disabilita l'interferenza dell'addestramento e traccia solo i centroidi
    gravity_softening: float = 0.01 # epsilon per la stabilità 1/(d^2 + epsilon)
    learnable_noise: bool = False  # se True, noise_std diventa il valore iniziale di un parametro addestrabile
    ema_decay: float = 0.99  # decay for running centroid & normalization stats
    project_centroids: bool = False  # se True, proietta i centroidi sull'ipercirconferenza di raggio t_q


@dataclass(kw_only=True)
class FlowConfig:
    flow_type: str          # "rectified" | "ot_cfm"
    solver: str             # "euler" | "midpoint" | "rk4"
    num_steps: int          # ODE steps at inference
    t_q: Optional[float]    # single quantization waypoint; None disables spatial bottleneck
    t_qs: Optional[List[float]] = None  # multi-waypoint list; if set, overrides t_q
    source_std: float = 1.0             # std of p0 (source distribution)
    commitment_weight: float = 0.25     # total_loss += commitment_weight * commit_loss
    quant_target: str = "x_tq"  # "x_tq" | "velocity" | "feature"

    def waypoints(self) -> List[float]:
        """Sorted list of active quantization waypoints."""
        if self.t_qs is not None and len(self.t_qs) > 0:
            return sorted(self.t_qs)
        if self.t_q is not None:
            return [self.t_q]
        return []


@dataclass(kw_only=True)
class TrainingConfig:
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    seed: int
    dataset_name: str
    eval_num_samples: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    bf16: bool
    fp16: bool
    report_to: Optional[str]
    wandb_project: Optional[str]
    wandb_run_name: Optional[str]
    # toy-2D dataset params (null for image datasets)
    gaussians_n: Optional[int]
    gaussians_radius: Optional[float]
    gaussians_std: Optional[float]
    swissroll_noise: Optional[float]
    run_id: Optional[str] = None   # injected by run_job.sh; unused in code
    # C2-style warmstart: load backbone weights from this checkpoint dir before training
    warmstart_backbone_checkpoint: Optional[str] = None
    # freeze backbone (requires_grad=False) before optimizer creation
    freeze_backbone: bool = False


@dataclass(kw_only=True)
class FlowQuantConfig:
    data_dim: int               # feature dim at the quantization bottleneck
    image_size: Optional[int]   # spatial resolution (Phase 2-3); None for Phase 1
    in_channels: Optional[int]  # image channels (Phase 2-3); None for Phase 1
    use_velocity_quant: bool    # Section 4: quantize v_theta instead of x_t

    flow_config: FlowConfig
    training_config: TrainingConfig

    velocity_ae_config: Optional[VelocityAEConfig] = None

    # Backbone — exactly one non-null
    mlp_backbone_config: Optional[MLPBackboneConfig]
    unet_backbone_config: Optional[UNetBackboneConfig]
    dit_backbone_config: Optional[DiTBackboneConfig]

    # Quantizer — exactly one non-null (unless use_velocity_quant=False and t_q=None)
    vq_config: Optional[VQConfig]
    fsq_config: Optional[FSQConfig]
    bsq_config: Optional[BSQConfig] = None

    # Inline velocity BSQ (C1-style): wraps backbone output with magnitude-preserving
    # binary quantization for all t < velocity_bsq_threshold during ODE integration.
    use_velocity_bsq: bool = False
    velocity_bsq_threshold: float = 0.75

    # Dequantizer — exactly one non-null for spatial bottleneck; both None for velocity quant
    stochastic_dequantizer_config: Optional[StochasticDequantizerConfig]
    residual_dequantizer_config: Optional[ResidualDequantizerConfig]
    linear_dequantizer_config: Optional[LinearDequantizerConfig] = None

    # Feature-space clustering waypoints (replaces VQ when set)
    feature_cluster_configs: Optional[List[FeatureClusterConfig]] = None

    def __post_init__(self) -> None:
        n_backbones = sum(c is not None for c in [
            self.mlp_backbone_config, self.unet_backbone_config, self.dit_backbone_config
        ])
        if n_backbones != 1:
            raise ValueError(f"exactly one backbone config must be non-null; got {n_backbones}")

        _has_waypoints = bool(self.flow_config.waypoints())
        _uses_feature_cluster = (self.feature_cluster_configs is not None
                                 and len(self.feature_cluster_configs) > 0)

        # Feature-cluster mode bypasses VQ/dequantizer entirely
        if _uses_feature_cluster:
            n_wps = len(self.flow_config.waypoints())
            n_fc = len(self.feature_cluster_configs)
            if n_wps != n_fc:
                raise ValueError(
                    f"feature_cluster_configs length ({n_fc}) must match "
                    f"number of waypoints ({n_wps})"
                )
            return  # skip VQ / dequantizer validation

        n_quantizers = sum(c is not None for c in [self.vq_config, self.fsq_config, self.bsq_config])
        if n_quantizers > 1:
            raise ValueError("at most one quantizer config must be non-null")
        if n_quantizers == 0 and (_has_waypoints or self.use_velocity_quant):
            raise ValueError("a quantizer is required when waypoints are set or use_velocity_quant=True")

        n_dequantizers = sum(c is not None for c in [
            self.stochastic_dequantizer_config, self.residual_dequantizer_config,
            self.linear_dequantizer_config,
        ])
        if n_dequantizers > 1:
            raise ValueError("at most one dequantizer config must be non-null")
        quant_target = self.flow_config.quant_target
        # Only "feature" VQ strictly requires a dequantizer (maps bottleneck → pixel space).
        # "velocity" uses z_q directly (optionally with dequantizer for projection back).
        # "x_tq" can use z_q directly as the waypoint (codebook vector in data space).
        needs_dequantizer = (
            _has_waypoints
            and not self.use_velocity_quant
            and quant_target == "feature"
        )
        if needs_dequantizer and n_dequantizers != 1:
            raise ValueError("a dequantizer is required when quant_target='feature'")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model_type"] = "FlowQuant"
        return d
