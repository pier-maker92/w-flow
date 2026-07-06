from typing import Any

from .configs import (
    FlowQuantConfig, FlowConfig, TrainingConfig,
    MLPBackboneConfig, UNetBackboneConfig, DiTBackboneConfig,
    VQConfig, FSQConfig, BSQConfig,
    StochasticDequantizerConfig, ResidualDequantizerConfig,
    LinearDequantizerConfig, VelocityAEConfig,
)
from .flow_model import FlowQuant


def _optional(cfg_dict: dict[str, Any], key: str, cls):
    raw = cfg_dict[key]
    return cls(**raw) if raw is not None else None


def build_model(cfg_dict: dict[str, Any]) -> FlowQuant:
    flow_config = FlowConfig(**cfg_dict["flow"])
    training_config = TrainingConfig(**cfg_dict["training"])

    mlp_backbone_config = _optional(cfg_dict, "mlp_backbone", MLPBackboneConfig)
    unet_backbone_config = _optional(cfg_dict, "unet_backbone", UNetBackboneConfig)
    dit_backbone_config = _optional(cfg_dict, "dit_backbone", DiTBackboneConfig)

    vq_config = _optional(cfg_dict, "vq", VQConfig)
    fsq_config = _optional(cfg_dict, "fsq", FSQConfig)
    bsq_config = _optional(cfg_dict, "bsq", BSQConfig) if "bsq" in cfg_dict else None

    stochastic_dequantizer_config = _optional(
        cfg_dict, "stochastic_dequantizer", StochasticDequantizerConfig
    )
    residual_dequantizer_config = _optional(
        cfg_dict, "residual_dequantizer", ResidualDequantizerConfig
    )
    _ld_raw = cfg_dict.get("linear_dequantizer")
    linear_dequantizer_config = LinearDequantizerConfig(**_ld_raw) if _ld_raw else None

    velocity_ae_config = _optional(cfg_dict, "velocity_ae_config", VelocityAEConfig) if "velocity_ae_config" in cfg_dict else None

    config = FlowQuantConfig(
        data_dim=cfg_dict["data_dim"],
        image_size=cfg_dict["image_size"],
        in_channels=cfg_dict["in_channels"],
        use_velocity_quant=cfg_dict["use_velocity_quant"],
        use_velocity_bsq=cfg_dict.get("use_velocity_bsq", False),
        velocity_bsq_threshold=cfg_dict.get("velocity_bsq_threshold", 0.75),
        flow_config=flow_config,
        training_config=training_config,
        mlp_backbone_config=mlp_backbone_config,
        unet_backbone_config=unet_backbone_config,
        dit_backbone_config=dit_backbone_config,
        vq_config=vq_config,
        fsq_config=fsq_config,
        bsq_config=bsq_config,
        stochastic_dequantizer_config=stochastic_dequantizer_config,
        residual_dequantizer_config=residual_dequantizer_config,
        linear_dequantizer_config=linear_dequantizer_config,
        velocity_ae_config=velocity_ae_config,
    )
    return FlowQuant(config=config)
