from typing import Any

from .configs import (
    FlowQuantConfig, FlowConfig, TrainingConfig,
    MLPBackboneConfig, UNetBackboneConfig, DiTBackboneConfig,
    VQConfig, FSQConfig, BSQConfig,
    StochasticDequantizerConfig, ResidualDequantizerConfig,
    LinearDequantizerConfig, VelocityAEConfig,
)
from .flow_model import FlowQuant


def _optional(cfg_dict: dict[str, Any], key1: str, key2: str, cls):
    raw = cfg_dict.get(key1)
    if raw is None:
        raw = cfg_dict.get(key2)
    return cls(**raw) if raw is not None else None


def build_model(cfg_dict: dict[str, Any]) -> FlowQuant:
    flow_raw = cfg_dict.get("flow") or cfg_dict.get("flow_config")
    flow_config = FlowConfig(**flow_raw) if flow_raw else None
    
    training_raw = cfg_dict.get("training") or cfg_dict.get("training_config")
    training_config = TrainingConfig(**training_raw) if training_raw else None

    mlp_backbone_config = _optional(cfg_dict, "mlp_backbone", "mlp_backbone_config", MLPBackboneConfig)
    unet_backbone_config = _optional(cfg_dict, "unet_backbone", "unet_backbone_config", UNetBackboneConfig)
    dit_backbone_config = _optional(cfg_dict, "dit_backbone", "dit_backbone_config", DiTBackboneConfig)

    vq_config = _optional(cfg_dict, "vq", "vq_config", VQConfig)
    fsq_config = _optional(cfg_dict, "fsq", "fsq_config", FSQConfig)
    bsq_config = _optional(cfg_dict, "bsq", "bsq_config", BSQConfig)

    stochastic_dequantizer_config = _optional(
        cfg_dict, "stochastic_dequantizer", "stochastic_dequantizer_config", StochasticDequantizerConfig
    )
    residual_dequantizer_config = _optional(
        cfg_dict, "residual_dequantizer", "residual_dequantizer_config", ResidualDequantizerConfig
    )
    
    _ld_raw = cfg_dict.get("linear_dequantizer") or cfg_dict.get("linear_dequantizer_config")
    linear_dequantizer_config = LinearDequantizerConfig(**_ld_raw) if _ld_raw else None

    velocity_ae_config = _optional(cfg_dict, "velocity_ae_config", "velocity_ae_config", VelocityAEConfig)

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
