import os
import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from transformers import Trainer, TrainingArguments, set_seed

from modules.builder import build_model
from modules.output_dataclasses import FlowOutput
from util import build_dataset, wandb_init

log = logging.getLogger(__name__)


class FlowQuantTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        x1 = inputs["x1"]
        outputs: FlowOutput = model(x1=x1)
        loss = outputs.loss
        if self.state.global_step % self.args.logging_steps == 0:
            log_dict = {
                "fm_loss": outputs.fm_loss.item() if outputs.fm_loss is not None else 0.0,
                "commitment_loss": outputs.commitment_loss.item() if outputs.commitment_loss is not None else 0.0,
                "perplexity": outputs.quantizer_output.perplexity.item()
                    if outputs.quantizer_output is not None and outputs.quantizer_output.perplexity is not None
                    else 0.0,
            }
            if outputs.ae_loss is not None:
                log_dict["ae_loss"] = outputs.ae_loss.item()
                
            unwrapped_model = model.module if hasattr(model, "module") else model
            if getattr(unwrapped_model, "feature_clusters", None) is not None:
                for i, fc in enumerate(unwrapped_model.feature_clusters):
                    if fc.log_noise_std is not None:
                        log_dict[f"noise_std_wp{i}"] = torch.exp(fc.log_noise_std).item()
                        
            self.log(log_dict)
        return (loss, outputs) if return_outputs else loss


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    training_cfg_dict = cfg_dict["training"]
    set_seed(training_cfg_dict["seed"])

    model = build_model(cfg_dict)

    # Warmstart: load backbone weights from a prior checkpoint (e.g. C1 → C2)
    warmstart_ckpt = training_cfg_dict.get("warmstart_backbone_checkpoint")
    if warmstart_ckpt:
        ckpt_path = Path(warmstart_ckpt) / "model.safetensors"
        log.info(f"Warmstart: loading backbone from {ckpt_path}")
        state = load_file(ckpt_path)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info(f"  missing keys : {missing}")
        log.info(f"  unexpected   : {unexpected}")

    # Freeze backbone (C2-style: train only quantizer/dequantizer)
    if training_cfg_dict.get("freeze_backbone"):
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        n_frozen = sum(p.numel() for p in model.backbone.parameters())
        log.info(f"Backbone frozen: {n_frozen:,} params set requires_grad=False")

    wandb_init(model.config.training_config, cfg_dict)

    train_ds, collator = build_dataset(model.config.training_config, split="train")
    eval_ds, _ = build_dataset(model.config.training_config, split="test")

    eval_n = model.config.training_config.eval_num_samples
    if eval_n < len(eval_ds):
        eval_ds = torch.utils.data.Subset(eval_ds, list(range(eval_n)))

    output_dir = training_cfg_dict["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(model.config.to_dict(), f, indent=2)

    bf16 = training_cfg_dict["bf16"]
    fp16 = training_cfg_dict["fp16"]
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg_dict["num_train_epochs"],
        per_device_train_batch_size=training_cfg_dict["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg_dict["per_device_train_batch_size"],
        learning_rate=training_cfg_dict["learning_rate"],
        lr_scheduler_type=training_cfg_dict["lr_scheduler_type"],
        warmup_ratio=training_cfg_dict["warmup_ratio"],
        save_steps=training_cfg_dict["save_steps"],
        eval_steps=training_cfg_dict["eval_steps"],
        logging_steps=training_cfg_dict["logging_steps"],
        bf16=bf16,
        fp16=fp16,
        report_to=training_cfg_dict["report_to"] or "none",
        eval_strategy="steps",
        save_strategy="steps",
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )

    trainer = FlowQuantTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    log.info(f"Training complete. Final model saved to {output_dir}")

    # Post-hoc waypoint-merge analysis (feature-cluster gravity runs).
    # Non-fatal: the checkpoint is already saved above, so a bug here must not
    # mark the run as failed.
    if training_cfg_dict.get("merge_analysis"):
        try:
            from evaluation.waypoint_merge import run_merge_sweep
            agent_dir = os.path.join(
                os.environ.get("SCRATCH", "."), "agente", "reports",
                "waypoint_merge", Path(output_dir).name,
            )
            log.info(f"Running waypoint-merge analysis -> {agent_dir}")
            res = run_merge_sweep(
                model=trainer.model,
                out_dir=agent_dir,
                thresholds_deg=training_cfg_dict.get("merge_thresholds_deg"),
                k_neighbors=training_cfg_dict.get("merge_k_neighbors", 6),
                target_threshold_deg=training_cfg_dict.get("merge_target_deg", 30.0),
            )
            log.info(f"Merge sweep K_eff by threshold: {res['k_eff_by_threshold']}")
        except Exception:
            log.exception("waypoint-merge analysis failed (checkpoint is saved; re-run offline)")


if __name__ == "__main__":
    main()
