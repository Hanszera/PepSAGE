#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from exp1_tokenizer.collate import LocalTokenizerCollate
from exp1_tokenizer.dataset import PeptideLocalTokenizerDataset, ProteinParquetLocalTokenizerDataset
from exp1_tokenizer.lightning import LocalFrameTokenizerModule

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
VARIANT_PRESETS = {
    "global_no_type": {
        "model": {
            "use_local_frame": False,
            "use_residue_type": False,
            "use_atom_slot": False,
            "use_atom_class": False,
        },
        "loss": {
            "local_mse_weight": 0.0,
            "global_mse_weight": 1.0,
            "intra_residue_distance_weight": 0.0,
            "bond_length_weight": 0.0,
            "bond_angle_weight": 0.0,
            "clash_weight": 0.0,
        },
    },
    "local_no_type": {
        "model": {
            "use_local_frame": True,
            "use_residue_type": False,
            "use_atom_slot": False,
            "use_atom_class": False,
        },
        "loss": {
            "local_mse_weight": 1.0,
            "global_mse_weight": 1.0,
            "intra_residue_distance_weight": 0.0,
            "bond_length_weight": 0.0,
            "bond_angle_weight": 0.0,
            "clash_weight": 0.0,
        },
    },
    "local_with_type": {
        "model": {
            "use_local_frame": True,
            "use_residue_type": True,
            "use_atom_slot": True,
            "use_atom_class": True,
        },
        "loss": {
            "local_mse_weight": 1.0,
            "global_mse_weight": 1.0,
            "intra_residue_distance_weight": 0.0,
            "bond_length_weight": 0.0,
            "bond_angle_weight": 0.0,
            "clash_weight": 0.0,
        },
    },
    "local_with_type_reg": {
        "model": {
            "use_local_frame": True,
            "use_residue_type": True,
            "use_atom_slot": True,
            "use_atom_class": True,
        },
    },
}


def resolve_output_dir(base_dir: Path, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_workspace_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (WORKSPACE_ROOT / path).resolve()
    return path


def apply_experiment_preset(cfg) -> tuple[str, int]:
    experiment_cfg = getattr(cfg, "experiment", {})
    variant = str(getattr(experiment_cfg, "variant", "local_with_type")).lower()
    if variant not in VARIANT_PRESETS:
        valid = ", ".join(sorted(VARIANT_PRESETS))
        raise ValueError(f"Unsupported experiment.variant: {variant}. Expected one of: {valid}")

    for section_name, overrides in VARIANT_PRESETS[variant].items():
        for key, value in overrides.items():
            OmegaConf.update(cfg, f"{section_name}.{key}", value, merge=True)

    chemistry_cfg = getattr(experiment_cfg, "chemistry", None)
    if chemistry_cfg is not None:
        if variant == "local_with_type_reg":
            OmegaConf.update(
                cfg,
                "loss.intra_residue_distance_weight",
                float(getattr(chemistry_cfg, "distance_weight", getattr(cfg.loss, "intra_residue_distance_weight", 0.0))),
                merge=True,
            )
            OmegaConf.update(cfg, "loss.bond_length_weight", float(getattr(chemistry_cfg, "bond_length_weight", 0.0)), merge=True)
            OmegaConf.update(cfg, "loss.bond_angle_weight", float(getattr(chemistry_cfg, "bond_angle_weight", 0.0)), merge=True)
            OmegaConf.update(cfg, "loss.clash_weight", float(getattr(chemistry_cfg, "clash_weight", 0.0)), merge=True)
        else:
            OmegaConf.update(cfg, "loss.intra_residue_distance_weight", 0.0, merge=True)
            OmegaConf.update(cfg, "loss.bond_length_weight", 0.0, merge=True)
            OmegaConf.update(cfg, "loss.bond_angle_weight", 0.0, merge=True)
            OmegaConf.update(cfg, "loss.clash_weight", 0.0, merge=True)

    compression_factor = int(getattr(experiment_cfg, "compression_factor", getattr(cfg.model, "compression_factor", 1)))
    if compression_factor < 1:
        raise ValueError("experiment.compression_factor must be >= 1")
    OmegaConf.update(cfg, "model.compression_factor", compression_factor, merge=True)

    pad_to_multiple_of = int(getattr(cfg.data, "pad_to_multiple_of", 1))
    OmegaConf.update(
        cfg,
        "data.pad_to_multiple_of",
        math.lcm(max(1, pad_to_multiple_of), compression_factor),
        merge=True,
    )
    return variant, compression_factor


def build_pepsage_dataset(split_cfg):
    return PeptideLocalTokenizerDataset(
        structure_dir=str(resolve_workspace_path(split_cfg.structure_dir)),
        dataset_dir=str(resolve_workspace_path(split_cfg.dataset_dir)),
        name=split_cfg.name,
        reset=split_cfg.reset,
        max_tokens=getattr(split_cfg, "max_tokens", None),
    )


def build_general_protein_dataset(split_cfg):
    return ProteinParquetLocalTokenizerDataset(
        split_cfg=split_cfg,
        max_tokens=getattr(split_cfg, "max_tokens", None),
    )


def build_dataloader(dataset, batch_size, num_workers, pad_to_multiple_of, shuffle, prefetch_factor):
    collate = LocalTokenizerCollate(pad_to_multiple_of=pad_to_multiple_of)
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate,
    )
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        **dataloader_kwargs,
    )


def build_source_loaders(cfg, source_name: str):
    source_cfg = cfg.data[source_name]
    train_split_cfg = source_cfg.train
    val_split_cfg = source_cfg.val

    if source_name == "pepsage":
        train_dataset = build_pepsage_dataset(train_split_cfg)
        val_dataset = build_pepsage_dataset(val_split_cfg)
    elif source_name == "general_protein":
        train_dataset = build_general_protein_dataset(train_split_cfg)
        val_dataset = build_general_protein_dataset(val_split_cfg)
    else:
        raise ValueError(f"Unknown source_name: {source_name}")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        pad_to_multiple_of=int(cfg.data.pad_to_multiple_of),
        shuffle=bool(getattr(train_split_cfg, "shuffle", True)),
        prefetch_factor=int(getattr(cfg.data, "prefetch_factor", 2)),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=int(cfg.data.val_batch_size),
        num_workers=int(cfg.data.num_workers),
        pad_to_multiple_of=int(cfg.data.pad_to_multiple_of),
        shuffle=bool(getattr(val_split_cfg, "shuffle", False)),
        prefetch_factor=int(getattr(cfg.data, "prefetch_factor", 2)),
    )
    return train_loader, val_loader


def merge_trainer_config(base_cfg, override_cfg=None):
    base = OmegaConf.to_container(base_cfg, resolve=True)
    if override_cfg is None:
        return base
    override = OmegaConf.to_container(override_cfg, resolve=True)
    base.update(override)
    return base


def load_config(config_path: str | Path):
    config_path = Path(config_path).resolve()
    cfg = OmegaConf.load(config_path)
    inherit_from = getattr(cfg, "inherit_from", None)
    if not inherit_from:
        return cfg

    parent_path = Path(str(inherit_from))
    if not parent_path.is_absolute():
        parent_path = (config_path.parent / parent_path).resolve()

    parent_cfg = load_config(parent_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    cfg_dict.pop("inherit_from", None)
    return OmegaConf.merge(parent_cfg, OmegaConf.create(cfg_dict))


def resolve_eval_ckpt_path(checkpoint_callback, ckpt_policy: str | None):
    if ckpt_policy is None:
        return None
    policy = str(ckpt_policy).lower()
    if policy == "best":
        return checkpoint_callback.best_model_path or None
    if policy == "last":
        return checkpoint_callback.last_model_path or None
    if policy in {"none", "current"}:
        return None
    return ckpt_policy


def run_final_evaluation(trainer, model, val_loader, checkpoint_callback, final_eval_cfg):
    if not bool(getattr(final_eval_cfg, "enabled", True)):
        return None

    model.enable_full_evaluation(
        max_batches=int(getattr(final_eval_cfg, "heavy_metrics_max_batches", 0)),
        log_prefix=str(getattr(final_eval_cfg, "log_prefix", "full_eval")),
    )
    ckpt_path = resolve_eval_ckpt_path(checkpoint_callback, getattr(final_eval_cfg, "ckpt_path", "best"))
    try:
        return trainer.validate(model=model, dataloaders=val_loader, ckpt_path=ckpt_path, verbose=True)
    finally:
        model.disable_full_evaluation()


def run_stage(model, train_loader, val_loader, output_dir: Path, trainer_kwargs, stage_name: str, final_eval_cfg=None, run_final_eval: bool = True):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(save_dir=str(output_dir), name="tensorboard")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{stage_name}-epoch{{epoch:03d}}-val_loss{{val/loss:.4f}}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        checkpoint_callback,
    ]
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        **trainer_kwargs,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if run_final_eval:
        run_final_evaluation(trainer, model, val_loader, checkpoint_callback, final_eval_cfg)
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_local_tokenizer.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    pl.seed_everything(int(cfg.seed), workers=True)
    train_mode = str(cfg.train_mode).lower()
    variant, compression_factor = apply_experiment_preset(cfg)

    model = LocalFrameTokenizerModule(
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        loss_config=OmegaConf.to_container(cfg.loss, resolve=True),
        optim_config=OmegaConf.to_container(cfg.optim, resolve=True),
        validation_config=OmegaConf.to_container(getattr(cfg, "validation", {}), resolve=True),
    )

    workdir = Path(__file__).resolve().parent
    output_dir = resolve_output_dir(workdir, cfg.output_dir)
    experiment_dir = output_dir / f"{variant}_k{compression_factor}"

    if train_mode in {"pepsage", "general_protein"}:
        train_loader, val_loader = build_source_loaders(cfg, train_mode)
        trainer_kwargs = merge_trainer_config(cfg.trainer)
        run_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=experiment_dir / train_mode,
            trainer_kwargs=trainer_kwargs,
            stage_name=train_mode,
            final_eval_cfg=getattr(cfg, "final_evaluation", {}),
            run_final_eval=True,
        )
    elif train_mode == "pretrain_finetune":
        pretrain_train_loader, pretrain_val_loader = build_source_loaders(cfg, "general_protein")
        pretrain_trainer_kwargs = merge_trainer_config(
            cfg.trainer,
            getattr(cfg.stages.pretrain, "trainer_overrides", None),
        )
        run_stage(
            model=model,
            train_loader=pretrain_train_loader,
            val_loader=pretrain_val_loader,
            output_dir=experiment_dir / "pretrain",
            trainer_kwargs=pretrain_trainer_kwargs,
            stage_name="pretrain",
            final_eval_cfg=getattr(cfg, "final_evaluation", {}),
            run_final_eval=False,
        )

        finetune_train_loader, finetune_val_loader = build_source_loaders(cfg, "pepsage")
        finetune_trainer_kwargs = merge_trainer_config(
            cfg.trainer,
            getattr(cfg.stages.finetune, "trainer_overrides", None),
        )
        run_stage(
            model=model,
            train_loader=finetune_train_loader,
            val_loader=finetune_val_loader,
            output_dir=experiment_dir / "finetune",
            trainer_kwargs=finetune_trainer_kwargs,
            stage_name="finetune",
            final_eval_cfg=getattr(cfg, "final_evaluation", {}),
            run_final_eval=True,
        )
    else:
        raise ValueError(f"Unsupported train_mode: {cfg.train_mode}")


if __name__ == "__main__":
    main()
