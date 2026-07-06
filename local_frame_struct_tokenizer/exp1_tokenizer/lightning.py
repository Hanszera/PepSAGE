from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch

from .metrics import compute_batch_eval_metrics
from .model import LocalFrameTokenizer


DEFAULT_OPTIM_CONFIG = {
    "lr": 3.0e-4,
    "weight_decay": 0.0,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "max_epochs": 100,
}

DEFAULT_VALIDATION_CONFIG = {
    "heavy_metrics_every_n_epochs": 0,
    "heavy_metrics_max_batches": 16,
    "run_heavy_metrics_on_epoch_1": False,
}


class LocalFrameTokenizerModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        loss_config: dict[str, Any] | None = None,
        optim_config: dict[str, Any] | None = None,
        validation_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.optim_config = dict(DEFAULT_OPTIM_CONFIG)
        if optim_config is not None:
            self.optim_config.update(optim_config)
        self.validation_config = dict(DEFAULT_VALIDATION_CONFIG)
        if validation_config is not None:
            self.validation_config.update(validation_config)
        self.model = LocalFrameTokenizer(model_config=model_config, loss_config=loss_config)
        self._full_evaluation_mode = False
        self._full_evaluation_max_batches = 0
        self._full_evaluation_log_prefix = "full_eval"
        self.save_hyperparameters(
            {
                "model": model_config or {},
                "loss": loss_config or {},
                "optim": self.optim_config,
                "validation": self.validation_config,
            }
        )

    def enable_full_evaluation(self, max_batches: int = 0, log_prefix: str = "full_eval") -> None:
        self._full_evaluation_mode = True
        self._full_evaluation_max_batches = max_batches
        self._full_evaluation_log_prefix = log_prefix

    def disable_full_evaluation(self) -> None:
        self._full_evaluation_mode = False
        self._full_evaluation_max_batches = 0
        self._full_evaluation_log_prefix = "full_eval"

    def _metric_stage_name(self, stage: str) -> str:
        if stage == "val" and self._full_evaluation_mode:
            return self._full_evaluation_log_prefix
        return stage

    def _should_run_heavy_val_metrics(self, batch_idx: int) -> bool:
        if self.trainer is not None and self.trainer.sanity_checking:
            return False

        if self._full_evaluation_mode:
            max_batches = int(self._full_evaluation_max_batches)
            return max_batches <= 0 or batch_idx < max_batches

        every_n_epochs = int(self.validation_config["heavy_metrics_every_n_epochs"])
        if every_n_epochs <= 0:
            return False

        epoch_one_based = int(self.current_epoch) + 1
        if epoch_one_based == 1 and not bool(self.validation_config["run_heavy_metrics_on_epoch_1"]):
            return False
        if epoch_one_based % every_n_epochs != 0:
            return False

        max_batches = int(self.validation_config["heavy_metrics_max_batches"])
        if max_batches > 0 and batch_idx >= max_batches:
            return False
        return True

    def _should_sync_dist(self) -> bool:
        if self.trainer is not None and int(getattr(self.trainer, "world_size", 1)) > 1:
            return True
        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )

    def _log_metric_dict(self, metrics: dict[str, torch.Tensor], stage: str, batch_size: int) -> None:
        sync_dist = self._should_sync_dist()
        for key, value in metrics.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.numel() != 1 or not torch.isfinite(value).all():
                continue
            self.log(
                f"{stage}/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str, batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch)
        losses = outputs["losses"]
        batch_size = batch["coords_local"].size(0)
        metric_stage = self._metric_stage_name(stage)
        sync_dist = self._should_sync_dist()
        self.log(
            f"{metric_stage}/loss",
            losses["loss"],
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        self._log_metric_dict({k: v for k, v in losses.items() if k != "loss"}, metric_stage, batch_size)

        if stage == "val" and self._should_run_heavy_val_metrics(batch_idx):
            eval_metrics = compute_batch_eval_metrics(
                batch=batch,
                pred_global=outputs["pred_global"],
                indices=outputs["indices"],
                codebook_size=self.model.quantizer.codebook_size,
                quantizer_mask=outputs["quantizer_mask"],
            )
            self._log_metric_dict(eval_metrics, metric_stage, batch_size)
        return losses["loss"]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val", batch_idx)

    def configure_optimizers(self):
        lr = float(self.optim_config["lr"])
        weight_decay = float(self.optim_config["weight_decay"])
        optimizer_name = str(self.optim_config["optimizer"]).lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_name = str(self.optim_config["scheduler"]).lower()
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.optim_config["max_epochs"]),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer
