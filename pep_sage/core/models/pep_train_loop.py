import os
import torch
import pytorch_lightning as pl
from core.config.config import Config
from core.models.bfn_pep import BFNModel
from core.utils.train import get_optimizer, get_scheduler, sum_weighted_losses
from core.utils.data import repeat_batch
import time


class PepTrainLoop(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.dynamics = BFNModel(self.cfg.dynamics)
        self._ensure_loss_weight_defaults()
        self.save_hyperparameters(self.cfg.todict())

    def forward(self):
        pass

    def _ensure_loss_weight_defaults(self):
        weights = self.cfg.train.loss_weights
        defaults = {
            "trans_loss": 1.0,
            "rot_loss": 0.1,
            "token_bfn_loss": 1.0,
            "aa_aux_loss": 0.5,
            "structure_decode_loss": 1.0,
            "bb_atom_loss": 0.0,
        }
        for key, default_val in defaults.items():
            if not hasattr(weights, key):
                setattr(weights, key, default_val)

    def training_step(self, batch, batch_idx):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        step_start = time.perf_counter()
        loss_dict = self.dynamics(batch)

        loss = sum_weighted_losses(loss_dict, self.cfg.train.loss_weights)
        step_time_ms = (time.perf_counter() - step_start) * 1000.0
        memory_mb = 0.0
        if self.device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)

        self.log_dict(
            {
                'lr': self.get_last_lr(),
                'train/loss': loss.detach(),
                'train/step_time_ms': torch.tensor(step_time_ms, device=loss.device),
                'train/max_memory_mb': torch.tensor(memory_mb, device=loss.device),
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=self.cfg.train.batch_size,
            sync_dist=True
        )
        self.log_dict(
            {
                f'train/{k}': v.detach() for k, v in loss_dict.items()
            },
            on_step=True,
            on_epoch=False,
            batch_size=self.cfg.train.batch_size,
            sync_dist=True
        )

        if not torch.isfinite(loss):
            return None

        return loss

    def validation_step(self, batch, batch_idx):
        sample_steps = int(getattr(self.cfg.train, "val_sample_steps", 100))
        sample_start = time.perf_counter()
        error = self.dynamics.fix_seq_sample(
            batch, num_steps=sample_steps,
            pos_norm=self.cfg.train.normalizer_dict.pos,
        )
        error["val/sample_runtime_s"] = time.perf_counter() - sample_start

        self.log_dict(
            error,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.cfg.train.batch_size,
            sync_dist=True
        )

        sum_batches, sum_loss = 0, 0.0
        loss_sums = {}
        num_graphs = batch['pos_heavyatom'].shape[0]
        for t_val in range(0, self.dynamics.discrete_steps, 10):
            sum_batches += 1
            t = torch.tensor(
                [t_val / float(self.cfg.dynamics.discrete_steps)],
                device=batch['pos_heavyatom'].device
            ).repeat(num_graphs, 1)

            if not self.cfg.dynamics.use_discrete_t and not self.cfg.dynamics.destination_prediction:
                t = torch.clamp(t, min=self.dynamics.t_min)

            loss_dict = self.dynamics(batch, t)
            loss = sum_weighted_losses(loss_dict, self.cfg.train.loss_weights)
            sum_loss += float(loss)
            for key, value in loss_dict.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value)

        recon_loss = {"val/recon_loss": sum_loss / sum_batches}
        legacy_names = {
            "trans_loss": "val/recon_loss_trans",
            "rot_loss": "val/recon_loss_rots",
            "token_bfn_loss": "val/recon_loss_token",
            "aa_aux_loss": "val/recon_loss_aa",
            "structure_decode_loss": "val/recon_loss_struct",
            "bb_atom_loss": "val/recon_loss_bb_atom",
        }
        for key, total in loss_sums.items():
            recon_loss[legacy_names.get(key, f"val/recon_{key}")] = total / sum_batches
        self.log_dict(
            recon_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.cfg.train.batch_size,
            sync_dist=True
        )
        return recon_loss["val/recon_loss"]

    def configure_optimizers(self):
        self.optim = get_optimizer(self.cfg.train.optimizer, self)
        self.scheduler, self.get_last_lr = get_scheduler(self.cfg.train, self.optim)
        return {
            'optimizer': self.optim,
            'lr_scheduler': self.scheduler,
        }

    def on_test_epoch_start(self):
        os.makedirs(self.cfg.accounting.test_outputs_dir, exist_ok=True)
        self.dic = {'id': [], 'len': [], 'tran': [], 'aar': [], 'rot': [], 'trans_loss': [], 'rot_loss': []}

    def test_step(self, batch, batch_idx):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        sample_start = time.perf_counter()
        batch_repeat = repeat_batch(batch, self.cfg.num_samples)
        traj_1 = self.dynamics.sample(
            batch_repeat, num_steps=self.cfg.sample_steps,
            pos_norm=self.cfg.train.normalizer_dict.pos,
        )
        sample_runtime_s = time.perf_counter() - sample_start
        sample_memory_mb = 0.0
        if self.device.type == "cuda":
            sample_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
        self.log_dict(
            {
                "test/sample_runtime_s": torch.tensor(sample_runtime_s, device=self.device),
                "test/sample_memory_mb": torch.tensor(sample_memory_mb, device=self.device),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
            sync_dist=True,
        )
        torch.save(traj_1, f'{self.cfg.accounting.test_outputs_dir}/{batch["id"][0]}_batchid_{batch_idx}.pt')
