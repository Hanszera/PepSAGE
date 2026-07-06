# # Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

from typing import Dict, Any, Optional
import torch
import torch.distributed as dist
from torch import nn
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

class EMA(Callback):
    """Exponential Moving Average (EMA) callback for model parameters.
    
    Maintains moving averages of model parameters using exponential decay.
    Automatically handles multi-GPU synchronization and checkpoint saving/loading.
    
    Args:
        decay: Decay factor for EMA (between 0 and 1). Higher means more smoothing.
        every_n_steps: Update EMA weights every N steps.
        ema_device: Device to store EMA weights on. None uses model device.
        sync_every_epoch: Whether to sync EMA across GPUs every epoch.
        strict_load: Whether to enforce matching parameter names when loading.
    """
    
    def __init__(
        self,
        decay: float = 0.999,
        every_n_steps: int = 1,
        ema_device: Optional[torch.device] = None,
        sync_every_epoch: bool = True,
        strict_load: bool = True,
    ):
        self.decay = decay
        self.every_n_steps = every_n_steps
        self.ema_device = ema_device
        self.sync_every_epoch = sync_every_epoch
        self.strict_load = strict_load
        
        self.ema_params: Dict[str, torch.Tensor] = {}
        self.backup_params: Dict[str, torch.Tensor] = {}
        self.step = 0
        self.initialized = False
        self._swapped = False

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        """Initialize EMA parameters from model parameters."""
        device = self.ema_device if self.ema_device is not None else pl_module.device
        if self.initialized:
            self.ema_params = {
                k: v.to(device)
                for k, v in self.ema_params.items()
            }
            return
            
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    self.ema_params[name] = param.detach().clone().to(device)
        
        self.initialized = True

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA weights every N steps."""
        self.step += 1
        if self.step % self.every_n_steps != 0:
            return
            
        device = self.ema_device if self.ema_device is not None else pl_module.device
        decay = self.decay ** self.every_n_steps  # Compensate for skipped steps
        
        for name, param in pl_module.named_parameters():
            if name in self.ema_params and param.requires_grad:
                ema = self.ema_params[name]
                param_data = param.data.to(device)
                ema.mul_(decay).add_(param_data, alpha=1 - decay)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Swap model parameters with EMA before validation."""
        self._swap_to_ema(pl_module)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore original parameters after validation."""
        self._restore_from_backup(pl_module)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Sync EMA across GPUs at epoch end if enabled."""
        if self.sync_every_epoch:
            self._sync_ema_across_gpus(pl_module)

    @rank_zero_only
    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Save EMA state when checkpointing."""
        checkpoint['ema_params'] = {
            k: v.cpu().clone() for k, v in self.ema_params.items()
        }
        checkpoint['step'] = self.step
        checkpoint['initialized'] = self.initialized

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        callback_state: Dict[str, Any],
    ) -> None:
        """
        Load EMA state from checkpoint. 
        NOTE: Only works in training and validation.
        """
        
        if self.strict_load:
            # Verify all current parameters exist in checkpoint
            current_params = set(
                name for name, p in pl_module.named_parameters() if p.requires_grad
            )
            saved_params = set(callback_state["ema_params"].keys())
            if current_params != saved_params:
                missing = current_params - saved_params
                extra = saved_params - current_params
                raise RuntimeError(
                    f"Parameter mismatch when loading EMA. Missing: {missing}, Extra: {extra}"
                )
        
        self.ema_params = {
            k: v.clone()
            for k, v in callback_state["ema_params"].items()
        }
        self.step = callback_state["step"]
        self.initialized = callback_state.get("initialized", False)
        self._swapped = False

    def _swap_to_ema(self, pl_module: LightningModule) -> None:
        """Replace model parameters with EMA versions."""
        if self._swapped or not self.initialized:
            return
            
        self.backup_params.clear()
        
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if name in self.ema_params and param.requires_grad:
                    self.backup_params[name] = param.data.clone()
                    param.data.copy_(self.ema_params[name].to(param.device))
        
        self._swapped = True

    def _restore_from_backup(self, pl_module: LightningModule) -> None:
        """Restore original model parameters from backup."""
        if not self._swapped or not self.backup_params:
            return
            
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if name in self.backup_params:
                    param.data.copy_(self.backup_params[name].to(param.device))
        
        self.backup_params.clear()
        self._swapped = False

    def _sync_ema_across_gpus(self, pl_module: LightningModule) -> None:
        """Average EMA parameters across all GPUs."""
        if not (dist.is_available() and dist.is_initialized()):
            return
            
        def _sync(ema: torch.Tensor) -> torch.Tensor:
            dist.all_reduce(ema, op=dist.ReduceOp.SUM)
            return ema / dist.get_world_size()
        
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if name in self.ema_params and param.requires_grad:
                    self.ema_params[name] = _sync(self.ema_params[name])