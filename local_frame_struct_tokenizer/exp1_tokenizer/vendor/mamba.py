import warnings
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn

warnings.simplefilter(action="ignore", category=FutureWarning)

from mamba_ssm.models.mixer_seq_simple import _init_weights, create_block

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


@dataclass
class MambaConfig:
    d_input: int = 3
    d_output: int = 3
    d_model: int = 2560
    n_layer: int = 64
    d_intermediate: int = 0
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    norm_epsilon: float = 1e-5
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    initializer_cfg: dict = field(default_factory=dict)
    bidirectional: bool = False


class MambaStack(nn.Module):
    config_cls = MambaConfig

    def __init__(self, config: MambaConfig) -> None:
        super(MambaStack, self).__init__()
        self.config = config
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
        self.bidirectional = config.bidirectional
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        has_input_projection = config.d_input != config.d_model
        self.input_projection = nn.Linear(config.d_input, config.d_model, bias=False) if has_input_projection else nn.Identity()
        has_output_projection = config.d_output != config.d_model
        self.output_projection = nn.Linear(config.d_model, config.d_output, bias=False) if has_output_projection else nn.Identity()

        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    d_intermediate=config.d_intermediate,
                    ssm_cfg=config.ssm_cfg,
                    attn_layer_idx=config.attn_layer_idx,
                    attn_cfg=config.attn_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                )
                for i in range(config.n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(config.d_model, eps=config.norm_epsilon)

        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(config.initializer_cfg if config.initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if config.d_intermediate == 0 else 2,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None, inference_params=None, **mixer_kwargs) -> torch.Tensor:
        hidden_states = self.input_projection(input_ids)
        residual = None
        for layer in self.layers:
            if mask is not None:
                hidden_states = hidden_states * (~mask.unsqueeze(-1))
                if residual is not None:
                    residual = residual * (~mask.unsqueeze(-1))

            hidden_states_forward, residual_forward = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )

            if self.bidirectional:
                hidden_states_backward = hidden_states.flip(1)
                residual_backward = residual.flip(1) if residual is not None else None
                hidden_states_backward, residual_backward = layer(
                    hidden_states_backward, residual_backward, inference_params=inference_params, **mixer_kwargs
                )
                hidden_states_backward = hidden_states_backward.flip(1)
                residual_backward = residual_backward.flip(1)
                hidden_states = hidden_states_forward + hidden_states_backward
                residual = residual_forward + residual_backward
            else:
                hidden_states = hidden_states_forward
                residual = residual_forward

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )

        output_states = self.output_projection(hidden_states)
        if mask is not None:
            output_states = output_states * (~mask.unsqueeze(-1))
        return output_states
