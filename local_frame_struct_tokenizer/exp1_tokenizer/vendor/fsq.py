from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor, int32
from torch.nn import Module


@dataclass
class FSQConfig:
    levels: List[int] = field(default_factory=lambda: [4, 4, 4, 4, 4, 4])
    num_codebooks: int = 1
    d_input: Optional[int] = None
    keep_num_codebooks_dim: Optional[bool] = None


def default(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


class FSQ(Module):
    config_cls = FSQConfig

    def __init__(self, config: FSQConfig):
        super(FSQ, self).__init__()
        self.config = config

        self.levels = self.config.levels
        _levels = torch.tensor(self.levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + self.levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.codebook_dim = len(self.levels)
        self.num_codebooks = self.config.num_codebooks
        self.effective_codebook_dim = self.codebook_dim * self.num_codebooks

        self.keep_num_codebooks_dim = default(self.config.keep_num_codebooks_dim, self.config.num_codebooks > 1)
        assert not (self.config.num_codebooks > 1 and not self.keep_num_codebooks_dim)

        self.d_input = default(self.config.d_input, self.effective_codebook_dim)
        self.has_projections = self.d_input != self.effective_codebook_dim
        self.project_in = nn.Linear(self.d_input, self.effective_codebook_dim) if self.has_projections else nn.Identity()
        self.project_out = nn.Linear(self.effective_codebook_dim, self.d_input) if self.has_projections else nn.Identity()

        self.codebook_size = self._levels.prod().item()

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        z = self.bound(z)
        quantized = z + (z.round() - z).detach()
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor, project_out: bool = True) -> Tensor:
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        return codes

    def forward(self, z: Tensor, with_hidden_codes: bool = False):
        assert z.shape[-1] == self.d_input, f"expected dimension of {self.d_input} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        codes = rearrange(codes, "b n c d -> b n (c d)")
        out = self.project_out(codes)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        if with_hidden_codes:
            return out, indices, codes
        return out, indices
