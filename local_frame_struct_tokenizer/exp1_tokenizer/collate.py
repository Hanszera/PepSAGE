from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate


class LocalTokenizerCollate:
    def __init__(self, pad_to_multiple_of: int = 8):
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_values = {
            "coords_local": 0.0,
            "coords_global": 0.0,
            "frame_rot": 0.0,
            "frame_trans": 0.0,
            "residue_type": 21,
            "atom_slot": 15,
            "atom_class": 3,
            "residue_index": -1,
            "token_mask": False,
        }

    @staticmethod
    def _pad_tensor(x: torch.Tensor, n: int, value: Any) -> torch.Tensor:
        if x.size(0) == n:
            return x
        pad_shape = [n - x.size(0)] + list(x.shape[1:])
        pad = torch.full(pad_shape, fill_value=value, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=0)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(sample["coords_local"].size(0) for sample in batch)
        if self.pad_to_multiple_of > 1:
            max_len = math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of

        padded = []
        for sample in batch:
            padded_sample = {}
            for key, value in sample.items():
                if key == "sample_id":
                    padded_sample[key] = value
                    continue
                if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.size(0) <= max_len:
                    padded_sample[key] = self._pad_tensor(value, max_len, self.pad_values.get(key, 0))
                else:
                    padded_sample[key] = value
            padded_sample["pad_mask"] = ~padded_sample["token_mask"]
            padded.append(padded_sample)

        sample_ids = [item.pop("sample_id") for item in padded]
        batch_out = default_collate(padded)
        batch_out["sample_id"] = sample_ids
        return batch_out
