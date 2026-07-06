from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from local_frame_struct_tokenizer.exp1_tokenizer.model import LocalFrameTokenizer
from local_frame_struct_tokenizer.exp1_tokenizer.pepsage_compat import (
    BBHeavyAtom,
    construct_3d_basis,
    global_to_local,
)


def _struct_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "todict"):
        return value.todict()
    return dict(value.__dict__)


def atom_slot_to_class(atom_slot: int) -> int:
    if atom_slot == int(BBHeavyAtom.CA):
        return 1
    if atom_slot in (int(BBHeavyAtom.N), int(BBHeavyAtom.C), int(BBHeavyAtom.O), int(BBHeavyAtom.OXT)):
        return 0
    return 2


class TokenizerBridge(nn.Module):
    """Simplified tokenizer bridge for exp4 StructToken-BFN.

    Only responsibilities:
    1. Load and freeze the pretrained tokenizer
    2. Build atom-level token batches from residue data
    3. Extract factored per-dimension FSQ targets
    4. Provide access to frozen quantizer and decoder for TokenStructureDecoder
    """

    def __init__(self, cfg: Any, max_num_heavyatoms: int):
        super().__init__()
        cfg_dict = _struct_to_dict(cfg)

        model_cfg = _struct_to_dict(cfg_dict.get("model"))
        loss_cfg = _struct_to_dict(cfg_dict.get("loss"))

        self.max_num_heavyatoms = int(max_num_heavyatoms)

        self.tokenizer = LocalFrameTokenizer(model_config=model_cfg, loss_config=loss_cfg)

        checkpoint_path = str(cfg_dict.get("checkpoint_path", "") or "").strip()
        if checkpoint_path:
            self._load_tokenizer_checkpoint(checkpoint_path)

        if int(self.tokenizer.model_config.get("compression_factor", 1)) != 1:
            raise ValueError("StructToken-BFN expects tokenizer compression_factor == 1.")

        self.tokenizer.eval()
        for param in self.tokenizer.parameters():
            param.requires_grad = False

        self.codebook_size = int(self.tokenizer.quantizer.codebook_size)
        self.latent_dim = int(self.tokenizer.model_config["latent_dim"])
        self.num_fsq_dims = len(self.tokenizer.quantizer.levels)
        self.fsq_levels = list(self.tokenizer.quantizer.levels)

    def _load_tokenizer_checkpoint(self, checkpoint_path: str) -> None:
        if self.tokenizer is None:
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        normalized = {}
        for key, value in state_dict.items():
            new_key = key
            stripped = True
            while stripped:
                stripped = False
                for prefix in ("model.model.", "model.", "tokenizer.", "module.", "net."):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        stripped = True
            normalized[new_key] = value
        missing, unexpected = self.tokenizer.load_state_dict(normalized, strict=False)
        if missing:
            warnings.warn(f"Tokenizer checkpoint missing keys: {missing[:8]}")
        if unexpected:
            warnings.warn(f"Tokenizer checkpoint unexpected keys: {unexpected[:8]}")

    def _build_token_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        batch_size = batch["aa"].size(0)
        device = batch["aa"].device

        tokenized_samples: list[dict[str, torch.Tensor]] = []
        max_tokens = 0
        for batch_idx in range(batch_size):
            residue_mask = batch["res_mask"][batch_idx].bool()
            residue_indices = torch.where(residue_mask)[0]
            if residue_indices.numel() == 0:
                tokenized_samples.append(
                    {
                        "coords_local": torch.zeros((0, 3), device=device),
                        "coords_global": torch.zeros((0, 3), device=device),
                        "frame_rot": torch.zeros((0, 3, 3), device=device),
                        "frame_trans": torch.zeros((0, 3), device=device),
                        "residue_type": torch.zeros((0,), dtype=torch.long, device=device),
                        "atom_slot": torch.zeros((0,), dtype=torch.long, device=device),
                        "atom_class": torch.zeros((0,), dtype=torch.long, device=device),
                        "residue_index": torch.zeros((0,), dtype=torch.long, device=device),
                        "token_mask": torch.zeros((0,), dtype=torch.bool, device=device),
                    }
                )
                continue

            aa = batch["aa"][batch_idx, residue_indices].long()
            pos_heavyatom = batch["pos_heavyatom"][batch_idx, residue_indices].float()
            mask_heavyatom = batch["mask_heavyatom"][batch_idx, residue_indices].bool()

            frames = construct_3d_basis(
                pos_heavyatom[None, :, BBHeavyAtom.CA],
                pos_heavyatom[None, :, BBHeavyAtom.C],
                pos_heavyatom[None, :, BBHeavyAtom.N],
            )[0]
            frame_trans = pos_heavyatom[:, BBHeavyAtom.CA]
            local_pos = global_to_local(frames[None], frame_trans[None], pos_heavyatom[None])[0]

            coords_local = []
            coords_global = []
            frame_rot = []
            frame_trans_tokens = []
            residue_type = []
            atom_slot = []
            atom_class = []
            full_residue_index = []

            for local_idx, full_idx in enumerate(residue_indices.tolist()):
                valid_atom_idx = torch.where(mask_heavyatom[local_idx])[0]
                for atom_idx in valid_atom_idx.tolist():
                    coords_local.append(local_pos[local_idx, atom_idx])
                    coords_global.append(pos_heavyatom[local_idx, atom_idx])
                    frame_rot.append(frames[local_idx])
                    frame_trans_tokens.append(frame_trans[local_idx])
                    residue_type.append(aa[local_idx])
                    atom_slot.append(atom_idx)
                    atom_class.append(atom_slot_to_class(atom_idx))
                    full_residue_index.append(full_idx)

            sample = {
                "coords_local": torch.stack(coords_local, dim=0),
                "coords_global": torch.stack(coords_global, dim=0),
                "frame_rot": torch.stack(frame_rot, dim=0),
                "frame_trans": torch.stack(frame_trans_tokens, dim=0),
                "residue_type": torch.stack(residue_type).long(),
                "atom_slot": torch.tensor(atom_slot, dtype=torch.long, device=device),
                "atom_class": torch.tensor(atom_class, dtype=torch.long, device=device),
                "residue_index": torch.tensor(full_residue_index, dtype=torch.long, device=device),
                "token_mask": torch.ones(len(coords_local), dtype=torch.bool, device=device),
            }
            max_tokens = max(max_tokens, sample["coords_local"].size(0))
            tokenized_samples.append(sample)

        if max_tokens == 0:
            return None

        padded: dict[str, list[torch.Tensor]] = {
            "coords_local": [],
            "coords_global": [],
            "frame_rot": [],
            "frame_trans": [],
            "residue_type": [],
            "atom_slot": [],
            "atom_class": [],
            "residue_index": [],
            "token_mask": [],
            "pad_mask": [],
        }

        def _pad_rows(tensor: torch.Tensor, pad_len: int, fill_value: float | int | bool = 0):
            if pad_len <= 0:
                return tensor
            pad_shape = (pad_len, *tensor.shape[1:])
            pad_tensor = torch.full(pad_shape, fill_value, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad_tensor], dim=0)

        for sample in tokenized_samples:
            length = sample["coords_local"].size(0)
            pad_len = max_tokens - length
            padded["coords_local"].append(_pad_rows(sample["coords_local"], pad_len, 0.0))
            padded["coords_global"].append(_pad_rows(sample["coords_global"], pad_len, 0.0))
            padded["frame_rot"].append(_pad_rows(sample["frame_rot"], pad_len, 0.0))
            padded["frame_trans"].append(_pad_rows(sample["frame_trans"], pad_len, 0.0))
            padded["residue_type"].append(_pad_rows(sample["residue_type"], pad_len, 21))
            padded["atom_slot"].append(_pad_rows(sample["atom_slot"], pad_len, 0))
            padded["atom_class"].append(_pad_rows(sample["atom_class"], pad_len, 0))
            padded["residue_index"].append(_pad_rows(sample["residue_index"], pad_len, 0))
            padded["token_mask"].append(_pad_rows(sample["token_mask"], pad_len, False))
            padded["pad_mask"].append(_pad_rows(torch.zeros(length, dtype=torch.bool, device=device), pad_len, True))

        return {key: torch.stack(values, dim=0) for key, values in padded.items()}

    @torch.no_grad()
    def prepare_factored_targets(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        """Extract per-dimension FSQ code targets from ground truth structures.

        Returns:
            factored_targets: (B, L, 6) in {0,1,2,3}, -100=ignore
            mean_codes:       (B, L, 128) projected code vectors
            valid_mask:       (B, L) bool
            gt_local_atoms:   (B, L, 15, 3) local-frame atom coordinates
        """
        token_batch = self._build_token_batch(batch)
        if token_batch is None:
            return None

        encoder_inputs = self.tokenizer.build_inputs(token_batch)
        encoding = self.tokenizer.encoder(encoder_inputs, token_batch["pad_mask"])
        compressed, c_pad_mask, _ = self.tokenizer.compress_hidden(
            encoding, token_batch["pad_mask"], token_batch["residue_index"],
        )

        z_projected = self.tokenizer.quantizer.project_in(compressed)
        z = self.tokenizer.quantizer.quantize(z_projected)

        if z.dim() == 4:
            z = z.squeeze(2)

        half_width = self.tokenizer.quantizer._levels // 2
        code_integers = ((z * half_width.float()) + half_width.float()).long().clamp(0, 3)

        quantized_out = self.tokenizer.quantizer.project_out(z)

        batch_size, num_res = batch["aa"].shape[:2]
        device = batch["aa"].device
        D = self.num_fsq_dims

        factored_targets = torch.full((batch_size, num_res, D), -100, dtype=torch.long, device=device)
        mean_codes = torch.zeros((batch_size, num_res, self.latent_dim), device=device)
        valid_mask = torch.zeros((batch_size, num_res), dtype=torch.bool, device=device)

        # build GT local atoms
        gt_local_atoms = torch.zeros((batch_size, num_res, self.max_num_heavyatoms, 3), device=device)
        gt_ca = batch["pos_heavyatom"][:, :, BBHeavyAtom.CA]
        gt_c = batch["pos_heavyatom"][:, :, BBHeavyAtom.C]
        gt_n = batch["pos_heavyatom"][:, :, BBHeavyAtom.N]
        gt_frames = construct_3d_basis(gt_ca, gt_c, gt_n)
        gt_frame_trans = gt_ca
        gt_local_pos = global_to_local(gt_frames, gt_frame_trans, batch["pos_heavyatom"])
        gt_local_atoms = gt_local_pos

        for b in range(batch_size):
            valid_tokens = token_batch["token_mask"][b]
            if valid_tokens.sum() == 0:
                continue

            residue_index = token_batch["residue_index"][b, valid_tokens]
            valid_codes = code_integers[b, ~c_pad_mask[b]]
            valid_projected = quantized_out[b, ~c_pad_mask[b]]

            n_valid = min(valid_codes.shape[0], residue_index.shape[0])
            residue_index = residue_index[:n_valid]
            valid_codes = valid_codes[:n_valid]
            valid_projected = valid_projected[:n_valid]

            for rid in residue_index.unique().tolist():
                atom_mask = (residue_index == rid)
                atom_codes = valid_codes[atom_mask]
                for d in range(D):
                    hist = torch.bincount(atom_codes[:, d], minlength=4)
                    factored_targets[b, rid, d] = hist.argmax()
                mean_codes[b, rid] = valid_projected[atom_mask].mean(0)
                valid_mask[b, rid] = True

        return {
            "factored_targets": factored_targets,
            "mean_codes": mean_codes,
            "valid_mask": valid_mask,
            "gt_local_atoms": gt_local_atoms,
        }
