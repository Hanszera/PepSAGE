from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import (
    bond_angle_deviation,
    bond_length_deviation,
    clash_score,
    codebook_summary,
    reconstruct_sample_pos14,
)
from .vendor import FSQ, FSQConfig, MambaConfig, MambaStack

from .pepsage_compat import global_to_local

DEFAULT_MODEL_CONFIG = {
    "residue_vocab_size": 22,
    "atom_slot_vocab_size": 16,
    "atom_class_vocab_size": 4,
    "residue_embed_dim": 32,
    "atom_slot_embed_dim": 16,
    "atom_class_embed_dim": 8,
    "latent_dim": 128,
    "model_dim": 128,
    "encoder_layers": 4,
    "decoder_layers": 6,
    "bidirectional": True,
    "fused_add_norm": False,
    "rms_norm": True,
    "quantizer_levels": [4, 4, 4, 4, 4, 4],
    "use_local_frame": True,
    "use_residue_type": True,
    "use_atom_slot": True,
    "use_atom_class": True,
    "compression_factor": 1,
}

DEFAULT_LOSS_CONFIG = {
    "local_mse_weight": 1.0,
    "global_mse_weight": 1.0,
    "intra_residue_distance_weight": 0.5,
    "bond_length_weight": 0.0,
    "bond_angle_weight": 0.0,
    "clash_weight": 0.0,
}


def with_defaults(config: dict[str, Any] | None, defaults: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    if config is not None:
        merged.update(config)
    return merged


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq = ((pred - target) ** 2).sum(dim=-1)
    return torch.sqrt(masked_mean(sq, mask) + 1e-8)

class LocalFrameTokenizer(nn.Module):
    def __init__(self, model_config: dict[str, Any] | None = None, loss_config: dict[str, Any] | None = None):
        super().__init__()
        self.model_config = with_defaults(model_config, DEFAULT_MODEL_CONFIG)
        self.loss_config = with_defaults(loss_config, DEFAULT_LOSS_CONFIG)

        feature_dim = 3
        if self.model_config["use_residue_type"]:
            feature_dim += self.model_config["residue_embed_dim"]
        if self.model_config["use_atom_slot"]:
            feature_dim += self.model_config["atom_slot_embed_dim"]
        if self.model_config["use_atom_class"]:
            feature_dim += self.model_config["atom_class_embed_dim"]
        latent_dim = int(self.model_config["latent_dim"])
        model_dim = int(self.model_config["model_dim"])
        self.compression_factor = int(self.model_config["compression_factor"])

        self.residue_embed = nn.Embedding(
            int(self.model_config["residue_vocab_size"]),
            int(self.model_config["residue_embed_dim"]),
        )
        self.atom_slot_embed = nn.Embedding(
            int(self.model_config["atom_slot_vocab_size"]),
            int(self.model_config["atom_slot_embed_dim"]),
        )
        self.atom_class_embed = nn.Embedding(
            int(self.model_config["atom_class_vocab_size"]),
            int(self.model_config["atom_class_embed_dim"]),
        )

        quantizer_cfg = FSQConfig(
            levels=list(self.model_config["quantizer_levels"]),
            d_input=latent_dim,
        )

        encoder_cfg = MambaConfig(
            d_input=feature_dim,
            d_output=latent_dim,
            d_model=model_dim,
            n_layer=int(self.model_config["encoder_layers"]),
            bidirectional=bool(self.model_config["bidirectional"]),
            fused_add_norm=bool(self.model_config["fused_add_norm"]),
            rms_norm=bool(self.model_config["rms_norm"]),
        )
        self.quantizer = FSQ(quantizer_cfg)
        decoder_cfg = MambaConfig(
            d_input=latent_dim,
            d_output=3,
            d_model=model_dim,
            n_layer=int(self.model_config["decoder_layers"]),
            bidirectional=bool(self.model_config["bidirectional"]),
            fused_add_norm=bool(self.model_config["fused_add_norm"]),
            rms_norm=bool(self.model_config["rms_norm"]),
        )
        self.encoder = MambaStack(encoder_cfg)
        self.decoder = MambaStack(decoder_cfg)

    def input_coords(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.model_config["use_local_frame"]:
            return batch["coords_local"]
        return batch["coords_global"]

    def build_inputs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = [self.input_coords(batch)]
        if self.model_config["use_residue_type"]:
            features.append(self.residue_embed(batch["residue_type"]))
        if self.model_config["use_atom_slot"]:
            features.append(self.atom_slot_embed(batch["atom_slot"]))
        if self.model_config["use_atom_class"]:
            features.append(self.atom_class_embed(batch["atom_class"]))
        return torch.cat(features, dim=-1)

    def compress_hidden(
        self,
        hidden: torch.Tensor,
        pad_mask: torch.Tensor,
        residue_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = hidden.shape
        gather_index = torch.zeros((bsz, seq_len), dtype=torch.long, device=hidden.device)
        if self.compression_factor <= 1:
            return hidden, pad_mask, gather_index + torch.arange(seq_len, device=hidden.device).view(1, -1)

        compressed_samples = []
        compressed_lengths = []
        k = self.compression_factor

        for batch_idx in range(bsz):
            valid_positions = torch.where(~pad_mask[batch_idx])[0]
            if valid_positions.numel() == 0:
                compressed_samples.append(hidden.new_zeros((0, dim)))
                compressed_lengths.append(0)
                continue

            hidden_valid = hidden[batch_idx, valid_positions]
            residue_valid = residue_index[batch_idx, valid_positions]
            sample_chunks = []
            current_index = 0
            cursor = 0
            while cursor < hidden_valid.size(0):
                residue_id = int(residue_valid[cursor].item())
                residue_end = cursor + 1
                while residue_end < hidden_valid.size(0) and int(residue_valid[residue_end].item()) == residue_id:
                    residue_end += 1

                residue_hidden = hidden_valid[cursor:residue_end]
                residue_positions = valid_positions[cursor:residue_end]
                for chunk_start in range(0, residue_hidden.size(0), k):
                    chunk_end = min(chunk_start + k, residue_hidden.size(0))
                    sample_chunks.append(residue_hidden[chunk_start:chunk_end].mean(dim=0))
                    gather_index[batch_idx, residue_positions[chunk_start:chunk_end]] = current_index
                    current_index += 1
                cursor = residue_end

            compressed_sample = torch.stack(sample_chunks, dim=0) if sample_chunks else hidden.new_zeros((0, dim))
            compressed_samples.append(compressed_sample)
            compressed_lengths.append(compressed_sample.size(0))

        max_len = max(compressed_lengths, default=0)
        compressed = hidden.new_zeros((bsz, max_len, dim))
        compressed_mask = torch.ones((bsz, max_len), dtype=torch.bool, device=hidden.device)
        for batch_idx, compressed_sample in enumerate(compressed_samples):
            length = compressed_sample.size(0)
            if length == 0:
                continue
            compressed[batch_idx, :length] = compressed_sample
            compressed_mask[batch_idx, :length] = False
        return compressed, compressed_mask, gather_index

    def expand_hidden(
        self,
        hidden: torch.Tensor,
        gather_index: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.size(1) == 0:
            return hidden.new_zeros((hidden.size(0), gather_index.size(1), hidden.size(-1)))
        expanded = hidden.gather(
            dim=1,
            index=gather_index.unsqueeze(-1).expand(-1, -1, hidden.size(-1)),
        )
        return expanded * token_mask.unsqueeze(-1).float()

    def decode_global(self, pred_local: torch.Tensor, frame_rot: torch.Tensor, frame_trans: torch.Tensor) -> torch.Tensor:
        rotated = torch.einsum("bnij,bnj->bni", frame_rot, pred_local)
        return rotated + frame_trans

    def intra_residue_distance_loss(
        self,
        pred_local: torch.Tensor,
        target_local: torch.Tensor,
        residue_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        batch_size = pred_local.size(0)
        for batch_idx in range(batch_size):
            valid_mask = mask[batch_idx]
            if valid_mask.sum() < 2:
                continue
            pred_valid = pred_local[batch_idx][valid_mask]
            target_valid = target_local[batch_idx][valid_mask]
            residue_valid = residue_index[batch_idx][valid_mask]
            for residue_id in residue_valid.unique():
                residue_mask = residue_valid == residue_id
                if residue_mask.sum() < 2:
                    continue
                pred_dist = torch.cdist(pred_valid[residue_mask], pred_valid[residue_mask], p=2)
                target_dist = torch.cdist(target_valid[residue_mask], target_valid[residue_mask], p=2)
                losses.append(F.mse_loss(pred_dist, target_dist))
        if not losses:
            return pred_local.new_tensor(0.0)
        return torch.stack(losses).mean()

    def chemistry_regularization(
        self,
        batch: dict[str, torch.Tensor],
        pred_global: torch.Tensor,
        target_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bond_len_values = []
        bond_ang_values = []
        clash_values = []

        for batch_idx in range(pred_global.size(0)):
            valid = batch["token_mask"][batch_idx].bool()
            pos14_pred, mask14, aa = reconstruct_sample_pos14(
                residue_index=batch["residue_index"][batch_idx],
                atom_slot=batch["atom_slot"][batch_idx],
                residue_type=batch["residue_type"][batch_idx],
                coords=pred_global[batch_idx],
                mask=valid,
            )
            pos14_target, _, _ = reconstruct_sample_pos14(
                residue_index=batch["residue_index"][batch_idx],
                atom_slot=batch["atom_slot"][batch_idx],
                residue_type=batch["residue_type"][batch_idx],
                coords=target_global[batch_idx],
                mask=valid,
            )
            if pos14_pred.size(0) == 0:
                continue
            if (aa < 20).any():
                bond_len = bond_length_deviation(pos14_pred, pos14_target, mask14, aa)
                if not torch.isnan(bond_len):
                    bond_len_values.append(bond_len)
                bond_ang = bond_angle_deviation(pos14_pred, pos14_target, mask14, aa)
                if not torch.isnan(bond_ang):
                    bond_ang_values.append(torch.deg2rad(bond_ang))
                clash = clash_score(pos14_pred, mask14, aa)
                if not torch.isnan(clash):
                    clash_values.append(clash)

        device = pred_global.device
        zero = torch.tensor(0.0, device=device)
        return {
            "bond_length_loss": torch.stack(bond_len_values).mean() if bond_len_values else zero,
            "bond_angle_loss": torch.stack(bond_ang_values).mean() if bond_ang_values else zero,
            "clash_loss": torch.stack(clash_values).mean() if clash_values else zero,
        }

    def compute_losses(
        self,
        batch: dict[str, torch.Tensor],
        pred_local: torch.Tensor,
        pred_global: torch.Tensor,
        pred_repr: torch.Tensor,
        indices: torch.Tensor,
        quantizer_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = batch["token_mask"]
        target_repr = self.input_coords(batch)
        repr_sq = ((pred_repr - target_repr) ** 2).sum(dim=-1)
        local_sq = ((pred_local - batch["coords_local"]) ** 2).sum(dim=-1)
        global_sq = ((pred_global - batch["coords_global"]) ** 2).sum(dim=-1)

        repr_mse = masked_mean(repr_sq, mask)
        local_mse = masked_mean(local_sq, mask)
        global_mse = masked_mean(global_sq, mask)
        distance_weight = float(self.loss_config["intra_residue_distance_weight"])
        bond_length_weight = float(self.loss_config["bond_length_weight"])
        bond_angle_weight = float(self.loss_config["bond_angle_weight"])
        clash_weight = float(self.loss_config["clash_weight"])

        if distance_weight > 0.0:
            distance_loss = self.intra_residue_distance_loss(
                pred_local=pred_local,
                target_local=batch["coords_local"],
                residue_index=batch["residue_index"],
                mask=mask,
            )
        else:
            distance_loss = pred_local.new_tensor(0.0)

        if bond_length_weight > 0.0 or bond_angle_weight > 0.0 or clash_weight > 0.0:
            chemistry_losses = self.chemistry_regularization(
                batch=batch,
                pred_global=pred_global,
                target_global=batch["coords_global"],
            )
        else:
            zero = pred_global.new_tensor(0.0)
            chemistry_losses = {
                "bond_length_loss": zero,
                "bond_angle_loss": zero,
                "clash_loss": zero,
            }

        loss = (
            float(self.loss_config["local_mse_weight"]) * local_mse
            + float(self.loss_config["global_mse_weight"]) * global_mse
            + distance_weight * distance_loss
            + bond_length_weight * chemistry_losses["bond_length_loss"]
            + bond_angle_weight * chemistry_losses["bond_angle_loss"]
            + clash_weight * chemistry_losses["clash_loss"]
        )
        codebook = codebook_summary(indices, ~quantizer_mask, self.quantizer.codebook_size)

        return {
            "loss": loss,
            "repr_mse": repr_mse,
            "local_mse": local_mse,
            "global_mse": global_mse,
            "distance_loss": distance_loss,
            "bond_length_loss": chemistry_losses["bond_length_loss"],
            "bond_angle_loss": chemistry_losses["bond_angle_loss"],
            "clash_loss": chemistry_losses["clash_loss"],
            "local_rmse": masked_rmse(pred_local, batch["coords_local"], mask),
            "global_rmse": masked_rmse(pred_global, batch["coords_global"], mask),
            **codebook,
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        pad_mask = batch["pad_mask"]
        encoder_inputs = self.build_inputs(batch)
        encoding = self.encoder(encoder_inputs, pad_mask)
        compressed_encoding, compressed_pad_mask, gather_index = self.compress_hidden(
            encoding,
            pad_mask,
            batch["residue_index"],
        )
        quantized, indices = self.quantizer(compressed_encoding)
        quantized = quantized.masked_fill(compressed_pad_mask.unsqueeze(-1), 0.0)
        indices = indices.masked_fill(compressed_pad_mask, self.quantizer.codebook_size)
        decoded_compressed = self.decoder(quantized, compressed_pad_mask)
        pred_repr = self.expand_hidden(decoded_compressed, gather_index, batch["token_mask"])

        if self.model_config["use_local_frame"]:
            pred_local = pred_repr
            pred_global = self.decode_global(pred_local, batch["frame_rot"], batch["frame_trans"])
        else:
            pred_global = pred_repr
            pred_local = global_to_local(batch["frame_rot"], batch["frame_trans"], pred_global)

        outputs = dict(batch)
        outputs["encoding"] = encoding
        outputs["compressed_encoding"] = compressed_encoding
        outputs["quantizer_mask"] = compressed_pad_mask
        outputs["compression_gather_index"] = gather_index
        outputs["quantized"] = quantized
        outputs["indices"] = indices
        outputs["pred_repr"] = pred_repr
        outputs["pred_local"] = pred_local
        outputs["pred_global"] = pred_global
        outputs["losses"] = self.compute_losses(
            batch=batch,
            pred_local=pred_local,
            pred_global=pred_global,
            pred_repr=pred_repr,
            indices=indices,
            quantizer_mask=compressed_pad_mask,
        )
        return outputs
