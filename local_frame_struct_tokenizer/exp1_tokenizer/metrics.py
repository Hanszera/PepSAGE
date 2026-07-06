from __future__ import annotations

import contextlib
import functools
import math
from typing import Any

import torch
import torch.nn.functional as F

from .pepsage_compat import get_torsion_angle
from .pepsage_compat import constants as protein_constants


BACKBONE_SLOTS = {0, 1, 2, 3, 14}
CHI_METRIC_NAMES = ["chi1", "chi2", "chi3", "chi4"]


def circular_abs_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def safe_mean(values: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not values:
        return torch.tensor(float("nan"), device=device)
    return torch.stack(values).mean()


def compute_codebook_histogram(indices: torch.Tensor, mask: torch.Tensor, codebook_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    valid = indices[mask]
    if valid.numel() == 0:
        hist = torch.zeros(codebook_size, device=indices.device)
        return hist, valid
    hist = torch.bincount(valid.reshape(-1), minlength=codebook_size).float()
    return hist, valid


def codebook_summary(indices: torch.Tensor, mask: torch.Tensor, codebook_size: int) -> dict[str, torch.Tensor]:
    hist, valid = compute_codebook_histogram(indices, mask, codebook_size)
    if valid.numel() == 0:
        zero = torch.tensor(0.0, device=indices.device)
        return {
            "codebook_perplexity": zero,
            "codebook_usage": zero,
            "dead_codes": torch.tensor(float(codebook_size), device=indices.device),
            "dead_code_fraction": torch.tensor(1.0, device=indices.device),
        }

    probs = hist / hist.sum().clamp_min(1.0)
    nonzero = probs > 0
    entropy = -(probs[nonzero] * probs[nonzero].log()).sum()
    dead_codes = (~nonzero).sum().float()
    return {
        "codebook_perplexity": entropy.exp(),
        "codebook_usage": nonzero.float().mean(),
        "dead_codes": dead_codes,
        "dead_code_fraction": dead_codes / float(codebook_size),
    }


def kabsch_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    pred_center = pred.mean(dim=0, keepdim=True)
    target_center = target.mean(dim=0, keepdim=True)
    pred0 = pred - pred_center
    target0 = target - target_center
    covariance = pred0.transpose(0, 1) @ target0
    u, _, vt = torch.linalg.svd(covariance)
    rotation = vt.transpose(0, 1) @ u.transpose(0, 1)
    if torch.det(rotation) < 0:
        vt = vt.clone()
        vt[-1] *= -1
        rotation = vt.transpose(0, 1) @ u.transpose(0, 1)
    aligned = pred0 @ rotation + target_center
    return aligned


def rmsd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(torch.sum((pred - target) ** 2, dim=-1)).clamp_min(1e-8))


def sample_length_bucket(num_residues: int) -> str:
    if num_residues <= 8:
        return "short"
    if num_residues <= 16:
        return "medium"
    return "long"


def reconstruct_sample_pos14(
    residue_index: torch.Tensor,
    atom_slot: torch.Tensor,
    residue_type: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = mask.bool()
    if valid.sum() == 0:
        return (
            coords.new_zeros((0, 15, 3)),
            torch.zeros((0, 15), dtype=torch.bool, device=coords.device),
            torch.zeros((0,), dtype=torch.long, device=coords.device),
        )

    residue_index = residue_index[valid].long()
    atom_slot = atom_slot[valid].long()
    residue_type = residue_type[valid].long()
    coords = coords[valid]

    num_residues = int(residue_index.max().item()) + 1
    pos14 = coords.new_zeros((num_residues, 15, 3))
    mask14 = torch.zeros((num_residues, 15), dtype=torch.bool, device=coords.device)
    aa = torch.full((num_residues,), 21, dtype=torch.long, device=coords.device)

    for idx in range(coords.size(0)):
        res_idx = int(residue_index[idx].item())
        atom_idx = int(atom_slot[idx].item())
        if atom_idx < 0 or atom_idx >= 15:
            continue
        pos14[res_idx, atom_idx] = coords[idx]
        mask14[res_idx, atom_idx] = True
        aa[res_idx] = residue_type[idx]

    return pos14, mask14, aa


@functools.lru_cache(maxsize=None)
def heavy_bond_pairs(aa_idx: int) -> tuple[tuple[int, int], ...]:
    if aa_idx >= int(protein_constants.AA.UNK):
        return tuple()
    restype = protein_constants.AA(aa_idx)
    name_to_idx = protein_constants.restype_atom14_name_to_index[restype]
    pairs = []
    for atom1_name, atom2_name, _ in protein_constants.restype_to_bonded_atom_name_pairs[restype]:
        if atom1_name in name_to_idx and atom2_name in name_to_idx:
            pairs.append((name_to_idx[atom1_name], name_to_idx[atom2_name]))
    return tuple(sorted(set(tuple(sorted(pair)) for pair in pairs)))


@functools.lru_cache(maxsize=None)
def heavy_angle_triplets(aa_idx: int) -> tuple[tuple[int, int, int], ...]:
    pairs = heavy_bond_pairs(aa_idx)
    neighbors: dict[int, set[int]] = {}
    for i, j in pairs:
        neighbors.setdefault(i, set()).add(j)
        neighbors.setdefault(j, set()).add(i)

    triplets = set()
    for center, nbrs in neighbors.items():
        nbr_list = sorted(nbrs)
        for idx in range(len(nbr_list)):
            for jdx in range(idx + 1, len(nbr_list)):
                left = nbr_list[idx]
                right = nbr_list[jdx]
                triplets.add((left, center, right))
    return tuple(sorted(triplets))


def angle_value(points: torch.Tensor) -> torch.Tensor:
    v1 = points[0] - points[1]
    v2 = points[2] - points[1]
    v1 = v1 / v1.norm().clamp_min(1e-8)
    v2 = v2 / v2.norm().clamp_min(1e-8)
    cos_theta = torch.clamp((v1 * v2).sum(), min=-0.999999, max=0.999999)
    return torch.acos(cos_theta)


def bond_length_deviation(pred_pos14: torch.Tensor, target_pos14: torch.Tensor, mask14: torch.Tensor, aa: torch.Tensor) -> torch.Tensor:
    deviations = []
    for res_idx in range(aa.size(0)):
        aa_idx = int(aa[res_idx].item())
        if aa_idx >= int(protein_constants.AA.UNK):
            continue
        for i, j in heavy_bond_pairs(aa_idx):
            if not (mask14[res_idx, i] and mask14[res_idx, j]):
                continue
            pred_len = torch.linalg.norm(pred_pos14[res_idx, i] - pred_pos14[res_idx, j], ord=2)
            target_len = torch.linalg.norm(target_pos14[res_idx, i] - target_pos14[res_idx, j], ord=2)
            deviations.append(torch.abs(pred_len - target_len))
    return safe_mean(deviations, pred_pos14.device)


def bond_angle_deviation(pred_pos14: torch.Tensor, target_pos14: torch.Tensor, mask14: torch.Tensor, aa: torch.Tensor) -> torch.Tensor:
    deviations = []
    for res_idx in range(aa.size(0)):
        aa_idx = int(aa[res_idx].item())
        if aa_idx >= int(protein_constants.AA.UNK):
            continue
        for i, j, k in heavy_angle_triplets(aa_idx):
            if not (mask14[res_idx, i] and mask14[res_idx, j] and mask14[res_idx, k]):
                continue
            pred_angle = angle_value(pred_pos14[res_idx, [i, j, k]])
            target_angle = angle_value(target_pos14[res_idx, [i, j, k]])
            deviations.append(torch.rad2deg(torch.abs(pred_angle - target_angle)))
    return safe_mean(deviations, pred_pos14.device)


def clash_score(pred_pos14: torch.Tensor, mask14: torch.Tensor, aa: torch.Tensor, threshold: float = 1.6) -> torch.Tensor:
    atom_positions = []
    residue_ids = []
    atom_slots = []
    for res_idx in range(mask14.size(0)):
        valid_slots = torch.where(mask14[res_idx])[0]
        for atom_idx in valid_slots.tolist():
            atom_positions.append(pred_pos14[res_idx, atom_idx])
            residue_ids.append(res_idx)
            atom_slots.append(atom_idx)

    if len(atom_positions) < 2:
        return pred_pos14.new_tensor(0.0)

    pos = torch.stack(atom_positions, dim=0)
    residue_ids_tensor = torch.tensor(residue_ids, device=pos.device)
    atom_slots_tensor = torch.tensor(atom_slots, device=pos.device)

    distances = torch.cdist(pos, pos, p=2)
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)

    bonded = torch.zeros_like(distances, dtype=torch.bool)
    for i in range(len(atom_positions)):
        res_i = int(residue_ids_tensor[i].item())
        atom_i = int(atom_slots_tensor[i].item())
        aa_i = int(aa[res_i].item()) if res_i < aa.size(0) else int(protein_constants.AA.UNK)
        bond_pairs = set(heavy_bond_pairs(aa_i))
        for j in range(i + 1, len(atom_positions)):
            res_j = int(residue_ids_tensor[j].item())
            atom_j = int(atom_slots_tensor[j].item())
            if res_i == res_j and tuple(sorted((atom_i, atom_j))) in bond_pairs:
                bonded[i, j] = True
            if res_j == res_i + 1 and atom_i == int(protein_constants.BBHeavyAtom.C) and atom_j == int(protein_constants.BBHeavyAtom.N):
                bonded[i, j] = True
            if res_i == res_j + 1 and atom_j == int(protein_constants.BBHeavyAtom.C) and atom_i == int(protein_constants.BBHeavyAtom.N):
                bonded[i, j] = True

    clash_mask = upper & (~bonded)
    if clash_mask.sum() == 0:
        return pred_pos14.new_tensor(0.0)
    return (distances[clash_mask] < threshold).float().mean()


def torsion_metrics(pred_pos14: torch.Tensor, target_pos14: torch.Tensor, aa: torch.Tensor) -> dict[str, torch.Tensor]:
    device = pred_pos14.device
    if aa.numel() == 0 or (aa >= int(protein_constants.AA.UNK)).all():
        return {f"{name}_mae_deg": torch.tensor(float("nan"), device=device) for name in CHI_METRIC_NAMES} | {
            "rotamer_recovery": torch.tensor(float("nan"), device=device)
        }

    pred_torsion, pred_mask = get_torsion_angle(pred_pos14, aa)
    target_torsion, target_mask = get_torsion_angle(target_pos14, aa)
    metrics = {}
    rotamer_hits = []
    rotamer_total = []
    for chi_idx, chi_name in enumerate(CHI_METRIC_NAMES, start=1):
        mask = pred_mask[:, chi_idx] & target_mask[:, chi_idx]
        if mask.any():
            diff = torch.rad2deg(circular_abs_diff(pred_torsion[:, chi_idx], target_torsion[:, chi_idx]))
            metrics[f"{chi_name}_mae_deg"] = diff[mask].mean()
            rotamer_hits.append((diff[mask] <= 20.0).float().mean())
            rotamer_total.append(diff[mask])
        else:
            metrics[f"{chi_name}_mae_deg"] = torch.tensor(float("nan"), device=device)
    if rotamer_total:
        all_diff = torch.cat(rotamer_total, dim=0)
        metrics["rotamer_recovery"] = (all_diff <= 20.0).float().mean()
    else:
        metrics["rotamer_recovery"] = torch.tensor(float("nan"), device=device)
    return metrics


def compute_batch_eval_metrics(
    batch: dict[str, torch.Tensor],
    pred_global: torch.Tensor,
    indices: torch.Tensor,
    codebook_size: int,
    quantizer_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = pred_global.device
    autocast_context = (
        torch.autocast(device_type=device.type, enabled=False)
        if device.type in {"cuda", "cpu"}
        else contextlib.nullcontext()
    )
    with autocast_context:
        pred_global = pred_global.float()
        indices = indices.long()
        quantizer_mask = quantizer_mask.bool()
        token_mask = batch["token_mask"].bool()
        atom_class = batch["atom_class"].long()
        residue_index = batch["residue_index"].long()
        atom_slot = batch["atom_slot"].long()
        residue_type = batch["residue_type"].long()
        target_global = batch["coords_global"].float()

        all_atom_rmsd_values = []
        sidechain_rmsd_values = []
        chi_metrics_store = {f"{name}_mae_deg": [] for name in CHI_METRIC_NAMES}
        rotamer_values = []
        clash_values = []
        bond_length_values = []
        bond_angle_values = []
        bucket_values = {"short": [], "medium": [], "long": []}

        for batch_idx in range(pred_global.size(0)):
            valid = token_mask[batch_idx]
            if valid.sum() == 0:
                continue

            pred_valid = pred_global[batch_idx][valid]
            target_valid = target_global[batch_idx][valid]
            pred_aligned = kabsch_align(pred_valid, target_valid)
            all_atom_rmsd_value = rmsd(pred_aligned, target_valid)
            all_atom_rmsd_values.append(all_atom_rmsd_value)

            side_mask = atom_class[batch_idx][valid] == 2
            if side_mask.any():
                sidechain_rmsd_values.append(rmsd(pred_aligned[side_mask], target_valid[side_mask]))

            pos14_pred, mask14, aa = reconstruct_sample_pos14(
                residue_index=residue_index[batch_idx],
                atom_slot=atom_slot[batch_idx],
                residue_type=residue_type[batch_idx],
                coords=pred_global[batch_idx],
                mask=valid,
            )
            pos14_target, _, _ = reconstruct_sample_pos14(
                residue_index=residue_index[batch_idx],
                atom_slot=atom_slot[batch_idx],
                residue_type=residue_type[batch_idx],
                coords=target_global[batch_idx],
                mask=valid,
            )
            if pos14_pred.size(0) > 0 and (aa < int(protein_constants.AA.UNK)).any():
                torsion = torsion_metrics(pos14_pred, pos14_target, aa)
                for key in chi_metrics_store:
                    if not torch.isnan(torsion[key]):
                        chi_metrics_store[key].append(torsion[key])
                if not torch.isnan(torsion["rotamer_recovery"]):
                    rotamer_values.append(torsion["rotamer_recovery"])

                clash = clash_score(pos14_pred, mask14, aa)
                if not torch.isnan(clash):
                    clash_values.append(clash)

                bond_len = bond_length_deviation(pos14_pred, pos14_target, mask14, aa)
                if not torch.isnan(bond_len):
                    bond_length_values.append(bond_len)

                bond_ang = bond_angle_deviation(pos14_pred, pos14_target, mask14, aa)
                if not torch.isnan(bond_ang):
                    bond_angle_values.append(bond_ang)

            num_residues = int(residue_index[batch_idx][valid].max().item()) + 1
            bucket_values[sample_length_bucket(num_residues)].append(all_atom_rmsd_value)

        codebook = codebook_summary(indices, ~quantizer_mask, codebook_size)

        metrics = {
            "all_atom_rmsd": safe_mean(all_atom_rmsd_values, device),
            "sidechain_rmsd": safe_mean(sidechain_rmsd_values, device),
            "rotamer_recovery": safe_mean(rotamer_values, device),
            "clash_score": safe_mean(clash_values, device),
            "bond_length_dev": safe_mean(bond_length_values, device),
            "bond_angle_dev_deg": safe_mean(bond_angle_values, device),
            **codebook,
        }
        for key, values in chi_metrics_store.items():
            metrics[key] = safe_mean(values, device)
        for bucket_name, values in bucket_values.items():
            metrics[f"{bucket_name}_all_atom_rmsd"] = safe_mean(values, device)
        return metrics
