from __future__ import annotations

import math

import torch

from . import constants


def _get_torsion(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sign = torch.sign((torch.cross(v1, v2, dim=-1) * v0).sum(-1))
    return sign * torch.acos((n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999))


def get_chi_angles(restype: int, pos14: torch.Tensor) -> torch.Tensor:
    chi_angles = torch.full((4,), fill_value=float("inf"), device=pos14.device, dtype=pos14.dtype)
    for chi_idx, atom_names in enumerate(constants.chi_angles_atoms[restype]):
        atom_indices = [constants.restype_atom14_name_to_index[restype][name] for name in atom_names]
        points = torch.stack([pos14[atom_idx] for atom_idx in atom_indices], dim=0)
        chi_angles[chi_idx] = _get_torsion(*torch.unbind(points, dim=0))
    return chi_angles


def get_psi_angle(pos14: torch.Tensor) -> torch.Tensor:
    return _get_torsion(pos14[0], pos14[1], pos14[2], pos14[3]).reshape(1)


def get_torsion_angle(pos14: torch.Tensor, aa: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
    torsion_values = []
    torsion_masks = []

    for residue_idx in range(pos14.shape[0]):
        if aa[residue_idx] < constants.AA.UNK:
            chi = get_chi_angles(int(aa[residue_idx].item()), pos14[residue_idx])
            psi = get_psi_angle(pos14[residue_idx])
            torsion = torch.cat([psi, chi], dim=0)
            torsion_mask = torsion.isfinite()
        else:
            torsion = torch.zeros((5,), device=pos14.device, dtype=pos14.dtype)
            torsion_mask = torch.zeros((5,), device=pos14.device, dtype=torch.bool)

        torsion_values.append(torsion.nan_to_num(posinf=0.0))
        torsion_masks.append(torsion_mask)

    torsion_tensor = torch.stack(torsion_values, dim=0) % (2 * math.pi)
    torsion_mask_tensor = torch.stack(torsion_masks, dim=0).bool()
    return torsion_tensor, torsion_mask_tensor
