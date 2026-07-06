from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .pepsage_compat import AA, BBHeavyAtom, PepDataset, construct_3d_basis, global_to_local
from .pepsage_compat import constants as pep_constants

BB_CLASS = 0
C_REF_CLASS = 1
SC_CLASS = 2

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
BACKBONE_SLOT_BY_NAME = {
    "N": int(BBHeavyAtom.N),
    "CA": int(BBHeavyAtom.CA),
    "C": int(BBHeavyAtom.C),
    "O": int(BBHeavyAtom.O),
    "OXT": int(BBHeavyAtom.OXT),
}


def _canonical_atom_names_by_aa() -> dict[int, list[str]]:
    canonical = {}
    for aa in AA:
        if aa == AA.UNK:
            continue
        canonical[int(aa)] = [name for name in pep_constants.restype_to_heavyatom_names[aa] if name and name != "OXT"]
    return canonical


CANONICAL_ATOM_NAMES_BY_AA = _canonical_atom_names_by_aa()
AA_BY_CANONICAL_ATOM_NAMES = {
    tuple(atom_names): aa_idx for aa_idx, atom_names in CANONICAL_ATOM_NAMES_BY_AA.items()
}


def atom_slot_to_class(atom_slot: int) -> int:
    if atom_slot == int(BBHeavyAtom.CA):
        return 1
    if atom_slot in (int(BBHeavyAtom.N), int(BBHeavyAtom.C), int(BBHeavyAtom.O), int(BBHeavyAtom.OXT)):
        return 0
    return 2


def _truncate_token_dict(token_dict: dict[str, Any], max_tokens: int | None) -> dict[str, Any]:
    if max_tokens is None or token_dict["coords_local"].size(0) <= max_tokens:
        return token_dict
    for key, value in list(token_dict.items()):
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.size(0) >= max_tokens:
            token_dict[key] = value[:max_tokens]
    return token_dict


def _build_pepsage_tokens(sample: dict[str, Any]) -> dict[str, Any]:
    peptide_mask = sample["generate_mask"].bool()
    aa = sample["aa"][peptide_mask].long()
    pos_heavyatom = sample["pos_heavyatom"][peptide_mask].float()
    mask_heavyatom = sample["mask_heavyatom"][peptide_mask].bool()

    frames = construct_3d_basis(
        pos_heavyatom[None, :, BBHeavyAtom.CA],
        pos_heavyatom[None, :, BBHeavyAtom.C],
        pos_heavyatom[None, :, BBHeavyAtom.N],
    )[0]
    frame_trans = pos_heavyatom[:, BBHeavyAtom.CA]
    local_pos = global_to_local(frames[None], frame_trans[None], pos_heavyatom[None])[0]

    coords_local = []
    coords_global = []
    frame_rot_tokens = []
    frame_trans_tokens = []
    residue_type = []
    atom_slot = []
    atom_class = []
    residue_index = []

    for res_idx in range(aa.size(0)):
        valid_atom_idx = torch.where(mask_heavyatom[res_idx])[0]
        for atom_idx in valid_atom_idx.tolist():
            coords_local.append(local_pos[res_idx, atom_idx])
            coords_global.append(pos_heavyatom[res_idx, atom_idx])
            frame_rot_tokens.append(frames[res_idx])
            frame_trans_tokens.append(frame_trans[res_idx])
            residue_type.append(aa[res_idx])
            atom_slot.append(atom_idx)
            atom_class.append(atom_slot_to_class(atom_idx))
            residue_index.append(res_idx)

    return {
        "sample_id": sample["id"],
        "coords_local": torch.stack(coords_local, dim=0),
        "coords_global": torch.stack(coords_global, dim=0),
        "frame_rot": torch.stack(frame_rot_tokens, dim=0),
        "frame_trans": torch.stack(frame_trans_tokens, dim=0),
        "residue_type": torch.tensor(residue_type, dtype=torch.long),
        "atom_slot": torch.tensor(atom_slot, dtype=torch.long),
        "atom_class": torch.tensor(atom_class, dtype=torch.long),
        "residue_index": torch.tensor(residue_index, dtype=torch.long),
        "token_mask": torch.ones(len(coords_local), dtype=torch.bool),
    }


def _protein_token_class_to_local_class(token_class: int) -> int:
    if token_class == C_REF_CLASS:
        return 1
    if token_class == BB_CLASS:
        return 0
    if token_class == SC_CLASS:
        return 2
    return 3


def _normalize_atom_name(atom_name: Any) -> str:
    return str(atom_name).strip().upper()


def _label_to_aa_index(label: Any) -> int | None:
    if label is None:
        return None
    if isinstance(label, bool):
        return None
    if isinstance(label, int):
        return int(label) if 0 <= int(label) <= int(AA.UNK) else None

    label_str = str(label).strip()
    if not label_str:
        return None
    if label_str.isdigit():
        label_int = int(label_str)
        return label_int if 0 <= label_int <= int(AA.UNK) else None
    if "_" in label_str:
        tail = label_str.split("_")[-1]
        if tail:
            label_str = tail
    try:
        return int(AA(label_str.upper()))
    except ValueError:
        return None


def _expand_residue_annotation(annotation: Any, residue_ids: torch.Tensor) -> list[Any] | None:
    if annotation is None:
        return None
    if residue_ids.numel() == 0:
        return None
    if isinstance(annotation, str):
        unique_residues = torch.unique_consecutive(residue_ids)
        if len(annotation) == len(unique_residues):
            return list(annotation)
        if len(annotation) > 0 and int(residue_ids.max().item()) < len(annotation):
            return [annotation[int(residue_id.item())] for residue_id in residue_ids]
        return None

    values = list(annotation)
    if len(values) == residue_ids.numel():
        return values
    if len(values) > 0 and int(residue_ids.max().item()) < len(values):
        return [values[int(residue_id.item())] for residue_id in residue_ids]
    unique_residues = torch.unique_consecutive(residue_ids)
    if len(values) == len(unique_residues):
        expanded = []
        for residue_idx, residue_id in enumerate(unique_residues.tolist()):
            count = int((residue_ids == residue_id).sum().item())
            expanded.extend([values[residue_idx]] * count)
        return expanded
    return None


def _infer_residue_type_from_atoms(atom_names: list[str]) -> int:
    return AA_BY_CANONICAL_ATOM_NAMES.get(tuple(atom_names), int(AA.UNK))


def _infer_semantic_tokens(
    residue_ids: torch.Tensor,
    token_class: torch.Tensor,
    atom_names: list[str] | None,
    residue_annotation: list[Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    residue_type = torch.full_like(residue_ids, fill_value=int(AA.UNK))
    atom_slot = torch.full_like(residue_ids, fill_value=15)
    if residue_ids.numel() == 0:
        return residue_type, atom_slot

    unique_residue_ids = torch.unique_consecutive(residue_ids)
    semantic_residue_count = 0
    for residue_id in unique_residue_ids.tolist():
        residue_mask = residue_ids == residue_id
        residue_indices = torch.where(residue_mask)[0]
        residue_atom_names = (
            [_normalize_atom_name(atom_names[int(idx)]) for idx in residue_indices.tolist()]
            if atom_names is not None
            else None
        )

        residue_aa = None
        if residue_annotation is not None:
            residue_aa = _label_to_aa_index(residue_annotation[int(residue_indices[0].item())])
        if residue_aa is None and residue_atom_names is not None:
            residue_aa = _infer_residue_type_from_atoms(residue_atom_names)
        if residue_aa is None:
            residue_aa = int(AA.UNK)

        residue_type[residue_indices] = residue_aa
        if residue_aa < int(AA.UNK):
            semantic_residue_count += 1

        if residue_atom_names is not None and residue_aa < int(AA.UNK):
            name_to_slot = {
                name: slot_idx
                for slot_idx, name in enumerate(pep_constants.restype_to_heavyatom_names[AA(residue_aa)])
                if name
            }
            for offset, atom_name in zip(residue_indices.tolist(), residue_atom_names):
                atom_slot[offset] = name_to_slot.get(atom_name, 15)
            continue

        sidechain_slot = 4
        for offset in residue_indices.tolist():
            token_cls = int(token_class[offset].item())
            if token_cls == C_REF_CLASS:
                atom_slot[offset] = int(BBHeavyAtom.CA)
            elif token_cls == BB_CLASS:
                atom_name = _normalize_atom_name(atom_names[offset]) if atom_names is not None else ""
                atom_slot[offset] = BACKBONE_SLOT_BY_NAME.get(atom_name, min(sidechain_slot, 15))
            else:
                atom_slot[offset] = min(sidechain_slot, 15)
                sidechain_slot += 1

    if residue_type.numel() > 0 and semantic_residue_count == 0:
        LOGGER.warning(
            "Could not infer residue types for general_protein sample; residue-aware features will degrade to UNK tokens."
        )
    return residue_type, atom_slot


def _build_protein_tokens(sample: dict[str, Any], sample_id: str) -> dict[str, Any]:
    structure = sample["structure"].float()
    residue_ids = sample["residue_ids"].long()
    token_class = sample["token_class"].long()
    atom_names = sample.get("atom_names")
    residue_annotation = sample.get("residue_annotation")
    residue_type_tensor, atom_slot_tensor = _infer_semantic_tokens(
        residue_ids=residue_ids,
        token_class=token_class,
        atom_names=atom_names,
        residue_annotation=residue_annotation,
    )

    unique_residue_ids = torch.unique_consecutive(residue_ids)
    coords_local = []
    coords_global = []
    frame_rot_tokens = []
    frame_trans_tokens = []
    residue_type = []
    atom_slot = []
    atom_class = []
    residue_index = []

    residue_counter = 0
    for residue_id in unique_residue_ids.tolist():
        residue_mask = residue_ids == residue_id
        residue_coords = structure[residue_mask]
        residue_classes = token_class[residue_mask]

        bb_mask = (residue_classes == BB_CLASS) | (residue_classes == C_REF_CLASS)
        bb_coords = residue_coords[bb_mask]
        if bb_coords.size(0) < 3:
            continue

        n_coord = bb_coords[0]
        ca_coord = bb_coords[1]
        c_coord = bb_coords[2]

        frame = construct_3d_basis(
            ca_coord.view(1, 1, 3),
            c_coord.view(1, 1, 3),
            n_coord.view(1, 1, 3),
        )[0, 0]
        local_coords = global_to_local(
            frame.view(1, 1, 3, 3),
            ca_coord.view(1, 1, 3),
            residue_coords.view(1, 1, -1, 3),
        )[0, 0]

        for local_slot in range(residue_coords.size(0)):
            source_index = torch.where(residue_mask)[0][local_slot]
            coords_local.append(local_coords[local_slot])
            coords_global.append(residue_coords[local_slot])
            frame_rot_tokens.append(frame)
            frame_trans_tokens.append(ca_coord)
            residue_type.append(int(residue_type_tensor[source_index].item()))
            atom_slot.append(int(atom_slot_tensor[source_index].item()))
            atom_class.append(_protein_token_class_to_local_class(int(residue_classes[local_slot])))
            residue_index.append(residue_counter)

        residue_counter += 1

    return {
        "sample_id": sample_id,
        "coords_local": torch.stack(coords_local, dim=0),
        "coords_global": torch.stack(coords_global, dim=0),
        "frame_rot": torch.stack(frame_rot_tokens, dim=0),
        "frame_trans": torch.stack(frame_trans_tokens, dim=0),
        "residue_type": torch.tensor(residue_type, dtype=torch.long),
        "atom_slot": torch.tensor(atom_slot, dtype=torch.long),
        "atom_class": torch.tensor(atom_class, dtype=torch.long),
        "residue_index": torch.tensor(residue_index, dtype=torch.long),
        "token_mask": torch.ones(len(coords_local), dtype=torch.bool),
    }


class PeptideLocalTokenizerDataset(Dataset):
    def __init__(
        self,
        structure_dir: str,
        dataset_dir: str,
        name: str,
        reset: bool = False,
        max_tokens: int | None = None,
    ):
        super().__init__()
        self.base_dataset = PepDataset(
            structure_dir=structure_dir,
            dataset_dir=dataset_dir,
            name=name,
            transform=None,
            reset=reset,
        )
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base_dataset[index]
        token_dict = _build_pepsage_tokens(sample)
        return _truncate_token_dict(token_dict, self.max_tokens)


def _resolve_parquet_dir(split_cfg) -> str:
    parquet_dir = getattr(split_cfg, "parquet_dir", None)
    if parquet_dir:
        parquet_path = Path(str(parquet_dir))
        if not parquet_path.is_absolute():
            parquet_path = (WORKSPACE_ROOT / parquet_path).resolve()
        return str(parquet_path)

    data_root = getattr(split_cfg, "data_root", None)
    dataset_name = getattr(split_cfg, "dataset_name", None)
    split_name = getattr(split_cfg, "split", None)
    if data_root is None or dataset_name is None or split_name is None:
        raise ValueError("general_protein config must provide either parquet_dir or data_root + dataset_name + split")

    data_root_path = Path(str(data_root))
    if not data_root_path.is_absolute():
        data_root_path = (WORKSPACE_ROOT / data_root_path).resolve()

    if dataset_name == "cath":
        return str(data_root_path / "cath" / "cath_v4_3_0" / "cath_40" / "processed" / split_name)
    if dataset_name == "alphafolddb":
        return str(data_root_path / "alphafold_db" / "processed")
    if dataset_name == "casp":
        return str(data_root_path / "casp" / split_name)
    return str(data_root_path / dataset_name / split_name)


def _random_rotation_matrix() -> torch.Tensor:
    q = torch.randn(4, dtype=torch.float32)
    q = q / q.norm().clamp_min(1e-8)
    w, x, y, z = q
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


class ProteinParquetLocalTokenizerDataset(Dataset):
    def __init__(self, split_cfg, max_tokens: int | None = None):
        super().__init__()
        self.parquet_dir = _resolve_parquet_dir(split_cfg)
        self.max_tokens = max_tokens
        self.nan_handling = getattr(split_cfg, "nan_handling", "remove")
        self.randomly_rotate = bool(getattr(split_cfg, "randomly_rotate", True))
        self.max_data = getattr(split_cfg, "max_data", None)
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        parquet_files = sorted(file for file in os.listdir(self.parquet_dir) if file.endswith(".parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.parquet_dir}")

        first_path = os.path.join(self.parquet_dir, parquet_files[0])
        available_columns = set(pd.read_parquet(first_path).columns)
        columns = ["structure", "residue_ids", "token_class", "unknown_structure"]
        optional_columns = ["atom", "residue_name", "residue_type", "seq"]
        selected_columns = columns + [col for col in optional_columns if col in available_columns]
        blocks = []
        total = 0
        for parquet_name in parquet_files:
            parquet_path = os.path.join(self.parquet_dir, parquet_name)
            block = pd.read_parquet(parquet_path)[selected_columns]
            blocks.append(block)
            total += len(block)
            if self.max_data is not None and total >= self.max_data:
                break
        if not blocks:
            raise FileNotFoundError(f"No parquet files found in {self.parquet_dir}")
        data = pd.concat(blocks, ignore_index=True)
        if self.max_data is not None:
            data = data.iloc[: self.max_data].reset_index(drop=True)
        return data

    def _process_row(self, row: pd.Series) -> dict[str, torch.Tensor]:
        structure = torch.tensor(row["structure"], dtype=torch.float32).reshape(-1, 3)
        residue_ids = torch.tensor(row["residue_ids"], dtype=torch.long)
        token_class = torch.tensor(row["token_class"], dtype=torch.long)
        unknown_structure = torch.tensor(row["unknown_structure"], dtype=torch.bool)
        atom_names = list(row["atom"]) if "atom" in row and row["atom"] is not None else None
        residue_annotation_source = None
        for key in ("residue_type", "residue_name", "seq"):
            if key in row and row[key] is not None:
                residue_annotation_source = row[key]
                break

        if self.nan_handling == "remove":
            removed_ids = torch.unique(residue_ids[unknown_structure])
            if removed_ids.numel() > 0:
                keep_mask = ~torch.isin(residue_ids, removed_ids)
                structure = structure[keep_mask]
                residue_ids = residue_ids[keep_mask]
                token_class = token_class[keep_mask]
                unknown_structure = unknown_structure[keep_mask]
                if atom_names is not None:
                    keep_list = keep_mask.tolist()
                    atom_names = [name for name, keep in zip(atom_names, keep_list) if keep]
                if residue_annotation_source is not None and not isinstance(residue_annotation_source, str):
                    annotation_values = list(residue_annotation_source)
                    if len(annotation_values) == keep_mask.numel():
                        keep_list = keep_mask.tolist()
                        residue_annotation_source = [
                            value for value, keep in zip(annotation_values, keep_list) if keep
                        ]
        elif self.nan_handling == "zero":
            structure[unknown_structure] = 0.0
        else:
            raise ValueError(f"Unsupported nan_handling: {self.nan_handling}")

        if self.randomly_rotate and (~unknown_structure).any():
            rot = _random_rotation_matrix()
            structure[~unknown_structure] = torch.einsum("ij,nj->ni", rot, structure[~unknown_structure])

        residue_annotation = _expand_residue_annotation(residue_annotation_source, residue_ids)

        return {
            "structure": structure,
            "residue_ids": residue_ids,
            "token_class": token_class,
            "atom_names": atom_names,
            "residue_annotation": residue_annotation,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._process_row(self.data.iloc[index])
        token_dict = _build_protein_tokens(sample, sample_id=f"general_protein_{index}")
        return _truncate_token_dict(token_dict, self.max_tokens)
