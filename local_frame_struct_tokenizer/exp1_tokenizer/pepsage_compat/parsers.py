from __future__ import annotations

from typing import Any

import torch
from Bio import PDB
from Bio.PDB import Selection
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue

from .constants import AA, BBHeavyAtom, max_num_heavyatoms, non_standard_residue_substitutions, restype_to_heavyatom_names


def _get_residue_heavyatom_info(residue: Residue) -> tuple[torch.Tensor, torch.Tensor]:
    pos_heavyatom = torch.zeros((max_num_heavyatoms, 3), dtype=torch.float32)
    mask_heavyatom = torch.zeros((max_num_heavyatoms,), dtype=torch.bool)
    restype = AA(residue.get_resname())

    for atom_idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
        if atom_name and atom_name in residue:
            pos_heavyatom[atom_idx] = torch.tensor(residue[atom_name].get_coord().tolist(), dtype=torch.float32)
            mask_heavyatom[atom_idx] = True

    return pos_heavyatom, mask_heavyatom


def parse_pdb(path: str, model_id: int = 0, unknown_threshold: float = 1.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(None, path)
    return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)


def parse_mmcif_assembly(path: str, model_id: int, assembly_id: int = 0, unknown_threshold: float = 1.0):
    parser = MMCIFParser()
    structure = parser.get_structure(None, path)
    mmcif_dict = parser._mmcif_dict
    if "_pdbx_struct_assembly_gen.asym_id_list" not in mmcif_dict:
        return parse_biopython_structure(structure[model_id], unknown_threshold=unknown_threshold)

    assemblies = [tuple(chains.split(",")) for chains in mmcif_dict["_pdbx_struct_assembly_gen.asym_id_list"]]
    label_to_auth = {}
    for label_asym_id, auth_asym_id in zip(mmcif_dict["_atom_site.label_asym_id"], mmcif_dict["_atom_site.auth_asym_id"]):
        label_to_auth[label_asym_id] = auth_asym_id
    model_real = list({structure[model_id][label_to_auth[ch]] for ch in assemblies[assembly_id]})
    return parse_biopython_structure(model_real, unknown_threshold=unknown_threshold)


def parse_biopython_structure(entity: Any, unknown_threshold: float = 1.0):
    chains = Selection.unfold_entities(entity, "C")
    chains.sort(key=lambda chain: chain.get_id())

    data: dict[str, list[Any]] = {
        "chain_id": [],
        "chain_nb": [],
        "resseq": [],
        "icode": [],
        "res_nb": [],
        "aa": [],
        "pos_heavyatom": [],
        "mask_heavyatom": [],
    }
    tensor_types = {
        "chain_nb": torch.LongTensor,
        "resseq": torch.LongTensor,
        "res_nb": torch.LongTensor,
        "aa": torch.LongTensor,
        "pos_heavyatom": torch.stack,
        "mask_heavyatom": torch.stack,
    }

    count_aa = 0
    count_unk = 0

    for chain_idx, chain in enumerate(chains):
        seq_this = 0
        residues = Selection.unfold_entities(chain, "R")
        residues.sort(key=lambda residue: (residue.get_id()[1], residue.get_id()[2]))
        for residue in residues:
            resname = residue.get_resname()
            if not AA.is_aa(resname):
                continue
            if not (residue.has_id("CA") and residue.has_id("C") and residue.has_id("N")):
                continue

            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK:
                count_unk += 1
                continue

            data["chain_id"].append(chain.get_id())
            data["chain_nb"].append(chain_idx)
            data["aa"].append(restype)

            pos_heavyatom, mask_heavyatom = _get_residue_heavyatom_info(residue)
            data["pos_heavyatom"].append(pos_heavyatom)
            data["mask_heavyatom"].append(mask_heavyatom)

            resseq_this = int(residue.get_id()[1])
            icode_this = residue.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                ca_distance = torch.linalg.norm(
                    data["pos_heavyatom"][-2][BBHeavyAtom.CA] - data["pos_heavyatom"][-1][BBHeavyAtom.CA],
                    ord=2,
                ).item()
                if ca_distance <= 4.0:
                    seq_this += 1
                else:
                    seq_this += max(2, resseq_this - data["resseq"][-1])

            data["resseq"].append(resseq_this)
            data["icode"].append(icode_this)
            data["res_nb"].append(seq_this)

    if not data["aa"]:
        return None, None
    if count_aa > 0 and (count_unk / count_aa) >= unknown_threshold:
        return None, None

    seq_map = {}
    for idx, (chain_id, resseq, icode) in enumerate(zip(data["chain_id"], data["resseq"], data["icode"])):
        seq_map[(chain_id, resseq, icode)] = idx

    for key, convert_fn in tensor_types.items():
        data[key] = convert_fn(data[key])

    return data, seq_map


def get_fasta_from_pdb(pdb_file: str) -> dict[str, str]:
    parser = PDBParser(QUIET=True)
    sequence_by_chain: dict[str, str] = {}
    structure = parser.get_structure("structure_name", pdb_file)

    for model in structure:
        for chain in model:
            sequence = []
            for residue in chain:
                resname = residue.get_resname()
                if not AA.is_aa(resname):
                    continue
                if resname == "UNK":
                    sequence.append("X")
                else:
                    sequence.append(PDB.Polypeptide.three_to_one(non_standard_residue_substitutions[resname]))
            sequence_by_chain[chain.id] = "".join(sequence)

    return sequence_by_chain
