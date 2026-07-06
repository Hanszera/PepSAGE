import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.torsion import get_torsion_angle, full_atom_reconstruction


class TokenStructureDecoder(nn.Module):
    """Dual-path decoder: trainable MLP + frozen tokenizer decoder."""

    def __init__(self, frozen_quantizer, frozen_decoder, node_dim=128,
                 latent_dim=128, max_num_heavyatoms=15, hidden_dim=256):
        super().__init__()
        self.max_num_heavyatoms = max_num_heavyatoms

        self.frozen_quantizer = frozen_quantizer
        for p in self.frozen_quantizer.parameters():
            p.requires_grad_(False)

        self.frozen_project_out = frozen_quantizer.project_out
        for p in self.frozen_project_out.parameters():
            p.requires_grad_(False)

        self.frozen_decoder = frozen_decoder
        for p in self.frozen_decoder.parameters():
            p.requires_grad_(False)

        self.structure_head = nn.Sequential(
            nn.Linear(node_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_num_heavyatoms * 3),
        )

    def _discretize_z(self, z_pred):
        """Convert continuous R^6 prediction to FSQ normalized codes.

        z_pred → bound() → round() → / half_width → normalized codes
        """
        z_bounded = self.frozen_quantizer.bound(z_pred)
        codes_discrete = z_bounded.round()
        half_width = self.frozen_quantizer._levels // 2
        return codes_discrete / half_width.float()

    def decode_primary(self, z_pred, node_embed, pred_rotmats, pred_trans):
        """Primary path: trainable MLP decode.

        z_pred:       (B, L, 6) continuous R^6 prediction
        node_embed:   (B, L, node_dim)
        pred_rotmats: (B, L, 3, 3)
        pred_trans:   (B, L, 3)

        Returns: (pred_global, pred_local, codes_norm)
        """
        codes_norm = self._discretize_z(z_pred)

        with torch.no_grad():
            code_vector = self.frozen_project_out(codes_norm)

        decoder_input = torch.cat([node_embed, code_vector], dim=-1)
        pred_local = self.structure_head(decoder_input)
        pred_local = pred_local.reshape(*z_pred.shape[:2],
                                        self.max_num_heavyatoms, 3)

        pred_global = torch.einsum("bnij,bnaj->bnai", pred_rotmats, pred_local) \
                    + pred_trans.unsqueeze(2)

        return pred_global, pred_local, codes_norm

    def decode_auxiliary(self, z_pred, pred_rotmats, pred_trans,
                         residue_index, pad_mask):
        """Auxiliary path: frozen tokenizer Mamba decoder.

        z_pred:        (B, L, 6) continuous R^6 prediction
        pred_rotmats:  (B, L, 3, 3)
        pred_trans:    (B, L, 3)
        residue_index: (B, T) atom-to-residue mapping
        pad_mask:      (B, T) True=padded

        Returns: (aux_pred_global, aux_pred_local) both (B, L, 15, 3)
        """
        codes_norm = self._discretize_z(z_pred)
        B, L = z_pred.shape[:2]

        with torch.no_grad():
            code_vector = self.frozen_project_out(codes_norm)

            T = residue_index.shape[1]
            atom_tokens = torch.zeros(B, T, code_vector.shape[-1],
                                      device=code_vector.device)
            for b in range(B):
                for t_idx in range(T):
                    if not pad_mask[b, t_idx]:
                        rid = residue_index[b, t_idx]
                        if rid < L:
                            atom_tokens[b, t_idx] = code_vector[b, rid]

            decoded_coords = self.frozen_decoder(atom_tokens, pad_mask)

            aux_local = torch.zeros(B, L, self.max_num_heavyatoms, 3,
                                    device=z_pred.device)
            atom_counts = torch.zeros(B, L, self.max_num_heavyatoms,
                                      device=z_pred.device)

            for b in range(B):
                per_res_atom_idx = torch.zeros(L, dtype=torch.long,
                                               device=z_pred.device)
                for t_idx in range(T):
                    if not pad_mask[b, t_idx]:
                        rid = residue_index[b, t_idx].item()
                        if rid < L:
                            a_idx = per_res_atom_idx[rid].item()
                            if a_idx < self.max_num_heavyatoms:
                                aux_local[b, rid, a_idx] = decoded_coords[b, t_idx]
                                atom_counts[b, rid, a_idx] = 1.0
                                per_res_atom_idx[rid] += 1

        aux_global = torch.einsum("bnij,bnaj->bnai", pred_rotmats, aux_local) \
                   + pred_trans.unsqueeze(2)

        return aux_global, aux_local


def refine_with_ideal_geometry(pos_heavyatom, rotmats, trans, aa, generate_mask):
    """Replace MLP-predicted atoms with ideal-geometry reconstruction.

    Extracts torsion angles from MLP-predicted coordinates, then reconstructs
    atoms using rigid group geometry with crystallographic ideal bond lengths.
    Only applied to generated residues (generate_mask=True).
    """
    B, L = aa.shape
    pos14_mlp = pos_heavyatom[:, :, :14, :]

    aa_safe = aa.clone()
    aa_safe[aa >= 20] = 0

    all_torsions = []
    for b in range(B):
        torsion_b, _ = get_torsion_angle(pos14_mlp[b], aa_safe[b])
        all_torsions.append(torsion_b.to(pos_heavyatom.device))
    angles = torch.stack(all_torsions)

    pos14_ideal, _, _ = full_atom_reconstruction(rotmats, trans, angles, aa_safe)

    refined = torch.cat([pos14_ideal, pos_heavyatom[:, :, 14:15, :]], dim=2)
    mask = generate_mask.bool()
    return torch.where(mask[:, :, None, None], refined, pos_heavyatom)
