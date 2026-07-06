import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenStructureDecoder(nn.Module):
    """Dual-path decoder: trainable MLP + frozen tokenizer decoder."""

    def __init__(self, frozen_quantizer, frozen_decoder, node_dim=128,
                 latent_dim=128, max_num_heavyatoms=15, hidden_dim=256):
        super().__init__()
        self.max_num_heavyatoms = max_num_heavyatoms

        # frozen components from exp1 tokenizer
        self.frozen_project_out = frozen_quantizer.project_out  # Linear(6→128)
        for p in self.frozen_project_out.parameters():
            p.requires_grad_(False)

        self.frozen_decoder = frozen_decoder
        for p in self.frozen_decoder.parameters():
            p.requires_grad_(False)

        # primary path: trainable MLP head
        self.structure_head = nn.Sequential(
            nn.Linear(node_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_num_heavyatoms * 3),
        )

    def _codes_to_normalized(self, code_indices):
        """Convert integer codes {0,1,2,3} to FSQ normalized {-1,-0.5,0,0.5}."""
        return (code_indices.float() - 2.0) / 2.0

    def decode_primary(self, code_indices, node_embed, pred_rotmats, pred_trans):
        """Primary path: trainable MLP decode.

        code_indices: (B, L, 6) integer codes in {0,1,2,3}
        node_embed:   (B, L, node_dim)
        pred_rotmats: (B, L, 3, 3)
        pred_trans:   (B, L, 3)

        Returns: (pred_global, pred_local) both (B, L, 15, 3)
        """
        codes_norm = self._codes_to_normalized(code_indices)

        with torch.no_grad():
            code_vector = self.frozen_project_out(codes_norm)  # (B, L, 128)

        decoder_input = torch.cat([node_embed, code_vector], dim=-1)
        pred_local = self.structure_head(decoder_input)
        pred_local = pred_local.reshape(*code_indices.shape[:2],
                                        self.max_num_heavyatoms, 3)

        pred_global = torch.einsum("bnij,bnaj->bnai", pred_rotmats, pred_local) \
                    + pred_trans.unsqueeze(2)

        return pred_global, pred_local

    def decode_auxiliary(self, code_indices, pred_rotmats, pred_trans,
                         residue_index, pad_mask):
        """Auxiliary path: frozen tokenizer Mamba decoder.

        Runs frozen decoder on code vectors to provide consistency signal.
        Uses residue-level codes expanded per atom.

        code_indices: (B, L, 6)
        pred_rotmats: (B, L, 3, 3)
        pred_trans:   (B, L, 3)
        residue_index: (B, T) atom-to-residue mapping
        pad_mask:      (B, T) True=padded

        Returns: (aux_pred_global, aux_pred_local) both (B, L, 15, 3)
        """
        codes_norm = self._codes_to_normalized(code_indices)
        B, L = code_indices.shape[:2]

        with torch.no_grad():
            code_vector = self.frozen_project_out(codes_norm)  # (B, L, 128)

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
                                    device=code_indices.device)
            atom_counts = torch.zeros(B, L, self.max_num_heavyatoms,
                                      device=code_indices.device)

            for b in range(B):
                per_res_atom_idx = torch.zeros(L, dtype=torch.long,
                                               device=code_indices.device)
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
