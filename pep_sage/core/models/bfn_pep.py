import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from core.models.edge import EdgeEmbedder
from core.models.node import NodeEmbedder
from core.models.ga import GAEncoder
from core.modules.protein.constants import BBHeavyAtom, max_num_heavyatoms
from core.modules.common.geometry import construct_3d_basis
from core.modules.so3.dist import uniform_so3
from core.dataset import all_atom, so3_utils

from core.models.bfn_base import BFNBase
from core.models.tokenizer_bridge import TokenizerBridge
from core.models.token_decoder import TokenStructureDecoder, refine_with_ideal_geometry

D_TOKEN = 6


class BFNModel(BFNBase):
    def __init__(self, cfg):
        super().__init__()
        self._model_cfg = cfg.model.encoder
        self._interpolant_cfg = cfg.model.interpolant

        self.node_embedder = NodeEmbedder(cfg.model.encoder.node_embed_size, max_num_heavyatoms)
        self.edge_embedder = EdgeEmbedder(cfg.model.encoder.edge_embed_size, max_num_heavyatoms)
        self.ga_encoder = GAEncoder(cfg.model.encoder.ipa)

        self.tokenizer_bridge = TokenizerBridge(
            getattr(cfg.model, "tokenizer_bridge", None),
            max_num_heavyatoms=max_num_heavyatoms,
        )

        self.token_decoder = TokenStructureDecoder(
            frozen_quantizer=self.tokenizer_bridge.tokenizer.quantizer,
            frozen_decoder=self.tokenizer_bridge.tokenizer.decoder,
            node_dim=cfg.model.encoder.ipa.c_s,
            latent_dim=self.tokenizer_bridge.latent_dim,
            max_num_heavyatoms=max_num_heavyatoms,
        )

        self.sample_trans = self._interpolant_cfg.sample_trans
        self.sample_rots = self._interpolant_cfg.sample_rots
        self.sample_tokens = getattr(self._interpolant_cfg, 'sample_tokens', True)

        self.sigma1_coord = cfg.sigma1_coord
        self.lambda1_rot = cfg.lambda1_rot
        self.sigma1_z = getattr(cfg, 'sigma1_z', 0.1)
        self.use_discrete_t = cfg.use_discrete_t
        self.discrete_steps = cfg.discrete_steps
        self.t_min = cfg.t_min
        self.destination_prediction = cfg.destination_prediction
        self.sampling_strategy = cfg.sampling_strategy

    def encode(self, batch):
        rotmats_1 = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA],
            ~batch['generate_mask']
        )
        structure_mask = context_mask if (self.sample_trans or self.sample_rots) else None
        sequence_mask = context_mask

        node_embed = self.node_embedder(
            batch['aa'], batch['res_nb'], batch['chain_nb'],
            batch['pos_heavyatom'], batch['mask_heavyatom'],
            structure_mask=structure_mask, sequence_mask=sequence_mask,
        )
        edge_embed = self.edge_embedder(
            batch['aa'], batch['res_nb'], batch['chain_nb'],
            batch['pos_heavyatom'], batch['mask_heavyatom'],
            structure_mask=structure_mask, sequence_mask=sequence_mask,
        )

        return rotmats_1, trans_1, node_embed, edge_embed

    def zero_center_part(self, pos, gen_mask, res_mask):
        center = torch.sum(pos * gen_mask[..., None], dim=1) / (
            torch.sum(gen_mask, dim=-1, keepdim=True) + 1e-8
        )
        center = center.unsqueeze(1)
        pos = pos - center
        pos = pos * res_mask[..., None]
        return pos, center

    def forward(self, batch, t=None):
        num_batch, num_res = batch['aa'].shape
        gen_mask = batch['generate_mask'].long()
        res_mask = batch['res_mask'].long()

        rotmats_1, trans_1, node_embed, edge_embed = self.encode(batch)
        trans_1_c = trans_1

        with torch.no_grad():
            token_targets = self.tokenizer_bridge.prepare_continuous_targets(batch)

        if token_targets is None:
            device = batch['aa'].device
            return {
                "trans_loss": torch.tensor(0.0, device=device),
                "rot_loss": torch.tensor(0.0, device=device),
                "token_bfn_loss": torch.tensor(0.0, device=device),
                "aa_aux_loss": torch.tensor(0.0, device=device),
                "structure_decode_loss": torch.tensor(0.0, device=device),
                "bb_atom_loss": torch.tensor(0.0, device=device),
            }

        z_true = token_targets["z_targets"]
        target_valid = token_targets["valid_mask"]
        gt_local_atoms = token_targets["gt_local_atoms"]

        with torch.no_grad():
            if t is None:
                t = torch.rand((num_batch, 1), device=batch['aa'].device)

            if self.sample_trans:
                trans_mu, trans_gamma = self.trans_bayesian_update(
                    t, sigma1=self.sigma1_coord, x=trans_1_c
                )
                trans_t_c = torch.where(
                    batch['generate_mask'][..., None], trans_mu, trans_1_c
                )
            else:
                trans_t_c = trans_1_c.detach().clone()

            if self.sample_rots:
                matrix_fisher = torch.stack([
                    self.sample_matrix_fisher_mixed(
                        self.get_lambdat(t_value, self.lambda1_rot) ** 2,
                        n_samples=num_res, device=batch['aa'].device
                    ) for t_value in t.reshape(-1)
                ], dim=0)
                rotmats_mu = torch.matmul(rotmats_1, matrix_fisher)
                rotmats_t = torch.where(
                    batch['generate_mask'][..., None, None], rotmats_mu, rotmats_1
                )
            else:
                rotmats_t = rotmats_1.detach().clone()

            if self.sample_tokens:
                z_mu, z_gamma = self.trans_bayesian_update(
                    t, sigma1=self.sigma1_z, x=z_true
                )
                z_t = torch.where(
                    batch['generate_mask'][..., None].bool(), z_mu, z_true
                )
            else:
                z_t = z_true.detach().clone()

        pred_rotmats_1, pred_trans_1, pred_z, pred_aa_logits, final_node_embed = \
            self.ga_encoder(
                t, rotmats_t, trans_t_c, z_t,
                node_embed, edge_embed, gen_mask, res_mask,
                return_node_embed=True,
            )

        if not self.use_discrete_t:
            raise NotImplementedError("Continuous time not implemented")

        i = (t * self.discrete_steps).int() + 1
        device = batch['aa'].device

        if self.sample_trans:
            trans_loss = self.dtime4continuous_loss(
                i=i, N=self.discrete_steps, sigma1=self.sigma1_coord,
                x_pred=pred_trans_1, x=trans_1_c, mask=batch['generate_mask'],
            )
            gt_bb_atoms = all_atom.to_atom37(trans_1_c, rotmats_1)[:, :, :3]
            pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
            bb_atom_loss = torch.sum(
                (gt_bb_atoms - pred_bb_atoms) ** 2 * gen_mask[..., None, None],
                dim=(-1, -2, -3)
            ) / (torch.sum(gen_mask, dim=-1) + 1e-8)
            bb_atom_loss = torch.mean(bb_atom_loss)
        else:
            trans_loss = torch.tensor(0.0, device=device)
            bb_atom_loss = torch.tensor(0.0, device=device)

        if self.sample_rots:
            rot_loss = self.dtime4so3_loss(
                i=i, N=self.discrete_steps, lambda1=self.lambda1_rot,
                x_pred=pred_rotmats_1, x=rotmats_1, mask=batch['generate_mask'],
            )
        else:
            rot_loss = torch.tensor(0.0, device=device)

        token_mask = gen_mask * target_valid.long()
        if self.sample_tokens and token_mask.sum() > 0:
            token_bfn_loss = self.dtime4continuous_loss(
                i=i, N=self.discrete_steps, sigma1=self.sigma1_z,
                x_pred=pred_z, x=z_true, mask=token_mask,
            )
        else:
            token_bfn_loss = torch.tensor(0.0, device=device)

        aa_gt = batch['aa']
        aa_loss_mask = gen_mask.bool() & target_valid
        if aa_loss_mask.any():
            aa_loss = F.cross_entropy(
                pred_aa_logits[aa_loss_mask], aa_gt[aa_loss_mask]
            )
        else:
            aa_loss = torch.tensor(0.0, device=device)

        atom_mask = batch['mask_heavyatom'].bool() & gen_mask.bool().unsqueeze(-1)
        if atom_mask.any():
            _, pred_local, _ = self.token_decoder.decode_primary(
                pred_z, final_node_embed, pred_rotmats_1, pred_trans_1,
            )
            struct_loss = ((pred_local - gt_local_atoms) ** 2).sum(-1)
            struct_loss = struct_loss[atom_mask].mean()
        else:
            struct_loss = torch.tensor(0.0, device=device)

        return {
            "trans_loss": trans_loss,
            "rot_loss": rot_loss,
            "token_bfn_loss": token_bfn_loss,
            "aa_aux_loss": aa_loss,
            "structure_decode_loss": struct_loss,
            "bb_atom_loss": bb_atom_loss,
        }

    @torch.no_grad()
    def sample(self, batch, num_steps, pos_norm):
        num_batch, num_res = batch['aa'].shape
        gen_mask = batch['generate_mask']
        res_mask = batch['res_mask']
        device = batch['aa'].device

        rotmats_1, trans_1, node_embed, edge_embed = self.encode(batch)

        if self.sample_trans:
            trans_0 = torch.zeros((num_batch, num_res, 3), device=device)
            trans_0, _ = self.zero_center_part(trans_0, gen_mask, res_mask)
            trans_0 = torch.where(gen_mask[..., None].bool(), trans_0, trans_1)
        else:
            trans_0 = trans_1.detach().clone()

        if self.sample_rots:
            rotmats_0 = uniform_so3(num_batch, num_res, device=device)
            rotmats_0 = torch.where(gen_mask[..., None, None].bool(), rotmats_0, rotmats_1)
        else:
            rotmats_0 = rotmats_1.detach().clone()

        z_0 = torch.zeros((num_batch, num_res, D_TOKEN), device=device)
        token_targets = self.tokenizer_bridge.prepare_continuous_targets(batch)
        if token_targets is not None:
            ctx_mask = ~gen_mask.bool() & token_targets["valid_mask"]
            z_0[ctx_mask] = token_targets["z_targets"][ctx_mask]

        rotmats_t, trans_t, z_t = rotmats_0, trans_0, z_0
        clean_traj = []

        for step_i in trange(1, num_steps + 1):
            t = torch.ones((num_batch, 1), device=device) * (step_i - 1) / num_steps
            if not self.use_discrete_t and not self.destination_prediction:
                t = torch.clamp(t, min=self.t_min)

            pred_R1, pred_t1, pred_z, pred_aa_logits = self.ga_encoder(
                t, rotmats_t, trans_t, z_t,
                node_embed, edge_embed, gen_mask.long(), res_mask.long(),
            )

            t_next = torch.ones((num_batch, 1), device=device) * step_i / num_steps

            if self.sample_trans:
                trans_t, _ = self.trans_bayesian_update(
                    t_next, sigma1=self.sigma1_coord, x=pred_t1
                )
                trans_t = torch.where(gen_mask[..., None].bool(), trans_t, trans_1)
            else:
                trans_t = trans_1.detach().clone()

            if self.sample_rots:
                matrix_fisher = torch.stack([
                    self.sample_matrix_fisher_mixed(
                        self.get_lambdat(tv, self.lambda1_rot) ** 2,
                        n_samples=num_res, device=device
                    ) for tv in t_next.reshape(-1)
                ], dim=0)
                rotmats_t = torch.matmul(pred_R1, matrix_fisher)
                rotmats_t = torch.where(
                    gen_mask[..., None, None].bool(), rotmats_t, rotmats_1
                )
            else:
                rotmats_t = rotmats_1.detach().clone()

            if self.sample_tokens:
                z_t, _ = self.trans_bayesian_update(
                    t_next, sigma1=self.sigma1_z, x=pred_z
                )
                z_t = torch.where(gen_mask[..., None].bool(), z_t, z_0)
            else:
                z_t = z_0.detach().clone()

            clean_traj.append({
                "trans": pred_t1.detach().clone().cpu() * pos_norm,
                "rotmats": pred_R1.detach().clone().cpu(),
                "z_pred": pred_z.detach().clone().cpu(),
                "seqs": F.softmax(pred_aa_logits, dim=-1).detach().clone().cpu(),
                "t": t.detach().clone().cpu(),
            })

        # final t=1 forward pass
        t = torch.ones((num_batch, 1), device=device)
        pred_R1, pred_t1, pred_z, pred_aa_logits, final_node_embed = \
            self.ga_encoder(
                t, rotmats_t, trans_t, z_t,
                node_embed, edge_embed, gen_mask.long(), res_mask.long(),
                return_node_embed=True,
            )

        trans_final = torch.where(gen_mask[..., None].bool(), pred_t1, trans_1) \
            if self.sample_trans else trans_1.detach().clone()
        rotmats_final = torch.where(gen_mask[..., None, None].bool(), pred_R1, rotmats_1) \
            if self.sample_rots else rotmats_1.detach().clone()

        pred_aa = F.softmax(pred_aa_logits, dim=-1)

        pos_heavyatom, pred_local, codes_norm = self.token_decoder.decode_primary(
            pred_z, final_node_embed,
            rotmats_final, trans_final * pos_norm,
        )

        aa_pred = pred_aa_logits.argmax(-1)
        pos_heavyatom = refine_with_ideal_geometry(
            pos_heavyatom, rotmats_final, trans_final * pos_norm, aa_pred, gen_mask,
        )

        batch['pos_heavyatom'] = batch['pos_heavyatom'] * pos_norm
        final_sample = {
            "trans": trans_final.detach().clone().cpu() * pos_norm,
            "rotmats": rotmats_final.detach().clone().cpu(),
            "seqs": pred_aa.detach().clone().cpu(),
            "z_pred": pred_z.detach().clone().cpu(),
            "pos_heavyatom": pos_heavyatom.detach().clone().cpu(),
            "t": t.detach().clone().cpu(),
            "batch": batch,
        }
        clean_traj.append(final_sample)
        return clean_traj

    @torch.no_grad()
    def fix_seq_sample(self, batch, num_steps, pos_norm):
        num_batch, num_res = batch['aa'].shape
        gen_mask = batch['generate_mask']
        res_mask = batch['res_mask']
        device = batch['aa'].device

        rotmats_1, trans_1, node_embed, edge_embed = self.encode(batch)

        if self.sample_trans:
            trans_0 = torch.zeros((num_batch, num_res, 3), device=device)
            trans_0, _ = self.zero_center_part(trans_0, gen_mask, res_mask)
            trans_0 = torch.where(gen_mask[..., None].bool(), trans_0, trans_1)
        else:
            trans_0 = trans_1.detach().clone()

        if self.sample_rots:
            rotmats_0 = uniform_so3(num_batch, num_res, device=device)
            rotmats_0 = torch.where(gen_mask[..., None, None].bool(), rotmats_0, rotmats_1)
        else:
            rotmats_0 = rotmats_1.detach().clone()

        z_0 = torch.zeros((num_batch, num_res, D_TOKEN), device=device)
        token_targets = self.tokenizer_bridge.prepare_continuous_targets(batch)
        if token_targets is not None:
            ctx_mask = ~gen_mask.bool() & token_targets["valid_mask"]
            z_0[ctx_mask] = token_targets["z_targets"][ctx_mask]

        rotmats_t, trans_t, z_t = rotmats_0, trans_0, z_0

        for step_i in range(1, num_steps + 1):
            t = torch.ones((num_batch, 1), device=device) * (step_i - 1) / num_steps
            if not self.use_discrete_t and not self.destination_prediction:
                t = torch.clamp(t, min=self.t_min)

            pred_R1, pred_t1, pred_z, pred_aa_logits = self.ga_encoder(
                t, rotmats_t, trans_t, z_t,
                node_embed, edge_embed, gen_mask.long(), res_mask.long(),
            )

            t_next = torch.ones((num_batch, 1), device=device) * step_i / num_steps

            if self.sample_trans:
                trans_t, _ = self.trans_bayesian_update(
                    t_next, sigma1=self.sigma1_coord, x=pred_t1
                )
                trans_t = torch.where(gen_mask[..., None].bool(), trans_t, trans_1)

            if self.sample_rots:
                matrix_fisher = torch.stack([
                    self.sample_matrix_fisher_mixed(
                        self.get_lambdat(tv, self.lambda1_rot) ** 2,
                        n_samples=num_res, device=device
                    ) for tv in t_next.reshape(-1)
                ], dim=0)
                rotmats_t = torch.matmul(pred_R1, matrix_fisher)
                rotmats_t = torch.where(
                    gen_mask[..., None, None].bool(), rotmats_t, rotmats_1
                )

            if self.sample_tokens:
                z_t, _ = self.trans_bayesian_update(
                    t_next, sigma1=self.sigma1_z, x=pred_z
                )
                z_t = torch.where(gen_mask[..., None].bool(), z_t, z_0)

        # final t=1 forward pass
        t = torch.ones((num_batch, 1), device=device)
        pred_R1, pred_t1, pred_z, pred_aa_logits, final_node_embed = \
            self.ga_encoder(
                t, rotmats_t, trans_t, z_t,
                node_embed, edge_embed, gen_mask.long(), res_mask.long(),
                return_node_embed=True,
            )
        pred_aa = F.softmax(pred_aa_logits, dim=-1)

        trans_final = torch.where(gen_mask[..., None].bool(), pred_t1, trans_1) \
            if self.sample_trans else trans_1.detach().clone()
        rotmats_final = torch.where(gen_mask[..., None, None].bool(), pred_R1, rotmats_1) \
            if self.sample_rots else rotmats_1.detach().clone()

        # compute validation errors
        trans_error = (
            ((trans_final - trans_1) ** 2).sum(dim=-1) * gen_mask
        ).sum() / gen_mask.sum()

        R_rel = torch.matmul(rotmats_1.transpose(-2, -1), rotmats_final)
        rotvec = so3_utils.rotmat_to_rotvec(R_rel)
        dist = torch.norm(rotvec, dim=-1)
        rots_error = (dist * gen_mask).sum(-1) / gen_mask.sum(-1)

        aar = (
            (pred_aa.argmax(-1) == batch['aa']) * gen_mask
        ).sum() / gen_mask.sum()

        # per-dimension token accuracy via discretization
        token_errors = {}
        if token_targets is not None:
            z_gt = token_targets["z_targets"]
            valid = gen_mask.bool() & token_targets["valid_mask"]
            if valid.any():
                gt_codes_norm = self.token_decoder._discretize_z(z_gt)
                pred_codes_norm = self.token_decoder._discretize_z(pred_z)
                half_width = self.token_decoder.frozen_quantizer._levels // 2
                gt_codes_int = (gt_codes_norm * half_width.float() + half_width.float()).long()
                pred_codes_int = (pred_codes_norm * half_width.float() + half_width.float()).long()
                for d in range(D_TOKEN):
                    correct = (pred_codes_int[:, :, d] == gt_codes_int[:, :, d]) & valid
                    token_errors[f'val/token_acc_dim{d}'] = correct.sum().item() / valid.sum().item()
                all_correct = ((pred_codes_int == gt_codes_int).all(dim=-1)) & valid
                token_errors['val/token_acc_joint'] = all_correct.sum().item() / valid.sum().item()

                z_mse = ((pred_z - z_gt) ** 2).sum(-1)
                token_errors['val/z_mse'] = z_mse[valid].mean().item()

        pos_heavyatom, pred_local, _ = self.token_decoder.decode_primary(
            pred_z, final_node_embed, rotmats_final, trans_final,
        )

        atom_mask = batch['mask_heavyatom'].bool() & gen_mask.bool().unsqueeze(-1)
        if atom_mask.any():
            atom_sq = ((pos_heavyatom - batch['pos_heavyatom']) ** 2).sum(dim=-1)
            struct_rmsd = atom_sq[atom_mask].mean().sqrt().item()
        else:
            struct_rmsd = 0.0

        error = {
            'val/trans_error': trans_error.item(),
            'val/aars_error': 1 - aar.item(),
            'val/rots_error': rots_error.mean().item(),
            'val/struct_rmsd': struct_rmsd,
        }
        error.update(token_errors)
        return error
