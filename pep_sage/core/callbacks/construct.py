import os
import sys
sys.path.append("/data10/java/CH")
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torch.nn.functional as F
from tqdm.auto import tqdm
from core.utils.train import recursive_to
from core.modules.protein.writers import save_pdb
from core.models.torsion import get_heavyatom_mask
import argparse
from easydict import EasyDict


class ConsPep(Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pep_dir = self.cfg.accounting.generated_pep_dir
        self.test_dir = self.cfg.accounting.test_outputs_dir

    def save_samples_sc(self, samples, save_dir):
        batch = recursive_to(samples['batch'], 'cpu')
        chain_id = [list(item) for item in zip(*batch['chain_id'])][0]
        icode = [' ' for _ in range(len(chain_id))]

        samples['seqs'] = samples['seqs'].argmax(-1)
        mask_new = get_heavyatom_mask(samples['seqs'])

        if 'pos_heavyatom' in samples and samples['pos_heavyatom'] is not None:
            pos_ha = samples['pos_heavyatom']
        else:
            pos_ha = batch['pos_heavyatom']

        if pos_ha.shape[2] < 15:
            pos_ha = F.pad(pos_ha, pad=(0, 0, 0, 15 - pos_ha.shape[2]), value=0.)

        pos_new = torch.where(
            batch['generate_mask'][:, :, None, None], pos_ha, batch['pos_heavyatom']
        )
        pos_new = torch.where(mask_new[:, :, :, None], pos_new, torch.zeros_like(pos_new))
        aa_new = samples['seqs']

        for i in range(self.cfg.num_samples):
            data_saved = {
                'chain_nb': batch['chain_nb'][0],
                'chain_id': chain_id,
                'resseq': batch['resseq'][0],
                'icode': icode,
                'aa': aa_new[i],
                'mask_heavyatom': mask_new[i],
                'pos_heavyatom': pos_new[i],
            }
            save_pdb(data_saved, path=os.path.join(save_dir, f'sample_{i}.pdb'))

        data_saved = {
            'chain_nb': batch['chain_nb'][0],
            'chain_id': chain_id,
            'resseq': batch['resseq'][0],
            'icode': icode,
            'aa': batch['aa'][0],
            'mask_heavyatom': batch['mask_heavyatom'][0],
            'pos_heavyatom': batch['pos_heavyatom'][0],
        }
        save_pdb(data_saved, path=os.path.join(save_dir, f'gt.pdb'))

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.global_rank == 0:
            print("Saving PEP results...")
            names = [n.split('.')[0] for n in os.listdir(self.test_dir) if n.split('.')[1] == 'pt']
            for name in tqdm(names):
                pdb_dir = os.path.join(self.pep_dir, name)
                sample = torch.load(os.path.join(self.test_dir, f'{name}.pt'))
                os.makedirs(pdb_dir, exist_ok=True)
                self.save_samples_sc(sample[-1], pdb_dir)

            print(f"PEP results saved to {self.test_dir}")


if __name__ == "__main__":
    pass
