import os
import sys
sys.path.append("/data10/java/CH")
import pytorch_lightning as pl
import numpy as np
import torch
import argparse
from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from tqdm import tqdm
from core.utils.geometry import get_chain_from_pdb, get_CA_dist, get_psi_chi, diff_ratio, get_seq 
from easydict import EasyDict


def _summarize_values(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "var": np.nan, "min": np.nan, "max": np.nan}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "var": float(np.var(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _aggregate_summaries(per_target_summary):
    aggregate = {}
    metric_names = set()
    for summary in per_target_summary.values():
        metric_names.update(summary.keys())

    for metric in sorted(metric_names):
        means = []
        vars_ = []
        stds = []
        for summary in per_target_summary.values():
            metric_summary = summary.get(metric)
            if not metric_summary or metric_summary["count"] == 0:
                continue
            means.append(metric_summary["mean"])
            vars_.append(metric_summary["var"])
            stds.append(metric_summary["std"])
        if not means:
            continue
        aggregate[metric] = {
            "num_targets": len(means),
            "mean_of_means": float(np.mean(means)),
            "std_of_means": float(np.std(means)),
            "mean_of_stds": float(np.mean(stds)),
            "mean_of_vars": float(np.mean(vars_)),
        }
    return aggregate


class EvalPep(Callback):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        self.cfg = cfg
        self.pep_dir = self.cfg.accounting.generated_pep_dir
    
    def eval_metric(self):
        """
        Evaluate the metrics of the generated samples and save them.
        """
        pdb_ids = os.listdir(self.pep_dir)
        eval_res = {}
        summary_res = {}
        for pdb_id in tqdm(pdb_ids, desc="Evaluating metrics"):
            gt_pdb_path = os.path.join(self.pep_dir, pdb_id, 'gt.pdb')
            gt_chain_id = pdb_id.split('_')[-3]  
            gt_chain = get_chain_from_pdb(gt_pdb_path, gt_chain_id)

            eval_res[pdb_id] = {
                        'gt_CA_dist': [],
                        'sample_CA_dist': [],
                        'aar':[]
                    }
            eval_res[pdb_id]['gt_CA_dist'].append(get_CA_dist(gt_chain))
            for i in range(self.cfg.num_samples):
                pdb_id_sample = f"sample_{i}.pdb"
                pdb_i_path = os.path.join(self.pep_dir, pdb_id, pdb_id_sample)
                
                if not os.path.exists(pdb_i_path):
                    rank_zero_info(f"Sample {pdb_i_path} does not exist.")
                else:
                    pdb_i_chain = get_chain_from_pdb(pdb_i_path, gt_chain_id)
                    try:
                        eval_res[pdb_id]['sample_CA_dist'].append(get_CA_dist(pdb_i_chain))
                    except Exception as e:
                        rank_zero_info(f"Error checking peptide validity for {pdb_i_path}: {e}")
                    
                    try:

                        aar = diff_ratio(get_seq(pdb_i_chain), get_seq(gt_chain))
                        eval_res[pdb_id]['aar'].append(aar)
                    except Exception as e:
                        rank_zero_info(f"Error calculating AAR for {pdb_i_path}: {e}")
            summary_res[pdb_id] = {
                'gt_CA_dist': _summarize_values(eval_res[pdb_id]['gt_CA_dist']),
                'sample_CA_dist': _summarize_values(eval_res[pdb_id]['sample_CA_dist']),
                'aar': _summarize_values(eval_res[pdb_id]['aar']),
            }
        if 'generated_pep_packsc' in self.pep_dir:
            torch.save(eval_res, os.path.join(self.cfg.accounting.logdir, 'eval_other_metrics_sc.pt'))
            torch.save(
                {
                    "per_target": summary_res,
                    "aggregate": _aggregate_summaries(summary_res),
                },
                os.path.join(self.cfg.accounting.logdir, 'eval_other_metrics_sc_summary.pt'),
            )
        else:
            torch.save(eval_res, os.path.join(self.cfg.accounting.logdir, 'eval_other_metrics.pt'))
            torch.save(
                {
                    "per_target": summary_res,
                    "aggregate": _aggregate_summaries(summary_res),
                },
                os.path.join(self.cfg.accounting.logdir, 'eval_other_metrics_summary.pt'),
            )
        
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Called when testing ends.
        """
        if trainer.global_rank == 0:
            self.eval_metric()
            
if __name__ == "__main__":

    pass
