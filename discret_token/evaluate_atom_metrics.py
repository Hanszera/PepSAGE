import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from tqdm import tqdm

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from core.config.config import Config
from local_frame_struct_tokenizer.exp1_tokenizer.metrics import (
    BACKBONE_SLOTS,
    bond_angle_deviation,
    bond_length_deviation,
    clash_score,
    kabsch_align,
    rmsd,
    sample_length_bucket,
    torsion_metrics,
)
from local_frame_struct_tokenizer.exp1_tokenizer.pepsage_compat.parsers import parse_pdb


def _summarize_values(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "var": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
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


def _infer_chain_id(target_id: str) -> str | None:
    parts = target_id.split("_")
    if len(parts) >= 3:
        return parts[-3]
    return None


def _select_chain(data: dict, chain_id: str | None) -> dict | None:
    if data is None:
        return None
    if not data["chain_id"]:
        return None

    if chain_id is None or chain_id not in set(data["chain_id"]):
        chain_id = data["chain_id"][0]

    indices = [idx for idx, cid in enumerate(data["chain_id"]) if cid == chain_id]
    if not indices:
        return None

    idx_tensor = torch.tensor(indices, dtype=torch.long)
    selected = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            selected[key] = value.index_select(0, idx_tensor)
        elif isinstance(value, list):
            selected[key] = [value[idx] for idx in indices]
        else:
            selected[key] = value
    return selected


def _match_residue_indices(gt_data: dict, pred_data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    gt_keys = {(int(gt_data["resseq"][idx].item()), gt_data["icode"][idx]): idx for idx in range(gt_data["aa"].shape[0])}
    pred_keys = {(int(pred_data["resseq"][idx].item()), pred_data["icode"][idx]): idx for idx in range(pred_data["aa"].shape[0])}

    common_keys = [key for key in gt_keys.keys() if key in pred_keys]
    if not common_keys:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    gt_idx = torch.tensor([gt_keys[key] for key in common_keys], dtype=torch.long)
    pred_idx = torch.tensor([pred_keys[key] for key in common_keys], dtype=torch.long)
    return gt_idx, pred_idx


def _safe_float(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
    return float(value)


def compute_pair_metrics(gt_data: dict, pred_data: dict) -> dict[str, float]:
    gt_idx, pred_idx = _match_residue_indices(gt_data, pred_data)
    if gt_idx.numel() == 0:
        return {}

    gt_pos14 = gt_data["pos_heavyatom"].index_select(0, gt_idx).float()
    pred_pos14 = pred_data["pos_heavyatom"].index_select(0, pred_idx).float()
    gt_mask14 = gt_data["mask_heavyatom"].index_select(0, gt_idx).bool()
    pred_mask14 = pred_data["mask_heavyatom"].index_select(0, pred_idx).bool()
    common_mask14 = gt_mask14 & pred_mask14
    aa = gt_data["aa"].index_select(0, gt_idx).long()

    flat_mask = common_mask14.reshape(-1)
    if flat_mask.sum() == 0:
        return {}

    pred_valid = pred_pos14.reshape(-1, 3)[flat_mask]
    target_valid = gt_pos14.reshape(-1, 3)[flat_mask]
    pred_aligned = kabsch_align(pred_valid, target_valid)
    all_atom_rmsd = rmsd(pred_aligned, target_valid)

    slots = torch.arange(pred_pos14.shape[1], device=pred_pos14.device).repeat(pred_pos14.shape[0])
    valid_slots = slots[flat_mask]
    side_mask = torch.ones_like(valid_slots, dtype=torch.bool)
    for slot in BACKBONE_SLOTS:
        side_mask &= valid_slots != slot
    if side_mask.any():
        sidechain_rmsd = rmsd(pred_aligned[side_mask], target_valid[side_mask])
    else:
        sidechain_rmsd = torch.tensor(float("nan"))

    torsion = torsion_metrics(pred_pos14, gt_pos14, aa)
    clash = clash_score(pred_pos14, pred_mask14, aa)
    bond_len = bond_length_deviation(pred_pos14, gt_pos14, common_mask14, aa)
    bond_angle = bond_angle_deviation(pred_pos14, gt_pos14, common_mask14, aa)

    metrics = {
        "all_atom_rmsd": _safe_float(all_atom_rmsd),
        "sidechain_rmsd": _safe_float(sidechain_rmsd),
        "rotamer_recovery": _safe_float(torsion["rotamer_recovery"]),
        "clash_score": _safe_float(clash),
        "bond_length_dev": _safe_float(bond_len),
        "bond_angle_dev_deg": _safe_float(bond_angle),
        f"{sample_length_bucket(int(gt_idx.numel()))}_all_atom_rmsd": _safe_float(all_atom_rmsd),
    }
    for key, value in torsion.items():
        metrics[key] = _safe_float(value)
    return metrics


class EvalAtomMetrics(Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pep_dir = self.cfg.accounting.generated_pep_dir

    def eval_metric(self):
        pdb_ids = os.listdir(self.pep_dir)
        eval_res = {}
        summary_res = {}

        for pdb_id in tqdm(pdb_ids, desc="Evaluating atom-level metrics"):
            gt_pdb_path = os.path.join(self.pep_dir, pdb_id, "gt.pdb")
            if not os.path.exists(gt_pdb_path):
                rank_zero_info(f"Ground-truth file missing: {gt_pdb_path}")
                continue

            chain_id = _infer_chain_id(pdb_id)
            gt_full, _ = parse_pdb(gt_pdb_path)
            gt_data = _select_chain(gt_full, chain_id)
            if gt_data is None:
                rank_zero_info(f"Unable to parse GT chain for {gt_pdb_path}")
                continue

            eval_res[pdb_id] = {}
            for sample_idx in range(self.cfg.num_samples):
                sample_path = os.path.join(self.pep_dir, pdb_id, f"sample_{sample_idx}.pdb")
                if not os.path.exists(sample_path):
                    continue
                try:
                    pred_full, _ = parse_pdb(sample_path)
                    pred_data = _select_chain(pred_full, chain_id)
                    if pred_data is None:
                        continue
                    metrics = compute_pair_metrics(gt_data, pred_data)
                    for key, value in metrics.items():
                        eval_res[pdb_id].setdefault(key, []).append(value)
                except Exception as exc:
                    rank_zero_info(f"Error processing atom metrics for {sample_path}: {exc}")

            summary_res[pdb_id] = {
                metric: _summarize_values(values) for metric, values in eval_res[pdb_id].items()
            }

        out_name = "eval_atom_metrics_sc" if "generated_pep_packsc" in self.pep_dir else "eval_atom_metrics"
        torch.save(eval_res, os.path.join(self.cfg.accounting.logdir, f"{out_name}.pt"))
        torch.save(
            {
                "per_target": summary_res,
                "aggregate": _aggregate_summaries(summary_res),
            },
            os.path.join(self.cfg.accounting.logdir, f"{out_name}_summary.pt"),
        )

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.global_rank == 0:
            self.eval_metric()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sc_packing", action="store_true")
    parser.add_argument("--pep_dir", type=str, default=None)
    args = parser.parse_args()

    config_file = os.path.join(args.root_dir, "config.yaml")
    cfg = Config(config_file)
    if args.sc_packing:
        cfg.accounting.generated_pep_dir = os.path.join(os.path.dirname(cfg.accounting.generated_pep_dir), "generated_pep_packsc")
    if args.pep_dir is not None:
        cfg.accounting.generated_pep_dir = args.pep_dir

    cfg.num_samples = args.num_samples
    eval_callback = EvalAtomMetrics(cfg=cfg)
    eval_callback.on_test_end(EasyDict({"global_rank": 0}), None)
