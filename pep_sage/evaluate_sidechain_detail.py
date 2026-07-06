"""
补充侧链分析脚本：骨架对齐后SC-RMSD、Per-AA-type SC-RMSD、χ联合准确率

在 GPU 服务器上运行，需要访问生成的 PDB 文件。
用法：
    python evaluate_sidechain_detail.py --root_dir logs/exp8_r6_gaussian_v4_fix --num_samples 64
    python evaluate_sidechain_detail.py --root_dir logs/pepsage --num_samples 64
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

EXP8_ROOT = Path(__file__).resolve().parents[0]
if str(EXP8_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP8_ROOT))

from exp1_local_frame_tokenizer.exp1_tokenizer.metrics import (
    BACKBONE_SLOTS,
    circular_abs_diff,
    kabsch_align,
    rmsd,
)
from exp1_local_frame_tokenizer.exp1_tokenizer.pepsage_compat.parsers import parse_pdb
from core.models.torsion import get_torsion_angle
from evaluate_atom_metrics import (
    _infer_chain_id,
    _select_chain,
    _match_residue_indices,
    _summarize_values,
    _aggregate_summaries,
)

# PLACEHOLDER_CONTINUE

# 氨基酸名称映射（用于 per-AA-type 分析）
AA_NAMES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL", "UNK",
]

# 按侧链复杂度分组
SHORT_SC = {0, 7, 14, 15, 16, 19}  # ALA, GLY, PRO, SER, THR, VAL
MEDIUM_SC = {2, 3, 4, 6, 9, 10}    # ASN, ASP, CYS, GLU, ILE, LEU
LONG_SC = {1, 5, 8, 11, 12, 13, 17, 18}  # ARG, GLN, HIS, LYS, MET, PHE, TRP, TYR


def compute_kabsch_transform(pred: torch.Tensor, target: torch.Tensor):
    """计算 kabsch 对齐的旋转矩阵和平移。"""
    pred_center = pred.mean(0)
    target_center = target.mean(0)
    pred_centered = pred - pred_center
    target_centered = target - target_center
    H = pred_centered.T @ target_centered
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    sign_matrix = torch.diag(torch.tensor([1.0, 1.0, d], device=pred.device))
    R = Vt.T @ sign_matrix @ U.T
    return R, pred_center, target_center

# PLACEHOLDER_METRICS


def compute_sidechain_detail_metrics(gt_data: dict, pred_data: dict) -> dict:
    """计算详细侧链指标。"""
    gt_idx, pred_idx = _match_residue_indices(gt_data, pred_data)
    if gt_idx.numel() == 0:
        return {}

    gt_pos14 = gt_data["pos_heavyatom"].index_select(0, gt_idx).float()
    pred_pos14 = pred_data["pos_heavyatom"].index_select(0, pred_idx).float()
    gt_mask14 = gt_data["mask_heavyatom"].index_select(0, gt_idx).bool()
    pred_mask14 = pred_data["mask_heavyatom"].index_select(0, pred_idx).bool()
    common_mask14 = gt_mask14 & pred_mask14
    aa = gt_data["aa"].index_select(0, gt_idx).long()
    L = aa.shape[0]

    flat_mask = common_mask14.reshape(-1)
    if flat_mask.sum() == 0:
        return {}

    pred_valid = pred_pos14.reshape(-1, 3)[flat_mask]
    target_valid = gt_pos14.reshape(-1, 3)[flat_mask]

    slots = torch.arange(pred_pos14.shape[1]).repeat(L)
    valid_slots = slots[flat_mask]

    bb_mask = torch.zeros_like(flat_mask)
    for i in range(L):
        for slot in BACKBONE_SLOTS:
            idx = i * pred_pos14.shape[1] + slot
            if idx < flat_mask.shape[0] and flat_mask[idx]:
                bb_mask[flat_mask.cumsum(0)[idx] - 1 if flat_mask[idx] else 0] = True

    # 重新计算：用 valid_slots 判断骨架
    bb_valid_mask = torch.zeros(valid_slots.shape[0], dtype=torch.bool)
    for slot in BACKBONE_SLOTS:
        bb_valid_mask |= (valid_slots == slot)
    sc_valid_mask = ~bb_valid_mask

    metrics = {}

    # --- 分析 1：骨架对齐后 SC-RMSD ---
    if bb_valid_mask.sum() >= 3 and sc_valid_mask.any():
        bb_pred = pred_valid[bb_valid_mask]
        bb_target = target_valid[bb_valid_mask]
        R, pred_c, target_c = compute_kabsch_transform(bb_pred, bb_target)
        pred_bb_aligned = (pred_valid - pred_c) @ R.T + target_c
        sc_rmsd_bb_aligned = rmsd(
            pred_bb_aligned[sc_valid_mask], target_valid[sc_valid_mask]
        )
        metrics["sc_rmsd_bb_aligned"] = float(sc_rmsd_bb_aligned)

# PLACEHOLDER_PERAA

    # --- 分析 2：Per-AA-type SC-RMSD ---
    residue_starts = []
    for i in range(L):
        start = (torch.arange(pred_pos14.shape[1]) + i * pred_pos14.shape[1])
        residue_starts.append(start)

    # 按残基分组，逐残基计算侧链RMSD（骨架对齐后）
    for group_name, group_set in [
        ("short_sc", SHORT_SC), ("medium_sc", MEDIUM_SC), ("long_sc", LONG_SC)
    ]:
        group_sc_dists = []
        for i in range(L):
            if int(aa[i].item()) not in group_set:
                continue
            res_mask = torch.zeros(pred_pos14.shape[1], dtype=torch.bool)
            for slot in range(pred_pos14.shape[1]):
                if slot not in BACKBONE_SLOTS and common_mask14[i, slot]:
                    res_mask[slot] = True
            if not res_mask.any():
                continue
            res_pred = pred_pos14[i, res_mask]
            res_gt = gt_pos14[i, res_mask]
            if bb_valid_mask.sum() >= 3:
                res_pred_aligned = (res_pred - pred_c) @ R.T + target_c
                dist = ((res_pred_aligned - res_gt) ** 2).sum(-1).sqrt().mean()
                group_sc_dists.append(float(dist))
        if group_sc_dists:
            metrics[f"{group_name}_sc_rmsd"] = float(np.mean(group_sc_dists))

    # Per-AA-type 细分
    for aa_idx in range(20):
        aa_mask_res = (aa == aa_idx)
        if not aa_mask_res.any():
            continue
        aa_sc_dists = []
        for i in range(L):
            if aa[i].item() != aa_idx:
                continue
            res_sc_mask = torch.zeros(pred_pos14.shape[1], dtype=torch.bool)
            for slot in range(pred_pos14.shape[1]):
                if slot not in BACKBONE_SLOTS and common_mask14[i, slot]:
                    res_sc_mask[slot] = True
            if not res_sc_mask.any():
                continue
            res_pred = pred_pos14[i, res_sc_mask]
            res_gt = gt_pos14[i, res_sc_mask]
            if bb_valid_mask.sum() >= 3:
                res_pred_aligned = (res_pred - pred_c) @ R.T + target_c
                dist = ((res_pred_aligned - res_gt) ** 2).sum(-1).sqrt().mean()
                aa_sc_dists.append(float(dist))
        if aa_sc_dists:
            metrics[f"aa_{AA_NAMES[aa_idx]}_sc_rmsd"] = float(np.mean(aa_sc_dists))

# PLACEHOLDER_CHI

    # --- 分析 3：χ 角联合准确率 ---
    pred_torsion, pred_tmask = get_torsion_angle(pred_pos14, aa)
    gt_torsion, gt_tmask = get_torsion_angle(gt_pos14, aa)

    chi_threshold = 20.0  # degrees
    for depth in range(1, 5):  # chi1, chi1+2, chi1+2+3, chi1+2+3+4
        joint_mask = torch.ones(L, dtype=torch.bool)
        joint_correct = torch.ones(L, dtype=torch.bool)
        for chi_idx in range(1, depth + 1):
            valid = pred_tmask[:, chi_idx] & gt_tmask[:, chi_idx]
            joint_mask &= valid
            if valid.any():
                diff_deg = torch.rad2deg(
                    circular_abs_diff(pred_torsion[:, chi_idx], gt_torsion[:, chi_idx])
                )
                correct = diff_deg <= chi_threshold
                joint_correct &= (correct | ~valid)
        if joint_mask.any():
            acc = float(joint_correct[joint_mask].float().mean())
            metrics[f"chi_joint_acc_depth{depth}"] = acc

    return metrics


# PLACEHOLDER_MAIN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="生成样本的根目录")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--output_dir", default="data_analyze/sc_analysis_outputs")
    parser.add_argument("--run_name", default=None, help="输出文件中的方法名称")
    args = parser.parse_args()

    pep_dir = args.root_dir
    if not os.path.isdir(pep_dir):
        print(f"Error: {pep_dir} not found")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name or os.path.basename(args.root_dir)

    pdb_ids = [d for d in os.listdir(pep_dir) if os.path.isdir(os.path.join(pep_dir, d))]
    eval_res = {}
    summary_res = {}

    for pdb_id in tqdm(pdb_ids, desc=f"Evaluating SC detail [{run_name}]"):
        gt_pdb_path = os.path.join(pep_dir, pdb_id, "gt.pdb")
        if not os.path.exists(gt_pdb_path):
            continue

        chain_id = _infer_chain_id(pdb_id)
        gt_full, _ = parse_pdb(gt_pdb_path)
        gt_data = _select_chain(gt_full, chain_id)
        if gt_data is None:
            continue

        eval_res[pdb_id] = {}
        for sample_idx in range(args.num_samples):
            sample_path = os.path.join(pep_dir, pdb_id, f"sample_{sample_idx}.pdb")
            if not os.path.exists(sample_path):
                continue
            try:
                pred_full, _ = parse_pdb(sample_path)
                pred_data = _select_chain(pred_full, chain_id)
                if pred_data is None:
                    continue
                metrics = compute_sidechain_detail_metrics(gt_data, pred_data)
                for key, value in metrics.items():
                    eval_res[pdb_id].setdefault(key, []).append(value)
            except Exception as exc:
                print(f"Error: {sample_path}: {exc}")

        summary_res[pdb_id] = {
            metric: _summarize_values(values)
            for metric, values in eval_res[pdb_id].items()
        }

    aggregate = _aggregate_summaries(summary_res)

    out_path = os.path.join(args.output_dir, f"sc_detail_{run_name}.pt")
    torch.save({
        "per_target": summary_res,
        "aggregate": aggregate,
        "run_name": run_name,
    }, out_path)

    print(f"\n=== {run_name} Aggregate Results ===")
    for metric, stats in sorted(aggregate.items()):
        print(f"  {metric}: {stats['mean_of_means']:.4f} ± {stats['std_of_means']:.4f}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()





