"""
Per-target bind_ratio vs RMSD 分析脚本。
从已有的 eval_metrics.pt 中提取数据，不需要 PDB 文件。

用法：
    python analyze_bind_rmsd_subset.py

输出：
    - sc_analysis_outputs/bind_ratio_by_rmsd.csv
    - 控制台打印各 RMSD 子集的 bind_ratio 对比
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
OUTPUT_DIR = Path(__file__).resolve().parent / "sc_analysis_outputs"

METHODS = {
    "pepsage": "pepsage",
    "PepGLAD": "pep_glad",
    "Discrete Token": "Discrete token",
    "Ours (no fix)": "exp8_r6_gaussian_v4_rebalance_2026-05-17-10_32_12",
    "Ours (fix)": "exp8_r6_gaussian_v4_fix",
}

RMSD_THRESHOLDS = [2.0, 3.0, 4.0, 5.0]


def load_per_target_metrics(log_dir: Path) -> dict:
    """加载 eval_metrics.pt，返回 {target_id: {metric: [values]}}"""
    pt_path = log_dir / "eval_metrics.pt"
    if not pt_path.exists():
        return {}
    return torch.load(pt_path, map_location="cpu")


def get_target_mean(data: dict, target_id: str, metric: str) -> float:
    """获取某 target 某指标的均值。"""
    if target_id not in data:
        return float("nan")
    values = data[target_id].get(metric, [])
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size > 0 else float("nan")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载所有方法的数据
    all_data = {}
    for name, dirname in METHODS.items():
        log_dir = LOGS_DIR / dirname
        data = load_per_target_metrics(log_dir)
        if data:
            all_data[name] = data
            print(f"Loaded {name}: {len(data)} targets")
        else:
            print(f"WARNING: {name} not found at {log_dir}")

    if not all_data:
        print("No data loaded!")
        sys.exit(1)

    # 获取所有 target 的公共集合
    all_targets = set()
    for data in all_data.values():
        all_targets.update(data.keys())

    # 计算每个方法每个 target 的 mean RMSD 和 mean bind_ratio
    results = []
    header = "target_id"
    for name in all_data:
        header += f",{name}_rmsd,{name}_bind_ratio"

    rows = [header]
    method_target_metrics = {}

    for name, data in all_data.items():
        method_target_metrics[name] = {}
        for target_id in all_targets:
            rmsd_val = get_target_mean(data, target_id, "rmsd")
            bind_val = get_target_mean(data, target_id, "bind_ratio")
            method_target_metrics[name][target_id] = {
                "rmsd": rmsd_val, "bind_ratio": bind_val
            }

    # 按 RMSD 阈值分析
    print("\n" + "=" * 70)
    print("Per-RMSD-threshold bind_ratio analysis")
    print("=" * 70)

    csv_rows = ["method,rmsd_threshold,num_targets,mean_bind_ratio,std_bind_ratio"]

    for threshold in RMSD_THRESHOLDS:
        print(f"\n--- Targets where Ours (fix) RMSD < {threshold} ---")
        # 用 Ours (fix) 的 RMSD 筛选 target 子集
        ours_data = method_target_metrics.get("Ours (fix)", {})
        subset_targets = [
            t for t, m in ours_data.items()
            if not np.isnan(m["rmsd"]) and m["rmsd"] < threshold
        ]
        print(f"  Num targets: {len(subset_targets)}")

        for name in all_data:
            bind_values = [
                method_target_metrics[name][t]["bind_ratio"]
                for t in subset_targets
                if t in method_target_metrics[name]
                and not np.isnan(method_target_metrics[name][t]["bind_ratio"])
            ]
            if bind_values:
                mean_br = np.mean(bind_values)
                std_br = np.std(bind_values)
                print(f"  {name:20s}: bind_ratio = {mean_br:.4f} ± {std_br:.4f} (n={len(bind_values)})")
                csv_rows.append(f"{name},{threshold},{len(bind_values)},{mean_br:.4f},{std_br:.4f}")

    # 保存 CSV
    csv_path = OUTPUT_DIR / "bind_ratio_by_rmsd.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_rows))
    print(f"\nSaved to: {csv_path}")

    # 额外：全局对比
    print("\n" + "=" * 70)
    print("Overall comparison (all targets)")
    print("=" * 70)
    for name in all_data:
        bind_values = [
            m["bind_ratio"] for m in method_target_metrics[name].values()
            if not np.isnan(m["bind_ratio"])
        ]
        rmsd_values = [
            m["rmsd"] for m in method_target_metrics[name].values()
            if not np.isnan(m["rmsd"])
        ]
        if bind_values:
            print(f"  {name:20s}: RMSD={np.mean(rmsd_values):.3f}, bind_ratio={np.mean(bind_values):.4f}")


if __name__ == "__main__":
    main()
