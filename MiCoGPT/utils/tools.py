import re, math, torch
import numpy as np
import pandas as pd
from typing import Optional
from torch.utils.data import Subset
from torch.utils.data import Subset
import torch.nn.functional as F
from transformers.trainer_callback import TrainerCallback

RANK_PREFIX_MAP = {
    "Kingdom": "k__",
    "Phylum": "p__",
    "Class": "c__",
    "Order": "o__",
    "Family": "f__",
    "Genus": "g__",
    "Species": "s__",
}

def extract_taxon(raw_name: str, rank: str) -> Optional[str]:

    if rank not in RANK_PREFIX_MAP:
        raise ValueError(f"Unknown rank: {rank!r}.")

    name = str(raw_name).strip()   # 统一成字符串并去掉首尾空白
    name = name.replace("; ", ";")
    prefix = RANK_PREFIX_MAP[rank]
    pattern = rf"{re.escape(prefix)}[^;]*"  # 匹配从前缀后到分号或字符串结束的内容
    m = re.search(pattern, name)
    if m:
        return m.group(0)
    print(
        f"[warning] Could not find {rank} (prefix {prefix!r}) in: {raw_name!r}"
    )
    return None


def split_train_val_by_project(dataset, val_ratio=0.1, project_col="Project_ID", random_state=42):

    if isinstance(dataset, Subset):
        base_corpus = dataset.dataset
        base_indices = np.array(dataset.indices)
    else:
        base_corpus = dataset
        base_indices = np.arange(len(dataset))

    meta_full = base_corpus.metadata
    if meta_full is None:
        raise ValueError("base_corpus.metadata 为空，无法按 Project_ID 划分。")
    if project_col not in meta_full.columns:
        raise ValueError(f"metadata 中没有列 '{project_col}'")

    meta = meta_full.iloc[base_indices].copy()
    n_samples = meta.shape[0]
    target_val = int(n_samples * val_ratio)

    project_ids = meta[project_col].to_numpy()
    mask_not_nan = pd.notna(project_ids)
    project_ids_nonan = project_ids[mask_not_nan]
    unique_projects = np.unique(project_ids_nonan)

    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_projects)

    is_val = np.zeros(n_samples, dtype=bool)
    val_projects = []
    val_count = 0

    for pid in unique_projects:
        if val_count >= target_val:
            break
        proj_mask = (project_ids == pid)
        proj_size = proj_mask.sum()
        if proj_size == 0:
            continue
        is_val |= proj_mask
        val_projects.append(pid)
        val_count += proj_size

    val_base_indices = base_indices[is_val]
    train_base_indices = base_indices[~is_val]

    train_set = Subset(base_corpus, train_base_indices.tolist())
    val_set   = Subset(base_corpus, val_base_indices.tolist())

    print(
        f"[split_by_project] val projects={len(val_projects)}, "
        f"val samples={len(val_set)} (target~{target_val}), total={n_samples}"
    )
    print(f"[split_by_project] Train={len(train_set)}, Val={len(val_set)}")
    return train_set, val_set



def split_train_val_by_project_stratified(
    dataset,
    project_col="Project_ID",
    val_ratio=0.10,
    min_project_samples=20,
    random_state=42,
    min_val_per_project=2,      # 每个 eligible project 至少抽 2 个进 val
):
    """
    逻辑：
    - 以 Project_ID 作为 study 分组
    - 只对样本数>=min_project_samples 的 project 做“内部抽样”进 val
    - 目标：val 总样本数 ≈ val_ratio * (当前dataset总样本数)
    - 小 project（<min_project_samples）全部放 train（不参与 val）
    """

    # 兼容 dataset 是 Subset 或 Corpus
    if isinstance(dataset, Subset):
        base_corpus = dataset.dataset
        base_indices = np.array(dataset.indices)
    else:
        base_corpus = dataset
        base_indices = np.arange(len(dataset))

    meta_full = base_corpus.metadata
    if meta_full is None:
        raise ValueError("base_corpus.metadata 为空")
    if project_col not in meta_full.columns:
        raise ValueError(f"metadata 中没有列 '{project_col}'")

    meta = meta_full.iloc[base_indices].copy()
    n_total = len(meta)
    target_val = int(round(n_total * val_ratio))

    proj = meta[project_col].astype(str).to_numpy()
    sizes = pd.Series(proj).value_counts()
    eligible = sizes[sizes >= min_project_samples]
    eligible_projects = list(eligible.index)
    eligible_total = int(eligible.sum())

    print(f"[split] total_samples={n_total}, target_val~{target_val}")
    print(f"[split] eligible_projects={len(eligible_projects)}, eligible_samples={eligible_total}")
    print(f"[split] ineligible_projects={int((sizes < min_project_samples).sum())}, ineligible_samples={int(sizes[sizes < min_project_samples].sum())}")

    if eligible_total == 0 or target_val <= 0:
        # 全部进 train
        train_set = Subset(base_corpus, base_indices.tolist())
        val_set = Subset(base_corpus, [])
        return train_set, val_set

    # 由于 val 只能从 eligible 中抽，所以在 eligible 内的“有效抽样比例”：
    r_eff = min(0.95, target_val / max(eligible_total, 1))  # 防止过大/过接近1
    rng = np.random.default_rng(random_state)

    # 先做一个“配额分配”：每个 project 抽多少个到 val，使得总和接近 target_val
    quotas = {}
    remainders = []
    for pid in eligible_projects:
        n = int(eligible[pid])
        ideal = n * r_eff
        q = int(np.floor(ideal))
        q = max(min_val_per_project, q)
        q = min(q, n - 1)  # 至少留 1 个在 train
        quotas[pid] = q
        remainders.append((pid, ideal - np.floor(ideal), n))

    cur = sum(quotas.values())
    diff = target_val - cur

    # 用 largest remainder 法把总数调到 target_val（在约束允许范围内）
    if diff != 0:
        if diff > 0:
            remainders.sort(key=lambda x: x[1], reverse=True)  # 余数大的先加
            for pid, _, n in remainders:
                if diff == 0: break
                if quotas[pid] < n - 1:
                    quotas[pid] += 1
                    diff -= 1
        else:
            remainders.sort(key=lambda x: x[1])               # 余数小的先减
            for pid, _, _ in remainders:
                if diff == 0: break
                if quotas[pid] > min_val_per_project:
                    quotas[pid] -= 1
                    diff += 1

    # 真正抽样
    is_val = np.zeros(len(meta), dtype=bool)
    for pid in eligible_projects:
        idx = np.where(proj == pid)[0]  # meta 内局部索引
        q = quotas[pid]
        pick = rng.choice(idx, size=q, replace=False)
        is_val[pick] = True

    val_base_indices = base_indices[is_val]
    train_base_indices = base_indices[~is_val]

    train_set = Subset(base_corpus, train_base_indices.tolist())
    val_set   = Subset(base_corpus, val_base_indices.tolist())

    print(f"[split] actual_val={len(val_set)} (target~{target_val}), train={len(train_set)}")
    return train_set, val_set

