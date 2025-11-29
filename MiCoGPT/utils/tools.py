import re
import numpy as np
import pandas as pd
from typing import Optional
from torch.utils.data import Subset

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
