import numpy as np
import pandas as pd
import torch
import os
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Subset
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2ForSequenceClassification
from MiCoGPT.utils.pretrain import attach_gated_prior_to_gpt2

def prepare_labels_for_subset(
    all_corpus,
    subset: Subset,
    label_col: str,
    encoder=None,
    fill_value: int = -1,
    verbose: bool = True,
):
    """
    输入：
      - all_corpus: 完整 MiCoGPTCorpus（必须有 .metadata, .sample_ids）
      - subset: 你选出来的微调子集（Subset(all_corpus, indices)）
      - label_col: metadata 里的标签列名，比如 "Is_Healthy"
      - encoder: 可选，传入已 fit 的 OneHotEncoder（推理/复用时用）；不传则在 subset 上 fit
    输出：
      labels_tensor: torch.long，长度= len(subset)，顺序与 subset 一致
      all_labels: np.int，长度=len(all_corpus)，非 subset 位置为 fill_value
      encoder: OneHotEncoder（fit 后的）
      num_labels: 类别数
    """
    meta = all_corpus.metadata
    if meta is None:
        raise ValueError("all_corpus.metadata is None")

    if label_col not in meta.columns:
        raise ValueError(f"metadata 中没有列 {label_col}")

    indices = np.array(subset.indices)  # 在 all_corpus 中的位置
    sample_ids = np.array(all_corpus.sample_ids)[indices]

    # 取出标签，并按 sample_id 对齐到 subset 的顺序
    labels_series = meta.loc[sample_ids, label_col]
    if labels_series.isna().any():
        # 如果你已经确保 subset 过滤过 notna，这里一般不会触发
        raise ValueError(f"subset 中仍然存在 NaN 标签：{labels_series.isna().sum()} 个")

    labels_df = labels_series.to_frame(name=label_col)

    # 编码
    if encoder is None:
        encoder = OneHotEncoder()
        arr = encoder.fit_transform(labels_df.values.reshape(-1, 1)).toarray()
    else:
        arr = encoder.transform(labels_df.values.reshape(-1, 1)).toarray()

    labels_tensor = torch.tensor(arr.argmax(axis=1), dtype=torch.long)
    num_labels = len(encoder.categories_[0])

    # 全局 labels（和 all_corpus 对齐）
    all_labels = np.full(len(all_corpus), fill_value=fill_value, dtype=int)
    all_labels[indices] = labels_tensor.numpy()

    if verbose:
        print(f"[labels] subset size={len(subset)}")
        print(f"[labels] num_labels={num_labels}")
        print(f"[labels] distribution:\n{pd.Series(labels_tensor.numpy()).value_counts().sort_index()}")

    return labels_tensor, all_labels, encoder, num_labels



def split_train_val_by_project_stratified_with_labels(
    dataset,
    label_col="Is_Healthy",
    project_col="Project_ID",
    val_ratio=0.20,
    min_project_samples=20,
    min_val_per_project=2,
    random_state=42,
    label_balance_strength=1.0,  # 越大越“拉平标签”，0 表示不管标签（退化成 pretrain 版）
):
    """
    逻辑：
    - 在当前 dataset 范围内按 Project_ID 统计样本数
    - 只对样本数 >= min_project_samples 的 project 做“内部抽样”进入 val
    - 小 project（以及 project_id 缺失）全部进 train
    - project 内抽样时引入 label 权重，尽量让 val 的 label 分布接近整体

    返回：
      train_set, val_set  (都是 Subset(base_corpus, global_indices))
    """

    # 兼容 dataset 是 Subset 或 Corpus
    if isinstance(dataset, Subset):
        base_corpus = dataset.dataset
        base_indices = np.array(dataset.indices)  # 在 base_corpus 中的全局索引
    else:
        base_corpus = dataset
        base_indices = np.arange(len(dataset))

    meta_full = base_corpus.metadata
    if meta_full is None:
        raise ValueError("base_corpus.metadata 为空")
    if project_col not in meta_full.columns:
        raise ValueError(f"metadata 中没有列 '{project_col}'")
    if label_col not in meta_full.columns:
        raise ValueError(f"metadata 中没有列 '{label_col}'")

    # 当前 dataset 范围内的 metadata（局部表）
    meta = meta_full.iloc[base_indices].copy()
    n_total = len(meta)
    target_val = int(round(n_total * val_ratio))

    # label（要求非空；你上游已经过滤过 notna，这里再防御一下）
    labels = meta[label_col]
    if labels.isna().any():
        raise ValueError(f"当前 dataset 中仍存在 NaN label：{labels.isna().sum()} 个，请先过滤。")

    # project（允许缺失，缺失的一律当 ineligible -> train）
    proj = meta[project_col]
    valid_proj_mask = proj.notna().to_numpy()

    # 只对 valid project 做 eligible 统计
    proj_valid = proj[valid_proj_mask].astype(str).to_numpy()
    sizes = pd.Series(proj_valid).value_counts()

    eligible = sizes[sizes >= min_project_samples]
    eligible_projects = list(eligible.index)
    eligible_total = int(eligible.sum())

    print(f"[split] total_samples={n_total}, target_val~{target_val}")
    print(f"[split] eligible_projects={len(eligible_projects)}, eligible_samples={eligible_total}")
    print(f"[split] ineligible_projects={int((sizes < min_project_samples).sum())}, ineligible_samples={int(sizes[sizes < min_project_samples].sum())}")
    print("[split] label_dist (overall):")
    print(labels.value_counts())

    if eligible_total == 0 or target_val <= 0:
        train_set = Subset(base_corpus, base_indices.tolist())
        val_set = Subset(base_corpus, [])
        print(f"[split] actual_val=0 (target~{target_val}), train={len(train_set)}")
        return train_set, val_set

    # 由于 val 只能从 eligible 中抽，所以在 eligible 内部的“有效抽样比例”
    r_eff = min(0.95, target_val / max(eligible_total, 1))
    rng = np.random.default_rng(random_state)

    # 全局 label 频数（用于做反频率加权）
    # 你也可以只用 eligible 部分的 label 分布，这里默认用整个当前 dataset 的分布
    label_counts = labels.astype(str).value_counts().to_dict()

    def label_weight(y):
        # 反频率权重：freq 越小 weight 越大
        # label_balance_strength=0 -> 全部权重=1（不管标签）
        if label_balance_strength <= 0:
            return 1.0
        c = float(label_counts.get(str(y), 1))
        return (1.0 / c) ** float(label_balance_strength)

    # project 配额分配（与 pretrain 类似）
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

    # largest remainder 调到 target_val（尽量接近）
    if diff != 0:
        if diff > 0:
            remainders.sort(key=lambda x: x[1], reverse=True)
            for pid, _, n in remainders:
                if diff == 0:
                    break
                if quotas[pid] < n - 1:
                    quotas[pid] += 1
                    diff -= 1
        else:
            remainders.sort(key=lambda x: x[1])
            for pid, _, _ in remainders:
                if diff == 0:
                    break
                if quotas[pid] > min_val_per_project:
                    quotas[pid] -= 1
                    diff += 1

    # 真正抽样：在每个 eligible project 内部按 label 权重抽 q 个
    is_val = np.zeros(n_total, dtype=bool)
    proj_full = proj.astype(str).to_numpy()
    labels_full = labels.astype(str).to_numpy()

    for pid in eligible_projects:
        idx = np.where(proj_full == pid)[0]  # meta 的局部索引
        n = len(idx)
        if n <= 1:
            continue
        q = min(quotas[pid], n - 1)

        # project 内样本权重（由 label 决定）
        w = np.array([label_weight(labels_full[i]) for i in idx], dtype=np.float64)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()

        pick = rng.choice(idx, size=q, replace=False, p=w)
        is_val[pick] = True

    # 映射回 base_corpus 的全局索引
    val_base_indices = base_indices[is_val]
    train_base_indices = base_indices[~is_val]

    train_set = Subset(base_corpus, train_base_indices.tolist())
    val_set   = Subset(base_corpus, val_base_indices.tolist())

    # 打印实际 val 标签分布
    val_labels = meta.iloc[np.where(is_val)[0]][label_col]
    print(f"[split] actual_val={len(val_set)} (target~{target_val}), train={len(train_set)}")
    print("[split] label_dist (val):")
    print(val_labels.value_counts())

    return train_set, val_set


class FinetuneDataset(Dataset):
    def __init__(self, base_corpus, indices, labels_array):
        self.base_corpus = base_corpus
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels_array, dtype=int)
        if len(self.indices) != len(self.labels):
            raise ValueError("indices 和 labels_array 的长度必须一致")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = int(self.indices[idx])
        item = dict(self.base_corpus[base_idx])  # 避免修改 base_corpus 内部对象
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item




def load_gpt2_cls_manual(
    model_dir: str,
    num_labels: int,
    mode: str,                  # "vanilla" 或 "gated"
    tokenizer=None,             # gated 需要
    npz_path=None,              # gated 需要
    g_min: float = 0.0,         # gated 需要
    init_w: float = 0.1,        # gated 需要
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode not in ("vanilla", "gated"):
        raise ValueError("mode must be 'vanilla' or 'gated'")

    # 1) config
    config = GPT2Config.from_pretrained(model_dir)
    config.num_labels = num_labels
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        config.pad_token_id = tokenizer.pad_token_id

    model = GPT2ForSequenceClassification(config)

    # 2) 如果你选择 gated：先把结构换成 gated（否则 state_dict key 对不上）
    if mode == "gated":
        if tokenizer is None or npz_path is None:
            raise ValueError("mode='gated' 需要 tokenizer 和 npz_path")
        attach_gated_prior_to_gpt2(
            model=model,
            tokenizer=tokenizer,
            npz_path=npz_path,
            g_min=g_min,
            init_w=init_w,
        )

    # 3) 读权重（bin/safetensors 都兼容）
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    st_path  = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        state = load_file(st_path, device="cpu")
    else:
        state = torch.load(bin_path, map_location="cpu")

    # 4) 加载（允许 lm_head.* unexpected；分类头 score.* missing 属于正常）
    incompatible = model.load_state_dict(state, strict=False)
    print(f"[load:{mode}] missing_keys={len(incompatible.missing_keys)}, unexpected_keys={len(incompatible.unexpected_keys)}")
    
    print("missing keys:", incompatible.missing_keys)
    print("unexpected keys:", incompatible.unexpected_keys)

    model.to(device)
    return model, device


import math
import numpy as np
import torch

def _q(x: torch.Tensor, p: float) -> float:
    return float(torch.quantile(x, p).item())

@torch.no_grad()
def print_gated_stats(model, tokenizer=None, npz_path=None, topk=0):
    """
    打印：
    - prior 非零行数量
    - base/prior 范数分位数
    - 当前对齐比：p50(base)/p50(prior)
    - 若给 tokenizer+npz_path：估计“相对原始 npz 的真实缩放倍率”
    - gate 的 logits 与 w 的统计（全 vocab + genus token）
    """
    wte = model.transformer.wte
    if not (hasattr(wte, "prior_matrix") and hasattr(wte, "gate_logits") and hasattr(wte, "base")):
        print("[gated-stats] current model wte is not GatedPriorEmbedding, skip.")
        return

    base_w = wte.base.weight.detach().float().cpu()            # [V,D]
    prior  = wte.prior_matrix.detach().float().cpu()           # [V,D]
    logits = wte.gate_logits.detach().float().cpu()            # [V]
    g_min  = float(getattr(wte, "g_min", 0.0))

    base_norm = base_w.norm(dim=-1)                            # [V]
    prior_norm = prior.norm(dim=-1)                            # [V]
    mask = prior_norm > 0
    K = int(mask.sum().item())
    V = prior.size(0)

    print(f"[gated-stats] vocab={V}, prior_nonzero_rows={K}, g_min={g_min}")

    if K > 0:
        b = base_norm[mask]
        p = prior_norm[mask]
        print(f"[gated-stats] base_norm  p10/p50/p90 = { _q(b,0.10):.4f} / { _q(b,0.50):.4f} / { _q(b,0.90):.4f}")
        print(f"[gated-stats] prior_norm p10/p50/p90 = { _q(p,0.10):.4f} / { _q(p,0.50):.4f} / { _q(p,0.90):.4f}")

        # 当前“再对齐一次需要的比例”（如果 prior 已经对齐，通常接近 1）
        s_now = _q(b, 0.50) / max(_q(p, 0.50), 1e-12)
        print(f"[gated-stats] current p50-align ratio (base/prior) = {s_now:.4f}  (≈1 means already aligned)")

    # gate -> w
    w_all = g_min + (1.0 - g_min) * torch.sigmoid(logits)      # [V]
    print(f"[gated-stats] gate_logits mean/min/max = {float(logits.mean()):.4f} / {float(logits.min()):.4f} / {float(logits.max()):.4f}")
    print(f"[gated-stats] w_all      mean/min/max = {float(w_all.mean()):.4f} / {float(w_all.min()):.4f} / {float(w_all.max()):.4f}")
    print(f"[gated-stats] w_all      p10/p50/p90 = { _q(w_all,0.10):.4f} / { _q(w_all,0.50):.4f} / { _q(w_all,0.90):.4f}")

    if K > 0:
        w_g = w_all[mask]
        print(f"[gated-stats] w_genus    mean/min/max = {float(w_g.mean()):.4f} / {float(w_g.min()):.4f} / {float(w_g.max()):.4f}")
        print(f"[gated-stats] w_genus    p10/p50/p90 = { _q(w_g,0.10):.4f} / { _q(w_g,0.50):.4f} / { _q(w_g,0.90):.4f}")

    # 估计“相对原始 npz 的真实缩放倍率”
    if tokenizer is not None and npz_path is not None and K > 0:
        data = np.load(str(npz_path), allow_pickle=True)
        genus = np.array(data["genus"], dtype=str)
        emb = data["embeddings"]

        raw = torch.zeros_like(prior)  # 未缩放 raw prior
        ids = []
        for g_str, vec in zip(genus, emb):
            tid = tokenizer.convert_tokens_to_ids(g_str)
            if tid is None or tid < 0 or tid >= V:
                continue
            raw[tid] = torch.from_numpy(vec).float()
            ids.append(tid)

        ids = sorted(set(ids))
        if len(ids) > 0:
            raw_norm = raw[ids].norm(dim=-1)
            mdl_norm = prior[ids].norm(dim=-1)
            ok = raw_norm > 0
            ratios = (mdl_norm[ok] / raw_norm[ok]).cpu()
            if ratios.numel() > 0:
                s_est = float(torch.quantile(ratios, 0.50).item())
                print(f"[gated-stats] est scale vs raw-npz (median of norm ratios) = {s_est:.4f}")
                print(f"[gated-stats] scale ratio p10/p50/p90 = { _q(ratios,0.10):.4f} / { _q(ratios,0.50):.4f} / { _q(ratios,0.90):.4f}")
