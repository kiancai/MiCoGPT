from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch.utils.data import Subset
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

###############################
# 获取标签
###############################
def prepare_labels_for_subset(
    all_corpus,
    subset,
    label_col: str = "Is_Healthy",
    verbose: bool = True,
):
    """
    从 subset 对应的 metadata 中取出 label_col 作为监督标签，并编码成 0..C-1
    
    返回：
      labels_tensor: torch.LongTensor, shape [len(subset)]
      all_labels:    List，长度为 len(subset)，每个元素是原始标签值（未编码）
      le:            sklearn.preprocessing.LabelEncoder（已 fit）
      num_labels:    int，类别数
    """

    # --------- 1) 取 subset 对应的 metadata ---------
    # subset_by_metadata 通常会返回 torch.utils.data.Subset
    if isinstance(subset, Subset):
        base = subset.dataset
        idx = np.asarray(subset.indices, dtype=int)
    else:
        base = subset
        idx = np.arange(len(subset), dtype=int)

    if not hasattr(base, "metadata") or base.metadata is None:
        raise ValueError("base dataset 没有 metadata，无法生成标签。")

    sub_meta = base.metadata.iloc[idx].copy()

    if label_col not in sub_meta.columns:
        raise KeyError(f"metadata 中找不到标签列：{label_col}")

    # --------- 2) 取出原始标签并做健壮性检查 ---------
    raw = sub_meta[label_col]

    # 你前面已经 subset 过滤了 notna()，这里再检查一次，避免静默把 NA 编成字符串 "nan"
    if raw.isna().any():
        na_cnt = int(raw.isna().sum())
        raise ValueError(
            f"subset 中仍然存在 {na_cnt} 个 NA 标签（{label_col}）。"
            "如果你希望丢弃 NA，请在 subset_by_metadata 的条件里加上 .notna()。"
        )

    # 保留原始标签（用于之后输出/对齐）
    all_labels = raw.tolist()

    # --------- 3) LabelEncoder 编码为 0..C-1，得到 labels_tensor ---------
    le = LabelEncoder()
    y = le.fit_transform(all_labels)  # numpy array, int

    labels_tensor = torch.tensor(y, dtype=torch.long)
    num_labels = int(len(le.classes_))

    # --------- 4) 可选：打印分布 ---------
    if verbose:
        import pandas as pd
        print(f"[labels] label_col = {label_col}")
        print(f"[labels] num_labels = {num_labels}")
        # 显示：原始标签 -> 编码ID
        mapping = {str(c): int(i) for i, c in enumerate(le.classes_)}
        print("[labels] label2id:", mapping)

        # 统计每个类的数量（按原始标签名统计）
        vc = pd.Series(all_labels).value_counts(dropna=False)
        print("[labels] label counts:")
        print(vc)

    return labels_tensor, all_labels, le, num_labels



###############################
# 划分数据集
###############################
def _subset_to_base_and_indices(ds):
    """兼容 Dataset / Subset，取出 base dataset 和对应 indices"""
    if isinstance(ds, Subset):
        return ds.dataset, np.asarray(ds.indices, dtype=int)
    return ds, np.arange(len(ds), dtype=int)

def get_raw_labels_from_subset(subset, label_col: str):
    """从 subset 对应的 metadata 取出原始标签（字符串/数值都行）"""
    base, idx = _subset_to_base_and_indices(subset)
    if not hasattr(base, "metadata") or base.metadata is None:
        raise ValueError("dataset 没有 metadata，无法取标签")
    s = base.metadata.iloc[idx][label_col]
    if s.isna().any():
        # 你前面已经筛掉 NA，这里再兜底检查一下
        raise ValueError(f"subset 中仍然存在 NA 标签：{label_col}")
    return s.tolist()

class SubsetWithLabels(Dataset):
    """把（train_subset/val_subset）+ labels_tensor 包成 Trainer 能用的 dataset"""
    def __init__(self, subset, labels_tensor: torch.LongTensor):
        self.subset = subset
        self.labels = labels_tensor
        assert len(self.subset) == len(self.labels)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        item = dict(self.subset[i])  # 你的 MiCoGPTCorpus / Subset 返回 dict(input_ids, attention_mask, ...)
        item["labels"] = self.labels[i]
        return item



###############################
###############################



























# =========【代码块 1/多】工具函数：一次性粘贴到 ipynb 的第一个 cell 里运行 =========
import os
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from pickle import load as pkl_load

from transformers import AutoConfig, GPT2ForSequenceClassification


# ============================================================
# 0) 通用小工具
# ============================================================
def set_seed(seed: int = 42):
    """尽量保证可复现（注意：某些 CUDA 算子仍可能有非确定性）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_subset(ds: Union[Dataset, Subset]) -> Tuple[Dataset, np.ndarray]:
    """
    返回 base_dataset 和 indices（indices 是 base_dataset 的索引顺序）。
    - 若 ds 是 Subset：base=ds.dataset, indices=np.array(ds.indices)
    - 否则：base=ds, indices=arange(len(ds))
    """
    if isinstance(ds, Subset):
        return ds.dataset, np.asarray(ds.indices, dtype=int)
    return ds, np.arange(len(ds), dtype=int)


def get_tokenizer_from_corpus_or_subset(corpus_or_subset):
    """兼容 MiCoGPTCorpus / Subset(MiCoGPTCorpus)"""
    base, _ = unwrap_subset(corpus_or_subset)
    if not hasattr(base, "tokenizer"):
        raise AttributeError("找不到 tokenizer：请确认传入的是 MiCoGPTCorpus 或其 Subset")
    return base.tokenizer


def get_sample_ids(corpus_or_subset) -> List[str]:
    """
    兼容 MiCoGPTCorpus / Subset(MiCoGPTCorpus)
    用于保存预测结果时对齐 sample_id
    """
    base, idx = unwrap_subset(corpus_or_subset)
    if not hasattr(base, "sample_ids"):
        # 没有 sample_ids 也能跑，只是预测输出无法对齐样本名
        return [str(i) for i in range(len(corpus_or_subset))]
    return [base.sample_ids[i] for i in idx.tolist()]


def get_metadata_df(corpus_or_subset) -> Optional[pd.DataFrame]:
    """兼容 MiCoGPTCorpus / Subset(MiCoGPTCorpus)"""
    base, idx = unwrap_subset(corpus_or_subset)
    meta = getattr(base, "metadata", None)
    if meta is None:
        return None
    return meta.iloc[idx].copy()


def load_micogpt_corpus(pkl_path: str):
    """
    读取你之前保存的语料对象（通常是 pickle 出来的 MiCoGPTCorpus）。
    """
    with open(pkl_path, "rb") as f:
        return pkl_load(f)


def ensure_pad_token(model, tokenizer):
    """
    GPT2 默认没有 pad_token。
    但你的 MiCoGPTCorpus 会用 pad_token_id 来生成 attention_mask，
    所以要保证 tokenizer.pad_token_id 与 model.config.pad_token_id 都存在。
    """
    if tokenizer.pad_token_id is None:
        # 常见做法：把 eos 当 pad（只要与你语料构建时一致即可）
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


# ============================================================
# 1) 兼容 v6 / v9 的 gated-prior embedding（用于 finetune 加载权重）
# ============================================================
class GatedPriorEmbeddingCompat(nn.Module):
    """
    兼容两类 gate：
      - v6: gate_logits: [V]
      - v9: gate_logits: [V, D]

    forward:
      E = base(input_ids) + w(input_ids) * prior_matrix[input_ids]
      w:
        v6 -> [B, T]      (再 unsqueeze 到 [B,T,1])
        v9 -> [B, T, D]
    """
    def __init__(self, base: nn.Embedding, vocab_size: int, n_embd: int, gate_rank: int, g_min: float = 0.0):
        super().__init__()
        self.base = base
        self.g_min = float(g_min)

        # prior_matrix 是 buffer（不训练），key 名需要与 checkpoint 对齐（prior_matrix）
        # dtype 与 base.weight 保持一致，避免 load_state_dict 因 dtype 不一致报错
        self.register_buffer(
            "prior_matrix",
            torch.zeros(vocab_size, n_embd, dtype=base.weight.dtype),
        )

        # gate_logits：v6 为 [V]，v9 为 [V,D]
        if gate_rank == 1:
            self.gate_logits = nn.Parameter(torch.zeros(vocab_size, dtype=base.weight.dtype))
        elif gate_rank == 2:
            self.gate_logits = nn.Parameter(torch.zeros(vocab_size, n_embd, dtype=base.weight.dtype))
        else:
            raise ValueError(f"gate_rank must be 1 or 2, got {gate_rank}")

    @property
    def weight(self):
        # 保持与 GPT2 embedding 的访问习惯一致（有些地方会访问 wte.weight）
        return self.base.weight

    def forward(self, input_ids: torch.LongTensor):
        base_emb = self.base(input_ids)                         # [B,T,D]
        prior_emb = self.prior_matrix[input_ids].to(base_emb.dtype)

        if self.gate_logits.dim() == 1:
            # v6：标量门控
            w = self.g_min + (1.0 - self.g_min) * torch.sigmoid(self.gate_logits[input_ids])  # [B,T]
            return base_emb + w.unsqueeze(-1) * prior_emb
        else:
            # v9：向量门控
            w = self.g_min + (1.0 - self.g_min) * torch.sigmoid(self.gate_logits[input_ids])  # [B,T,D]
            return base_emb + w * prior_emb


def _is_local_dir(p: str) -> bool:
    return os.path.isdir(p)


def _find_checkpoint_files(model_dir: str):
    """
    兼容 HF save_pretrained 的多种落盘形式：
      - pytorch_model.bin
      - model.safetensors
      - sharded: *.index.json
    """
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    bin_index = os.path.join(model_dir, "pytorch_model.bin.index.json")

    st_path = os.path.join(model_dir, "model.safetensors")
    st_index = os.path.join(model_dir, "model.safetensors.index.json")

    if os.path.isfile(bin_index):
        return ("bin_sharded", bin_index)
    if os.path.isfile(st_index):
        return ("st_sharded", st_index)
    if os.path.isfile(st_path):
        return ("st_single", st_path)
    if os.path.isfile(bin_path):
        return ("bin_single", bin_path)
    return (None, None)


def _read_index_json(index_path: str):
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_checkpoint_keys(model_dir: str) -> set:
    """返回 checkpoint 中的所有 key，用于判定是否存在 gated prior。"""
    ckpt_type, ckpt_ref = _find_checkpoint_files(model_dir)
    if ckpt_type is None:
        return set()

    if ckpt_type == "bin_single":
        sd = torch.load(ckpt_ref, map_location="cpu")
        return set(sd.keys())

    if ckpt_type == "st_single":
        try:
            from safetensors.torch import load_file
        except Exception:
            return set()
        sd = load_file(ckpt_ref, device="cpu")
        return set(sd.keys())

    index = _read_index_json(ckpt_ref)
    return set(index.get("weight_map", {}).keys())


def _load_tensor_shape_from_checkpoint(model_dir: str, key: str) -> Optional[Tuple[int, ...]]:
    """
    为了判断 v6(1D gate) / v9(2D gate)，只读取 gate_logits 的 shape。
    """
    ckpt_type, ckpt_ref = _find_checkpoint_files(model_dir)
    if ckpt_type is None:
        return None

    if ckpt_type == "bin_single":
        sd = torch.load(ckpt_ref, map_location="cpu")
        return tuple(sd[key].shape) if key in sd else None

    if ckpt_type == "st_single":
        try:
            from safetensors.torch import load_file
        except Exception:
            return None
        sd = load_file(ckpt_ref, device="cpu")
        return tuple(sd[key].shape) if key in sd else None

    # sharded：从 index 找到该 key 在哪个 shard
    index = _read_index_json(ckpt_ref)
    weight_map = index.get("weight_map", {})
    if key not in weight_map:
        return None

    shard_filename = weight_map[key]
    shard_path = os.path.join(model_dir, shard_filename)

    if ckpt_type == "bin_sharded":
        shard_sd = torch.load(shard_path, map_location="cpu")
        return tuple(shard_sd[key].shape) if key in shard_sd else None

    if ckpt_type == "st_sharded":
        try:
            from safetensors.torch import load_file
        except Exception:
            return None
        shard_sd = load_file(shard_path, device="cpu")
        return tuple(shard_sd[key].shape) if key in shard_sd else None

    return None


def _coerce_state_dict_dtypes(sd: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    避免 load_state_dict 因 dtype 不一致报错：
    - 对于 sd 中「也存在于 model.state_dict()」的 key，
      如果 dtype 不一致，则把 sd[key] cast 成 model 对应 dtype。
    """
    model_sd = model.state_dict()
    for k, v in list(sd.items()):
        if not isinstance(v, torch.Tensor):
            continue
        if k in model_sd and model_sd[k].dtype != v.dtype:
            sd[k] = v.to(dtype=model_sd[k].dtype)
    return sd


def _load_weights_into_model(model: nn.Module, model_dir: str):
    """
    把本地目录的权重加载进 model（strict=False）。
    支持单文件 & sharded（bin / safetensors）。
    """
    ckpt_type, ckpt_ref = _find_checkpoint_files(model_dir)
    if ckpt_type is None:
        raise FileNotFoundError(f"Cannot find checkpoint in: {model_dir}")

    missing_all, unexpected_all = [], []

    def _apply_state_dict(sd):
        sd = _coerce_state_dict_dtypes(sd, model)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            missing_all.extend(missing)
        if unexpected:
            unexpected_all.extend(unexpected)

    if ckpt_type == "bin_single":
        sd = torch.load(ckpt_ref, map_location="cpu")
        _apply_state_dict(sd)
        return missing_all, unexpected_all

    if ckpt_type == "st_single":
        from safetensors.torch import load_file
        sd = load_file(ckpt_ref, device="cpu")
        _apply_state_dict(sd)
        return missing_all, unexpected_all

    # sharded
    index = _read_index_json(ckpt_ref)
    shard_files = sorted(set(index.get("weight_map", {}).values()))

    if ckpt_type == "bin_sharded":
        for fn in shard_files:
            sd = torch.load(os.path.join(model_dir, fn), map_location="cpu")
            _apply_state_dict(sd)
        return missing_all, unexpected_all

    if ckpt_type == "st_sharded":
        from safetensors.torch import load_file
        for fn in shard_files:
            sd = load_file(os.path.join(model_dir, fn), device="cpu")
            _apply_state_dict(sd)
        return missing_all, unexpected_all

    return missing_all, unexpected_all


def load_model_compat(model_name_or_path: str, num_labels: int, g_min: float = 0.0) -> GPT2ForSequenceClassification:
    """
    同一段 finetune 代码同时支持：
      1) 普通 GPT2（HF hub 或本地目录）
      2) v6 gated-prior（gate_logits: [V]）
      3) v9 gated-prior（gate_logits: [V, D]）

    关键策略：
      - 本地 checkpoint：先用 config 初始化 GPT2ForSequenceClassification，
        再“按需 patch transformer.wte”，最后 strict=False 加载权重。
    """
    if not _is_local_dir(model_name_or_path):
        # hub / 普通预训练：直接 from_pretrained
        return GPT2ForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)

    # 本地：用 config 初始化（避免 from_pretrained 因结构不一致而丢权重）
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.num_labels = num_labels
    model = GPT2ForSequenceClassification(config)

    keys = _list_checkpoint_keys(model_name_or_path)

    # 检测 gated-prior（v6/v9）：checkpoint 里会有 transformer.wte.base.weight + transformer.wte.gate_logits
    gated = ("transformer.wte.base.weight" in keys) and ("transformer.wte.gate_logits" in keys)

    if gated:
        gate_shape = _load_tensor_shape_from_checkpoint(model_name_or_path, "transformer.wte.gate_logits")
        gate_rank = 1 if (gate_shape is None or len(gate_shape) == 1) else 2

        vocab_size = model.config.vocab_size
        n_embd = model.config.n_embd
        base = model.transformer.wte

        model.transformer.wte = GatedPriorEmbeddingCompat(
            base=base,
            vocab_size=vocab_size,
            n_embd=n_embd,
            gate_rank=gate_rank,
            g_min=g_min,
        )
        print(f"[compat] Detected gated-prior checkpoint. gate_rank={gate_rank} (v6=1, v9=2)")

    # 加载权重（transformer.* 会正确灌入；分类头 score.* 缺失是正常的）
    missing, unexpected = _load_weights_into_model(model, model_name_or_path)
    if missing:
        print(f"[compat] missing keys (first 20): {missing[:20]}")
    if unexpected:
        print(f"[compat] unexpected keys (first 20): {unexpected[:20]}")

    return model


def print_gated_stats(model: nn.Module, tokenizer=None, topk: int = 10):
    """打印 gate 的一些统计，方便确认 v6/v9 的 checkpoint 是否正确加载。"""
    wte = getattr(getattr(model, "transformer", None), "wte", None)
    if wte is None or (not hasattr(wte, "gate_logits")):
        print("[gate] This model has no gate_logits (probably plain GPT2).")
        return

    with torch.no_grad():
        gl = wte.gate_logits.detach().float().cpu()
        if gl.dim() == 1:
            w = torch.sigmoid(gl)  # [V]
            print(f"[gate] v6-style scalar gate. w: mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")
            if tokenizer is not None:
                top = torch.topk(w, k=min(topk, w.numel())).indices.tolist()
                bot = torch.topk(-w, k=min(topk, w.numel())).indices.tolist()
                print("[gate] top tokens:", [tokenizer.convert_ids_to_tokens(i) for i in top])
                print("[gate] bot tokens:", [tokenizer.convert_ids_to_tokens(i) for i in bot])
        else:
            w = torch.sigmoid(gl)   # [V,D]
            w_mean = w.mean(dim=1)  # [V]
            print(f"[gate] v9-style vector gate. w_mean(token): mean={w_mean.mean():.4f}, std={w_mean.std():.4f}, min={w_mean.min():.4f}, max={w_mean.max():.4f}")


# ============================================================
# 2) 标签生成 + 数据集封装（Sequence Classification）
# ============================================================
@dataclass
class LabelEncoding:
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def build_labels_from_metadata(
    metadata: pd.DataFrame,
    label_col: str,
    allow_values: Optional[List[str]] = None,
    unknown_label: int = -1,
) -> Tuple[np.ndarray, LabelEncoding]:
    """
    从 metadata 的某一列生成整数标签。
    - unknown_label（默认 -1）：未知/缺失 -> 作为“预测集”
    - allow_values：只允许这些类别进入训练空间，其它都标成 unknown_label

    返回：
      labels: shape [N]
      encoding: label2id/id2label
    """
    if label_col not in metadata.columns:
        raise KeyError(f"metadata 中找不到列：{label_col}")

    raw = metadata[label_col].astype("string").fillna("")  # 统一成字符串，空值用 ""

    # 训练空间里的类别集合
    if allow_values is not None:
        allow_set = set(map(str, allow_values))
        classes = sorted(list(allow_set))
        known_mask = raw.isin(classes)
    else:
        # 自动：把非空的都当作“已知类别”
        classes = sorted([x for x in raw.unique().tolist() if str(x) != ""])
        known_mask = raw != ""

    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    labels = np.full(len(metadata), unknown_label, dtype=np.int64)
    known_idx = np.where(known_mask.to_numpy())[0]
    for i in known_idx:
        labels[i] = label2id[str(raw.iat[i])]

    return labels, LabelEncoding(label2id=label2id, id2label=id2label)


def split_train_val_pred(
    labels: np.ndarray,
    val_ratio: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    先把 unknown_label(-1) 的样本单独当作 pred 集；
    其余样本再划分 train/val。

    返回：
      train_idx, val_idx, pred_idx  （都是对“当前 dataset 顺序”的索引）
    """
    assert labels.ndim == 1
    N = labels.shape[0]
    all_idx = np.arange(N)

    pred_idx = all_idx[labels < 0]
    known_idx = all_idx[labels >= 0]
    known_y = labels[labels >= 0]

    if known_idx.size == 0:
        raise ValueError("没有任何已知标签样本（labels 全是 -1），无法训练。")

    rng = np.random.RandomState(random_state)

    if val_ratio <= 0.0:
        return known_idx, np.array([], dtype=int), pred_idx

    if stratify and len(np.unique(known_y)) > 1:
        # 分类别抽 val，避免整体抽样导致某些类完全消失
        val_mask = np.zeros_like(known_y, dtype=bool)
        for c in np.unique(known_y):
            c_pos = np.where(known_y == c)[0]
            rng.shuffle(c_pos)
            n_val = int(round(len(c_pos) * val_ratio))
            # 类别样本太少时，避免把该类抽空到 val 导致 train 中没有该类
            if len(c_pos) <= 1 or n_val <= 0:
                continue
            n_val = max(1, n_val)
            n_val = min(n_val, len(c_pos) - 1)
            val_mask[c_pos[:n_val]] = True
        val_idx = known_idx[val_mask]
        train_idx = known_idx[~val_mask]
    else:
        perm = known_idx.copy()
        rng.shuffle(perm)
        n_val = int(round(len(perm) * val_ratio))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

    return train_idx, val_idx, pred_idx


def split_train_val_by_group(
    metadata: pd.DataFrame,
    group_col: str,
    candidate_idx: np.ndarray,
    val_ratio: float = 0.2,
    random_state: int = 42,
    min_group_samples: int = 2,
    min_val_per_group: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按 group_col（例如 Project_ID）做“组内按比例划分”：
      - 每个 group 内随机抽 val_ratio 做 val
      - 组太小（<min_group_samples）则全部进 train（避免 train 为空或 val 太小）

    用法：你可以先把 labels!=-1 的 idx 作为 candidate_idx，再按 Project_ID 划分 train/val。
    """
    if group_col not in metadata.columns:
        raise KeyError(f"metadata 中找不到 group_col: {group_col}")

    rng = np.random.RandomState(random_state)
    groups = metadata.iloc[candidate_idx][group_col].astype(str).to_numpy()

    train_idx, val_idx = [], []
    for g in np.unique(groups):
        loc = candidate_idx[np.where(groups == g)[0]]
        if len(loc) < min_group_samples:
            train_idx.extend(loc.tolist())
            continue
        loc = loc.copy()
        rng.shuffle(loc)
        n_val = max(min_val_per_group, int(round(len(loc) * val_ratio)))
        n_val = min(n_val, len(loc) - 1)  # 至少留 1 个在 train
        val_idx.extend(loc[:n_val].tolist())
        train_idx.extend(loc[n_val:].tolist())

    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


class FinetuneDataset(Dataset):
    """把 MiCoGPTCorpus（或其 Subset） + labels 包装成 Trainer 可用的数据集。"""
    def __init__(self, corpus_or_subset: Union[Dataset, Subset], indices: np.ndarray, labels: np.ndarray):
        self.corpus = corpus_or_subset
        self.indices = np.asarray(indices, dtype=int)
        self.labels = np.asarray(labels, dtype=np.int64)
        if len(self.indices) != len(self.labels):
            raise ValueError(f"indices 长度({len(self.indices)}) != labels 长度({len(self.labels)})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        item = dict(self.corpus[int(self.indices[i])])
        item["labels"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item


class InferenceDataset(Dataset):
    """仅推理用：不返回 labels（trainer.predict 会直接输出 logits）。"""
    def __init__(self, corpus_or_subset: Union[Dataset, Subset], indices: np.ndarray):
        self.corpus = corpus_or_subset
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        return dict(self.corpus[int(self.indices[i])])


def compute_metrics(eval_pred):
    """HF Trainer 的分类指标"""
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "acc": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def decode_predictions(logits: np.ndarray, encoding: LabelEncoding) -> List[str]:
    """把 logits -> 预测类别名"""
    pred_ids = np.argmax(logits, axis=-1).tolist()
    return [encoding.id2label[int(i)] for i in pred_ids]


def save_prediction_csv(
    out_path: str,
    sample_ids: List[str],
    pred_label_names: List[str],
    pred_label_ids: Optional[List[int]] = None,
    pred_probs: Optional[np.ndarray] = None,
):
    """
    保存预测结果到 CSV：
      - sample_id
      - pred_label
      - （可选）pred_id
      - （可选）各类别概率 prob_0..prob_{C-1}
    """
    df = pd.DataFrame({"sample_id": sample_ids, "pred_label": pred_label_names})
    if pred_label_ids is not None:
        df["pred_id"] = pred_label_ids

    if pred_probs is not None:
        for j in range(pred_probs.shape[1]):
            df[f"prob_{j}"] = pred_probs[:, j]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
