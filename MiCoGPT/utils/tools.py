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
    min_val_per_project=2,      # 每个 eligible project 至少抽 1 个进 val
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


def _suggest_k(n, lam):
    lam = float(lam)
    n = float(n)
    return n * (1.0 - lam) / max(lam, 1e-12)

# tools.py 顶部确保有这些 import（你大概率已有）
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from transformers.trainer_callback import TrainerCallback

class ProjectAggregatedEvalCallback(TrainerCallback):
    def __init__(
        self,
        trainer,                 # 需要传 trainer 进来
        eval_subset: Subset,     # 你的 val_set（Subset）
        project_col="Project_ID",
        shrink_k=3000.0,        # 你想要的默认值也可以改这里
        worst_frac=0.10,
        log_toksum_once=True,    # 第一次 eval 打印/记录 tok_sum 分布 + 建议 k
    ):
        self.trainer = trainer
        self.eval_subset = eval_subset
        self.project_col = project_col
        self.shrink_k = float(shrink_k)
        self.worst_frac = float(worst_frac)
        self.log_toksum_once = bool(log_toksum_once)
        self._running = False
        self._tok_logged = False

        base = eval_subset.dataset
        if base.metadata is None:
            raise ValueError("eval_subset.dataset.metadata 为空")
        if project_col not in base.metadata.columns:
            raise ValueError(f"metadata 中没有列 '{project_col}'")

    @torch.no_grad()
    def _compute_project_losses(self):
        trainer = self.trainer
        model = trainer.model
        model.eval()

        dl = trainer.get_eval_dataloader(self.eval_subset)

        base = self.eval_subset.dataset
        meta = base.metadata
        val_indices = np.array(self.eval_subset.indices)  # base_corpus 的行号（依赖 eval dataloader 顺序不乱）

        loss_sum = {}
        tok_sum = {}

        total_loss = 0.0
        total_tok = 0

        pos = 0
        for batch in dl:
            batch = trainer._prepare_inputs(batch)

            # 需要 labels 来算 token-level loss
            if "labels" not in batch:
                labels = batch["input_ids"].clone()
                if "attention_mask" in batch:
                    labels[batch["attention_mask"] == 0] = -100
                else:
                    tok = getattr(trainer, "tokenizer", None)
                    pad_id = tok.pad_token_id if tok is not None else model.config.pad_token_id
                    labels[labels == pad_id] = -100
                batch["labels"] = labels

            labels = batch["labels"]
            bsz = labels.size(0)

            base_idx = val_indices[pos:pos + bsz]
            pos += bsz
            proj_ids = meta.iloc[base_idx][self.project_col].astype(str).to_numpy()

            model_inputs = {k: v for k, v in batch.items() if k != "labels"}
            out = model(**model_inputs)
            logits = out.logits  # [B,T,V]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            V = shift_logits.size(-1)
            loss_flat = F.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(bsz, -1)  # [B, T-1]

            mask = (shift_labels != -100)
            loss_per_sample = (loss_flat * mask).sum(dim=1)           # [B]
            tok_per_sample  = mask.sum(dim=1).to(torch.long)          # [B]

            total_loss += float(loss_per_sample.sum().cpu().item())
            total_tok  += int(tok_per_sample.sum().cpu().item())

            for pid, ls, nt in zip(proj_ids, loss_per_sample, tok_per_sample):
                nt_i = int(nt.item())
                if nt_i == 0:
                    continue
                loss_sum[pid] = loss_sum.get(pid, 0.0) + float(ls.cpu().item())
                tok_sum[pid]  = tok_sum.get(pid, 0)   + nt_i

        micro = total_loss / max(total_tok, 1)
        proj_loss = {pid: loss_sum[pid] / max(tok_sum[pid], 1) for pid in loss_sum.keys()}
        return micro, proj_loss, tok_sum

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._running or metrics is None:
            return

        self._running = True
        try:
            micro, proj_loss, tok_sum = self._compute_project_losses()
            if len(proj_loss) == 0:
                return

            # 记录你实际用的 shrink_k（避免“我明明传了10000但我不确定生效没”）
            metrics["eval_shrink_k"] = float(self.shrink_k)

            # 第一次 eval：log tok_sum 分布 + 给 shrink_k 建议
            if self.log_toksum_once and (not self._tok_logged):
                self._tok_logged = True

                tok = np.array(list(tok_sum.values()), dtype=np.float64)
                tok_min = float(tok.min())
                tok_p10 = float(np.quantile(tok, 0.10))
                tok_p50 = float(np.quantile(tok, 0.50))
                tok_p90 = float(np.quantile(tok, 0.90))
                tok_max = float(tok.max())

                def _suggest_k(n, lam):
                    # lam = n/(n+k)  =>  k = n*(1-lam)/lam
                    return float(n * (1.0 - lam) / max(lam, 1e-12))

                metrics.update({
                    "eval_tok_min": tok_min,
                    "eval_tok_p10": tok_p10,
                    "eval_tok_p50": tok_p50,
                    "eval_tok_p90": tok_p90,
                    "eval_tok_max": tok_max,
                    "suggest_shrink_k_p10_lam01": _suggest_k(tok_p10, 0.10),
                    "suggest_shrink_k_p10_lam02": _suggest_k(tok_p10, 0.20),
                    "suggest_shrink_k_p50_lam05": _suggest_k(tok_p50, 0.50),
                })

                print(
                    f"[tok_sum] min={tok_min:.0f}, p10={tok_p10:.0f}, p50={tok_p50:.0f}, p90={tok_p90:.0f}, max={tok_max:.0f}\n"
                    f"[suggest_k] p10@lam=0.1 -> {metrics['suggest_shrink_k_p10_lam01']:.0f} | "
                    f"p10@lam=0.2 -> {metrics['suggest_shrink_k_p10_lam02']:.0f} | "
                    f"p50@lam=0.5 -> {metrics['suggest_shrink_k_p50_lam05']:.0f}"
                )

            # sqrt 加权（不是 project 平权；更接近“token多的study更可信”）
            ws = np.array([math.sqrt(tok_sum[p]) for p in proj_loss.keys()], dtype=np.float64)
            vs = np.array([proj_loss[p] for p in proj_loss.keys()], dtype=np.float64)
            proj_sqrt = float((ws * vs).sum() / max(ws.sum(), 1e-12))

            # shrinkage（小study向micro拉回）
            shrink = {}
            for p, Lp in proj_loss.items():
                n = float(tok_sum[p])
                lam = n / (n + self.shrink_k)
                shrink[p] = lam * Lp + (1.0 - lam) * micro

            proj_shrink = float(np.mean(list(shrink.values())))
            k = max(1, int(math.ceil(self.worst_frac * len(shrink))))
            worst10 = float(np.mean(sorted(shrink.values(), reverse=True)[:k]))

            metrics["eval_proj_micro_recomputed"] = float(micro)
            metrics["eval_proj_sqrt"] = proj_sqrt
            metrics["eval_proj_shrink"] = proj_shrink
            metrics["eval_proj_worst10_shrink"] = worst10
            metrics["eval_proj_count"] = float(len(shrink))

            print(
                f"[ProjEval] projects={len(shrink)} | k={self.shrink_k:.0f} | "
                f"sqrt={proj_sqrt:.4f} | shrink={proj_shrink:.4f} | "
                f"worst{int(self.worst_frac*100)}(shrink)={worst10:.4f} | micro={micro:.4f}"
            )
        finally:
            self._running = False

