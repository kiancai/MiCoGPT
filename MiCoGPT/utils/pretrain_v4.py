import math
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

# PART1: 读取 npz 并构造 prior_matrix
def build_prior_matrix_from_npz(tokenizer, npz_path: str, vocab_size: int, n_embd: int):
    # 加载提前计算好的 genus 数组和 embeddings 作为后续使用的先验向量
    data = np.load(npz_path, allow_pickle=True)
    genus_array = np.array(data["genus"], dtype=str)
    emb = data["embeddings"]

    # 读取提前计算好的 genus 数组和 embeddings, 检查维度是否匹配
    if emb.shape[1] != n_embd:
        raise ValueError(f"NPZ embedding dim={emb.shape[1]} != model n_embd={n_embd}")

    # 初始化 prior_matrix 为全 0
    # 因为 GPT2 的权重默认为 float32，所以 embedding 也为 float32
    prior = torch.zeros(vocab_size, n_embd, dtype=torch.float32)

    missing = []
    genus_token_ids = []

    # 因为不会出现 genus 冗余，所以直接写入 prior_matrix 即可
    for g_str, vec in zip(genus_array, emb):
        token_id = tokenizer.convert_tokens_to_ids(g_str)

        # token_id 为 None 或越界都视为 missing
        if token_id is None or token_id < 0 or token_id >= vocab_size:
            missing.append(g_str)
            continue

        # 直接把该 genus 的 DNA embedding 写入对应 token_id 的 prior 行
        prior[token_id] = torch.from_numpy(vec).to(torch.float32)
        genus_token_ids.append(token_id)

    # unique + sort，保证输出稳定
    genus_token_ids = sorted(set(genus_token_ids))

    print(f"[prior] npz genus: {len(genus_array)}")
    print(f"[prior] prior unique token_id: {len(genus_token_ids)}")
    print(f"[prior] missing genus: {len(missing)}")
    if missing:
        print(f"[prior] missing genus example: {missing[:5]} ...")
    return prior, genus_token_ids, missing


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# PART2: Embedding Wrapper
class GatedPriorEmbedding(nn.Module):
    # 自定义了一个 GatedPriorEmbedding 类，用于替换 GPT2 模型的 embedding 层
    # E_eff = E_train + w_vec(token) ⊙ E_prior
    # w_vec(token) = g_min + (1-g_min) * sigmoid(gate_logits[token, :])
    #
    # 与原版区别：
    # - gate_logits 从每 token 一个标量，改为每 token 一个 D 维向量（逐维门控）
    # - 这样先验可以在不同维度上被不同强度地注入

    def __init__(
        self,
        base: nn.Embedding,
        prior_matrix: torch.Tensor,
        g_min: float = 0.0,
        init_w: float = 0.1,
    ):
        super().__init__()

        # 简单参数检查
        if not (0.0 <= g_min < 1.0 and 0.0 < init_w < 1.0 and init_w >= g_min):
            raise ValueError("Invalid parameters: g_min、init_w")

        # base 为 GPT2 的 wte 层（可训练）
        self.base = base

        # 先验向量矩阵，固定不训练（buffer 会跟随 device/dtype 保存与迁移）
        # 形状: [V, D]
        self.register_buffer("prior_matrix", prior_matrix)

        self.g_min = float(g_min)

        vocab_size, n_embd = prior_matrix.shape

        # 让 w 的初值约等于 init_w
        # 将 init_w 映射回 gate_logits 的初值：logit( (init_w-g_min)/(1-g_min) )
        inner = (init_w - self.g_min) / max(1e-6, (1.0 - self.g_min))
        inner = min(max(inner, 1e-6), 1 - 1e-6)  # 防止出现 log(0) 或 log(∞)
        logit0 = math.log(inner / (1.0 - inner))

        # 每个 token 一个 D 维 gate_logits（逐维门控），初值全为 logit0
        # 形状: [V, D]
        self.gate_logits = nn.Parameter(torch.full((vocab_size, n_embd), logit0, dtype=torch.float32))

    @property
    def weight(self):
        # 为了保持 GPT2 的 tie_weights 正常工作
        # HF 的 tie_weights 会用到 get_input_embeddings().weight
        return self.base.weight

    def forward(self, input_ids: torch.LongTensor):
        # 前向传播：计算 E_eff = E_train + w_vec(token) ⊙ E_prior
        #
        # base_emb: GPT2 的 wte 输出，即 E_train, 形状 [B, T, D]
        # prior_emb: 先验向量，根据 input_ids 取出, 形状 [B, T, D]
        # gate_logits[input_ids]: 逐 token、逐维 logits, 形状 [B, T, D]
        # w_vec: 映射到 [g_min, 1] 的逐维门控, 形状 [B, T, D]
        #
        # 返回：E_eff, 形状 [B, T, D]
        base_emb = self.base(input_ids)          # [B, T, D]
        prior_emb = self.prior_matrix[input_ids] # [B, T, D]

        # 逐维门控（注意：这里 w 的形状是 [B, T, D]）
        w = self.g_min + (1.0 - self.g_min) * torch.sigmoid(self.gate_logits[input_ids])  # [B, T, D]

        return base_emb + w * prior_emb


# PART3: 挂到 GPT2 模型上
def attach_gated_prior_to_gpt2(
    model,
    tokenizer,
    npz_path,
    g_min: float = 0.0,
    init_w: float = 0.1,
    shuffle_prior: bool = False,
    shuffle_seed: int = 42,
    prior_scale=None,   # None: 自动对齐；float: 手动缩放；1.0: 不缩放
):
    vocab_size = model.config.vocab_size
    n_embd = model.config.n_embd

    prior_matrix, genus_token_ids, missing = build_prior_matrix_from_npz(
        tokenizer=tokenizer,
        npz_path=str(npz_path),
        vocab_size=vocab_size,
        n_embd=n_embd,
    )

    # 随机打乱先验向量（只在有 prior 的 token 行之间置换）
    if shuffle_prior and len(genus_token_ids) > 1:
        g = torch.Generator(device="cpu")
        g.manual_seed(shuffle_seed)
        ids = torch.tensor(genus_token_ids, dtype=torch.long)
        perm = torch.randperm(ids.numel(), generator=g)
        prior_matrix[ids] = prior_matrix[ids[perm]]
        print(f"[prior] shuffled prior vectors among {len(genus_token_ids)} genus tokens (seed={shuffle_seed})")

    # 先拿到 base embedding（原始 wte）
    base_wte = model.transformer.wte

    # ====== 对齐 / 手动缩放 prior（全局缩放 prior_matrix）======
    # 注意：即便门控变成向量，这个“初始化尺度对齐”仍然有意义，
    # 因为它会影响训练初期 prior 支路的数值量级与 gate 的有效梯度范围。
    with torch.no_grad():
        if isinstance(prior_scale, (int, float)):
            # 手动指定全局缩放系数
            s = float(prior_scale)
            prior_matrix = prior_matrix * s
            print(f"[prior] applied MANUAL scale s={s:.4f} to prior_matrix")
        else:
            # 自动对齐（默认：p50 align）
            base_norm = base_wte.weight.detach().float().norm(dim=-1)   # [V]
            prior_norm = prior_matrix.detach().float().norm(dim=-1)     # [V]
            mask = prior_norm > 0

            if mask.any():
                s = torch.quantile(base_norm[mask], 0.50) / torch.clamp_min(
                    torch.quantile(prior_norm[mask], 0.50), 1e-12
                )
                prior_matrix = prior_matrix * s
                print(f"[prior] applied AUTO scale s={float(s.item()):.4f} to prior_matrix (p50 align)")
            else:
                print("[prior] no nonzero prior rows found, skip scaling")

    # 搬到模型 device 与 dtype（与 base_wte.weight 对齐）
    prior_matrix = prior_matrix.to(dtype=base_wte.weight.dtype, device=base_wte.weight.device)

    # 用向量门控版本替换 wte
    model.transformer.wte = GatedPriorEmbedding(
        base=base_wte,
        prior_matrix=prior_matrix,
        g_min=g_min,
        init_w=init_w,
    )

    # 继续使用 HF 的 tie_weights：让 lm_head 的 base 权重与 wte.base.weight 共享
    model.tie_weights()
    return genus_token_ids, missing


class GatedPriorLMHead(nn.Module):
    # 将先验信息也注入到输出 logits：
    #
    # logits = base_lm_head(hidden) + scale * (hidden @ prior_w^T)
    # prior_w = prior_matrix ⊙ w_vec_vocab
    # w_vec_vocab = g_min + (1-g_min) * sigmoid(gate_logits)   # [V, D]
    #
    # 注意：这里的 gate_logits 是“每 token 一个 D 维向量”，因此 prior_w 也是逐维缩放得到的 [V, D]

    def __init__(self, base_lm_head: nn.Linear, wte: nn.Module, prior_logits_scale: float = 1.0):
        super().__init__()
        self.base = base_lm_head      # 原来的 lm_head（保持 tie 的那部分逻辑）
        self.wte = wte                # 直接引用 GatedPriorEmbedding，从这里读取 prior_matrix / gate_logits / g_min
        self.scale = float(prior_logits_scale)

    @property
    def weight(self):
        # 兼容外部访问 lm_head.weight（例如某些保存/检查逻辑）
        return self.base.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, T, D]
        logits = self.base(hidden_states)  # [B, T, V]

        # vocab 级别、逐维门控：w_vec_vocab 形状 [V, D]
        w = self.wte.g_min + (1.0 - self.wte.g_min) * torch.sigmoid(self.wte.gate_logits)  # [V, D]
        w = w.to(dtype=hidden_states.dtype, device=hidden_states.device)

        # prior_w: [V, D]（逐维缩放后的先验“输出权重增量”）
        prior_w = self.wte.prior_matrix.to(dtype=hidden_states.dtype, device=hidden_states.device) * w  # [V, D]

        # 使用 F.linear: hidden @ prior_w^T -> [B, T, V]
        logits = logits + self.scale * F.linear(hidden_states, prior_w)
        return logits


def attach_gated_prior_lm_head(model, prior_logits_scale: float = 1.0):
    # 必须在 model.tie_weights() 之后调用（否则 tie 会要求 lm_head 还是 nn.Linear）
    model.lm_head = GatedPriorLMHead(model.lm_head, model.transformer.wte, prior_logits_scale)


# 下面是一份计算 val 集合的 balanced_loss 的代码
# 用 balanced_loss 评估模型在 val 集合上的性能
def _ensure_labels(batch: dict, pad_token_id: int) -> dict:
    if "labels" in batch:
        return batch
    labels = batch["input_ids"].clone()
    if "attention_mask" in batch:
        labels[batch["attention_mask"] == 0] = -100
    else:
        labels[labels == pad_token_id] = -100
    batch["labels"] = labels
    return batch

class BalancedEvalTrainer(Trainer):
    def __init__(self, *args, project_col="Project_ID", mode="macro", T=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_col = project_col
        self.mode = mode      # "macro" or "temp"
        self.T = float(T)

    @torch.no_grad()
    def _compute_balanced_loss(self, eval_dataset):
        model = self.model
        model.eval()

        dl = self.get_eval_dataloader(eval_dataset)

        # eval_dataset 可能是 Subset：需要拿到 base + indices 才能对齐 metadata
        if hasattr(eval_dataset, "dataset") and hasattr(eval_dataset, "indices"):
            base = eval_dataset.dataset
            base_indices = np.array(eval_dataset.indices)
        else:
            base = eval_dataset
            base_indices = np.arange(len(eval_dataset))

        meta = base.metadata.iloc[base_indices]
        proj_ids_all = meta[self.project_col].astype(str).to_numpy()

        loss_sum = {}
        tok_sum = {}
        pos = 0

        pad_id = self.tokenizer.pad_token_id if self.tokenizer is not None else model.config.pad_token_id

        for batch in dl:
            batch = self._prepare_inputs(batch)
            batch = _ensure_labels(batch, pad_token_id=pad_id)

            labels = batch["labels"]
            bsz = labels.size(0)

            # 按 dataloader 顺序对应 metadata 行
            proj_ids = proj_ids_all[pos:pos + bsz]
            pos += bsz

            out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask", None))
            logits = out.logits  # [B,T,V]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            V = shift_logits.size(-1)
            loss_flat = F.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(bsz, -1)

            mask = (shift_labels != -100)
            loss_per_sample = (loss_flat * mask).sum(dim=1)      # 每条样本 loss 总和
            tok_per_sample = mask.sum(dim=1).to(torch.long)      # 每条样本有效 token 数

            for pid, ls, nt in zip(proj_ids, loss_per_sample, tok_per_sample):
                nt_i = int(nt.item())
                if nt_i == 0:
                    continue
                loss_sum[pid] = loss_sum.get(pid, 0.0) + float(ls.item())
                tok_sum[pid] = tok_sum.get(pid, 0) + nt_i

        # 每个 project 的平均 token loss
        proj_loss = {p: loss_sum[p] / max(tok_sum[p], 1) for p in loss_sum.keys()}
        if not proj_loss:
            return float("nan")

        # 聚合成 balanced loss
        if self.mode == "macro":
            # 每个 project 等权
            return float(np.mean(list(proj_loss.values())))

        elif self.mode == "temp":
            # 温度加权：w_p ∝ n_p^(alpha), alpha=1/T（这里用 token 数做 n_p）
            alpha = 1.0 / max(self.T, 1e-6)
            ps = list(proj_loss.keys())
            ns = np.array([tok_sum[p] for p in ps], dtype=np.float64)
            ws = np.power(ns, alpha)
            ws = ws / ws.sum()
            ls = np.array([proj_loss[p] for p in ps], dtype=np.float64)
            return float((ws * ls).sum())

        else:
            raise ValueError("mode must be 'macro' or 'temp'")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_ds is None:
            raise ValueError("eval_dataset is None")

        # 跑标准 evaluation_loop 拿到 eval_loss 等指标（不触发 callbacks）
        dataloader = self.get_eval_dataloader(eval_ds)
        start_time = time.time()

        output = self.evaluation_loop(
            dataloader,
            description="Evaluation",
            prediction_loss_only=True,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        metrics = output.metrics  # 这里已经有 eval_loss, eval_runtime 等

        # 在 callbacks 之前注入你要的指标
        balanced = self._compute_balanced_loss(eval_ds)
        metrics[f"{metric_key_prefix}_balanced_loss"] = balanced  # => eval_balanced_loss

        # 补充 runtime（可选，保持和原 Trainer 输出风格一致）
        metrics[f"{metric_key_prefix}_runtime"] = round(time.time() - start_time, 4)

        # 现在再 log + 触发 callbacks（EarlyStopping 在这里就能看到 eval_balanced_loss 了）
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics
