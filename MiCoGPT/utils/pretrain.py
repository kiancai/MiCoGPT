import math
import torch
import numpy as np
import torch.nn as nn
from transformers.trainer_callback import TrainerCallback
from contextlib import contextmanager

# PART1: 读取 npz 并构造 prior_matrix
def load_genus_embeddings(npz_path: str):
    # 加载提前计算好的 genus 数组和 embeddings 作为后续使用的先验向量
    data = np.load(npz_path, allow_pickle=True)
    genus = np.array(data["genus"], dtype=str)
    emb = data["embeddings"]
    return genus, emb

def build_prior_matrix_from_npz(tokenizer, npz_path: str, vocab_size: int, n_embd: int):
    
    # 读取提前计算好的 genus 数组和 embeddings, 检查维度是否匹配
    genus_array, emb = load_genus_embeddings(npz_path)
    # 检查读取的 embeddings 维度与 config 中设置的模型维度是否匹配（n_embd=256）
    if emb.shape[1] != n_embd:
        raise ValueError(f"NPZ embedding dim={emb.shape[1]} != model n_embd={n_embd}")

    # 初始化 prior_matrix 为全 0
    # 因为 GPT2 的权重默认为 float32，所以 embedding 也为 float32
    prior = torch.zeros(vocab_size, n_embd, dtype=torch.float32)

    # MiCoGPTokenizer 其实没有定义 unk_token_id，这里不起作用
    # 但 HuggingFace 常见 tokenizer 很多都有 unk_token_id
    unk_id = getattr(tokenizer, "unk_token_id", None)
    missing = []

    # 防止多个 genus 映射到同一个 token_id
    # 出现这种情况时，取平均作为先验向量
    # 但因为实际上我已经提前去过冗余求平均了
    # 所以这里其实也不会起作用
    acc = {}
    cnt = {}

    for g_str, vec in zip(genus_array, emb):
        token_id = tokenizer.convert_tokens_to_ids(g_str)

        # 关键：很多 tokenizer 找不到不会抛 KeyError，而是返回 unk_id
        if token_id is None or (unk_id is not None and token_id == unk_id):
            missing.append(g_str)
            continue
        if token_id < 0 or token_id >= vocab_size:
            missing.append(g_str)
            continue
        # 如果多个 genus 映射到同一个 token_id，取平均作为先验向量
        if token_id not in acc:
            acc[token_id] = torch.zeros(n_embd, dtype=torch.float32)
            cnt[token_id] = 0
        acc[token_id] += torch.from_numpy(vec).to(torch.float32)
        cnt[token_id] += 1

    genus_token_ids = []
    for token_id in acc:
        prior[token_id] = acc[token_id] / max(cnt[token_id], 1)
        genus_token_ids.append(token_id)

    genus_token_ids.sort()

    print(f"[prior] npz genus 总数: {len(genus_array)}")
    print(f"[prior] 写入 prior 的 unique token_id 数: {len(genus_token_ids)}")
    print(f"[prior] missing/unk genus 数: {len(missing)}")
    if missing:
        print(f"[prior] missing 示例: {missing[:5]} ...")
    return prior, genus_token_ids, missing



# PART2: Embedding Wrapper

def _logit(p: float) -> float:
    # sigmoid 的反函数，用来“反推 logits 初值”
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))

class GatedPriorEmbedding(nn.Module):
    # 自定义了一个 GatedPriorEmbedding 类，用于替换 GPT2 模型的 embedding 层
    """
      E_eff = E_train + w(token) * E_prior
      w(token) = g_min + (1-g_min)*sigmoid(gate_logits[token])

    - base: 原始 nn.Embedding（可训练）
    - prior_matrix: 固定先验 [vocab, dim]（buffer，不训练）
    - gate_logits: 每个 token 一个标量 [vocab]（训练）
    """

    def __init__(self, base: nn.Embedding, prior_matrix: torch.Tensor, g_min: float = 0.1, init_w: float = 0.5):
        super().__init__()
        # 基本参数检查，确保 g_min 和 init_w 在合法范围内
        if not (0.0 <= g_min < 1.0):
            raise ValueError("g_min must be in [0, 1).")
        if not (0.0 < init_w < 1.0):
            raise ValueError("init_w must be in (0, 1).")
        if init_w < g_min:
            raise ValueError("init_w should be >= g_min.")

        # base 为 GPT2 的 wte 层
        self.base = base
        # 先验向量，固定不训练
        self.register_buffer("prior_matrix", prior_matrix)
        self.g_min = float(g_min)

        vocab_size = prior_matrix.shape[0]

        # 让 w 的初值约等于 init_w
        # 比如我设置 init_w=0.5，那么 gate_logits 初值为 logit(0.5) = 0.0
        inner = (init_w - self.g_min) / max(1e-6, (1.0 - self.g_min))
        logit0 = _logit(inner)

        # 可以关闭 ablation
        # 为了在 forward 时可以对同一个 batch 计算两次
        # 监控 prior 对 loss 有没有实质贡献
        self._disable_prior = False

        # 每个 token 一个 gate_logits，初值为 logit0
        self.gate_logits = nn.Parameter(torch.full((vocab_size,), logit0, dtype=torch.float32))

    @property
    def weight(self):
        # 为了保持 GPT2 的 tie_weights 正常工作
        # lm_head.weight <-> wte.weight
        # 确保在训练时，wte.weight 是可训练的
        return self.base.weight

    @contextmanager
    def prior_disabled(self):
        """临时关闭 prior（forward 只返回 base_emb）。"""
        old = self._disable_prior
        self._disable_prior = True
        try:
            yield
        finally:
            self._disable_prior = old

    def forward(self, input_ids: torch.LongTensor):
        # 前向传播：计算 E_eff = E_train + w(token) * E_prior
        # base_emb 为 GPT2 的 wte 层输出，即 E_train
        # prior_emb 为先验向量，根据 input_ids 索引取出
        # logits 为每个 token 的 gate_logits，根据 input_ids 索引取出
        # w 为每个 token 的门控权重，通过 sigmoid 映射到 [g_min, 1]
        # 返回 E_eff = E_train + w(token) * E_prior
        base_emb = self.base(input_ids)                        # [B, T, D]
        if self._disable_prior:
            return base_emb
        prior_emb = self.prior_matrix[input_ids]               # [B, T, D]
        logits = self.gate_logits[input_ids]                   # [B, T]
        w = self.g_min + (1.0 - self.g_min) * torch.sigmoid(logits)  # [B, T]
        return base_emb + w.unsqueeze(-1) * prior_emb


# PART3: 挂到 GPT2 模型上
def attach_gated_prior_to_gpt2(model, tokenizer, npz_path, g_min: float = 0.1, init_w: float = 0.5):
    vocab_size = model.config.vocab_size
    n_embd = model.config.n_embd

    prior_matrix, genus_token_ids, missing = build_prior_matrix_from_npz(
        tokenizer=tokenizer,
        npz_path=str(npz_path),
        vocab_size=vocab_size,
        n_embd=n_embd,
    )

    base_wte = model.transformer.wte
    prior_matrix = prior_matrix.to(dtype=base_wte.weight.dtype, device=base_wte.weight.device)

    model.transformer.wte = GatedPriorEmbedding(
        base=base_wte,
        prior_matrix=prior_matrix,
        g_min=g_min,
        init_w=init_w,
    )

    model.tie_weights()
    return genus_token_ids, missing


# PART4: 打印 gate 统计

def summarize_gate(model, topk: int = 10):
    wte = model.transformer.wte
    if not isinstance(wte, GatedPriorEmbedding):
        print("[gate] current wte is not GatedPriorEmbedding")
        return

    with torch.no_grad():
        w = wte.g_min + (1.0 - wte.g_min) * torch.sigmoid(wte.gate_logits)  # [V]
        prior_norm = wte.prior_matrix.norm(dim=-1)                          # [V]
        mask = prior_norm > 0                                               # genus token rows

        w_eff = w[mask]
        print(f"[gate] g_min={wte.g_min}, prior_nonzero_tokens={mask.sum().item()}")
        print(f"[gate] w mean={w_eff.mean().item():.4f}, std={w_eff.std().item():.4f}, min={w_eff.min().item():.4f}, max={w_eff.max().item():.4f}")

        vals, idxs = torch.topk(w_eff, k=min(topk, w_eff.numel()))
        token_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)[idxs]
        print("[gate] top using prior token_ids:", token_ids.tolist())
        print("[gate] top weights:", [float(x) for x in vals.cpu().tolist()])



def _ensure_labels(batch: dict, pad_token_id: int):
    # 确保 batch 里有 labels（Trainer eval / collator 有时会给，有时不会）。
    if "labels" in batch:
        return batch
    labels = batch["input_ids"].clone()
    if "attention_mask" in batch:
        labels[batch["attention_mask"] == 0] = -100
    else:
        labels[labels == pad_token_id] = -100
    batch["labels"] = labels
    return batch


def _cosine_rows(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    # 计算每行的 cosine 相似度：a,b: [N,D] -> 返回每行 cosine: [N]
    # 接近1说明向量方向相似，接近-1说明向量方向相反，接近0说明向量正交
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_n * b_n).sum(dim=-1)


class PriorDiagnosticsCallback(TrainerCallback):
    """
    每次 evaluate：
      1) 打印 gate 分布（只统计 prior 非零 token）
      2) 打印“抵消倾向”统计：delta 是否沿着 -prior 方向漂
      3) 在 eval 的第一个 batch 上做 quick ablation：开 prior vs 关 prior 的 loss 差
    """
    def __init__(self, tokenizer, topk: int = 10, do_ablation: bool = True):
        self.tokenizer = tokenizer
        self.topk = topk
        self.do_ablation = do_ablation

        self._base_init = None          # 训练开始时的 base.weight 快照（用于 delta）
        self._did_ablate = False        # 每次 eval 只做一次 ablation

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # 在训练一开始缓存 base.weight，用于后续比较 “漂移 delta”
        if model is None:
            return
        wte = model.transformer.wte
        if not isinstance(wte, GatedPriorEmbedding):
            return

        with torch.no_grad():
            self._base_init = wte.base.weight.detach().clone().to(torch.float32)  # [V,D]
        print("[PriorDiagnostics] cached initial base embedding for delta stats.")

    def _print_gate_stats(self, model):
        wte = model.transformer.wte
        with torch.no_grad():
            w = wte.g_min + (1.0 - wte.g_min) * torch.sigmoid(wte.gate_logits)  # [V]
            prior_norm = wte.prior_matrix.norm(dim=-1)                          # [V]
            mask = prior_norm > 0
            w_eff = w[mask]

            print(f"[gate] g_min={wte.g_min}, prior_nonzero_tokens={mask.sum().item()}")
            print(f"[gate] w mean={w_eff.mean().item():.4f}, std={w_eff.std().item():.4f}, min={w_eff.min().item():.4f}, max={w_eff.max().item():.4f}")

            vals, idxs = torch.topk(w_eff, k=min(self.topk, w_eff.numel()))
            token_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)[idxs]
            print("[gate] top token_ids:", token_ids.tolist())
            print("[gate] top weights:", [float(x) for x in vals.cpu().tolist()])

    def _print_cancellation_stats(self, model):
        """
        判断是否出现“抵消倾向”：
        看 base embedding 的变化 delta 是否更多沿着 -prior 方向（cos(delta, prior) 偏负）
        """
        if self._base_init is None:
            print("[cancel] base_init not cached; skip delta stats.")
            return

        wte = model.transformer.wte
        with torch.no_grad():
            base_now = wte.base.weight.detach().to(torch.float32)          # [V,D]
            prior = wte.prior_matrix.detach().to(torch.float32)            # [V,D]
            prior_norm = prior.norm(dim=-1)
            mask = prior_norm > 0

            delta = (base_now - self._base_init)[mask]                     # [N,D]
            prior_m = prior[mask]                                          # [N,D]
            base_m  = base_now[mask]                                       # [N,D]

            # 1) cos(delta, prior)：是否沿 -prior 漂（抵消倾向）
            cos_dp = _cosine_rows(delta, prior_m)
            # 2) cos(base, prior)：当前 base 本身是否已经“反向对齐 prior”
            cos_bp = _cosine_rows(base_m, prior_m)

            print(f"[cancel] cos(delta, prior): mean={cos_dp.mean().item():.4f}, std={cos_dp.std().item():.4f}, min={cos_dp.min().item():.4f}, max={cos_dp.max().item():.4f}")
            print(f"[cancel] cos(base,  prior): mean={cos_bp.mean().item():.4f}, std={cos_bp.std().item():.4f}, min={cos_bp.min().item():.4f}, max={cos_bp.max().item():.4f}")

            # 负相关比例（越高越像在往 -prior 走）
            frac_neg = (cos_dp < 0).float().mean().item()
            frac_strong = (cos_dp < -0.5).float().mean().item()
            print(f"[cancel] frac cos(delta,prior)<0: {frac_neg:.3f},  frac < -0.5: {frac_strong:.3f}")

            # 给一个更直观的“投影系数” a = <delta, prior> / ||prior||^2
            # 若 a 很负，表示 delta 在 prior 方向上有明显反向投影
            denom = (prior_m.norm(dim=-1) ** 2).clamp_min(1e-8)
            a = (delta * prior_m).sum(dim=-1) / denom
            print(f"[cancel] proj a= <delta,prior>/||prior||^2: mean={a.mean().item():.4f}, std={a.std().item():.4f}, min={a.min().item():.4f}, max={a.max().item():.4f}")

            # 打印最负的 topk（最像在往 -prior 抵消的 token）
            k = min(self.topk, a.numel())
            vals, idxs = torch.topk(-a, k=k)  # -a 最大 => a 最小
            # idxs 是 mask 压缩后的索引，需要映射回 vocab token_id
            token_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)[idxs]
            print("[cancel] most negative proj token_ids:", token_ids.tolist())
            print("[cancel] most negative proj a:", [float((-vals).cpu().tolist()[i]) for i in range(k)])

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        wte = model.transformer.wte
        if not isinstance(wte, GatedPriorEmbedding):
            return

        self._did_ablate = False  # 每次 eval 重置，只做一次 ablation

        print(f"\n[PriorDiagnostics] step={state.global_step}")
        self._print_gate_stats(model)
        self._print_cancellation_stats(model)

    def on_prediction_step(self, args, state, control, model=None, inputs=None, **kwargs):
        """
        eval 时每个 batch 都会触发；我们只在第一个 batch 做一次 ablation：
          loss_on  = 正常（开 prior）
          loss_off = with prior_disabled()（关 prior）
        """
        if (not self.do_ablation) or self._did_ablate:
            return
        if model is None or inputs is None:
            return

        wte = model.transformer.wte
        if not isinstance(wte, GatedPriorEmbedding):
            return

        # 只取第一个 batch 做 quick ablation，避免太慢
        self._did_ablate = True

        # inputs 里可能包含非 tensor（极少见），只保留 tensor 并搬到设备上
        batch = {k: v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}
        batch = _ensure_labels(batch, pad_token_id=self.tokenizer.pad_token_id)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            loss_on = model(**batch).loss.detach().float().cpu().item()
            with wte.prior_disabled():
                loss_off = model(**batch).loss.detach().float().cpu().item()

        if was_training:
            model.train()

        print(f"[ablation] loss_on(prior)= {loss_on:.4f} | loss_off(no_prior)= {loss_off:.4f} | Δ(off-on)= {(loss_off - loss_on):.4f}")



