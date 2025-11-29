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

    # 先拿到 base embedding
    base_wte = model.transformer.wte

    # 在 CPU/float32 上做稳健的范数对齐
    with torch.no_grad():
        base_norm = base_wte.weight.detach().float().norm(dim=-1)   # [V]
        prior_norm = prior_matrix.detach().float().norm(dim=-1)     # [V]
        mask = prior_norm > 0

        if mask.any():
            base_eff = base_norm[mask]
            prior_eff = prior_norm[mask]

            def q(x, p):
                return float(torch.quantile(x, p).item())
            print(
                "[norm] base  p10/p50/p90 = "
                f"{q(base_eff,0.10):.4f} / {q(base_eff,0.50):.4f} / {q(base_eff,0.90):.4f}"
            )
            print(
                "[norm] prior p10/p50/p90 = "
                f"{q(prior_eff,0.10):.4f} / {q(prior_eff,0.50):.4f} / {q(prior_eff,0.90):.4f}"
            )
            s = torch.quantile(base_eff, 0.50) / torch.clamp_min(torch.quantile(prior_eff, 0.50), 1e-12)
            print(f"[norm] suggested prior_scale (p50 align) = {float(s.item()):.4f}")
            prior_matrix = prior_matrix * s
            print(f"[prior] applied global scale s={float(s.item()):.4f} to prior_matrix (p50 align)")

    # with torch.no_grad():
    #     base_norm = base_wte.weight.detach().float().norm(dim=-1)          # [V]
    #     prior_norm = prior_matrix.detach().float().norm(dim=-1)            # [V]
    #     mask = prior_norm > 0
    #     if mask.any():
    #         base_p50 = torch.quantile(base_norm[mask], 0.50)
    #         prior_p50 = torch.quantile(prior_norm[mask], 0.50).clamp_min(1e-12)
    #         s = base_p50 / prior_p50
    #         prior_matrix = prior_matrix * s
    #         print(f"[prior] applied global scale s={float(s.item()):.4f} to prior_matrix (p50 align)")

    # 再搬到模型设备与 dtype
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


import math
import torch
from transformers.trainer_callback import TrainerCallback


class PriorDiagnosticsCallback(TrainerCallback):
    """
    每次 evaluate：
      1) gate 分布统计（只统计 prior 非零 token）
      2) “抵消倾向”统计：delta 是否沿 -prior 漂
      3) eval 的第一个 batch 上做 quick ablation：开 prior vs 关 prior

    写入的 key 都以 eval_ 开头：
      - 会进入 Trainer 的 metrics（eval 输出）
      - 并且通过 trainer.log(...) 强制写入 log_history / CSV
    """
    def __init__(self, tokenizer, topk: int = 10, do_ablation: bool = True, trainer=None):
        self.tokenizer = tokenizer
        self.topk = int(topk)
        self.do_ablation = bool(do_ablation)

        # 用于强制写 log_history（如果不给 trainer，也可以只靠 metrics）
        self.trainer = trainer

        # 训练开始时记录 base embedding，用于后续 delta
        self._base_init = None  # [V,D] float32

        # ablation 控制：每轮 eval 只做一次
        self._did_ablate = False
        self._last_ablation = None  # {"loss_on":..., "loss_off":..., "delta":...}

        # 防止某些版本/多次回调导致重复写
        self._last_logged_step = None

    # ---------- helpers ----------
    @staticmethod
    def _safe_mean_std_min_max(x: torch.Tensor):
        if x is None or x.numel() == 0:
            nan = float("nan")
            return nan, nan, nan, nan
        x = x.float()
        mean = float(x.mean().item())
        std = float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0
        mn = float(x.min().item())
        mx = float(x.max().item())
        return mean, std, mn, mx

    @staticmethod
    def _safe_quantile(x: torch.Tensor, q: float):
        if x is None or x.numel() == 0:
            return float("nan")
        if x.numel() == 1:
            return float(x.item())
        return float(torch.quantile(x.float(), q).item())

    def _log(self, payload: dict):
        """把 payload 强制写进 log_history（并保证是 float / 可序列化）。"""
        if not payload:
            return
        # Trainer/JSON 对类型比较敏感：尽量都转 float
        safe = {}
        for k, v in payload.items():
            if isinstance(v, (int, float)):
                safe[k] = float(v)
            elif torch.is_tensor(v) and v.numel() == 1:
                safe[k] = float(v.item())
            else:
                # 复杂对象不写入 csv，避免报错
                continue

        if self.trainer is not None and len(safe) > 0:
            self.trainer.log(safe)

    # ---------- lifecycle ----------
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        wte = getattr(getattr(model, "transformer", None), "wte", None)
        if wte is None or (wte.__class__.__name__ != "GatedPriorEmbedding" and "GatedPriorEmbedding" not in str(type(wte))):
            return

        with torch.no_grad():
            self._base_init = wte.base.weight.detach().clone().to(torch.float32)  # [V,D]
        print("[PriorDiagnostics] cached initial base embedding for delta stats.")

    # ---------- metrics builders ----------
    def _gate_metrics(self, wte):
        with torch.no_grad():
            w = wte.g_min + (1.0 - wte.g_min) * torch.sigmoid(wte.gate_logits)     # [V]
            prior_norm = wte.prior_matrix.norm(dim=-1)                              # [V]
            mask = prior_norm > 0
            w_eff = w[mask]

            mean, std, mn, mx = self._safe_mean_std_min_max(w_eff)
            p10 = self._safe_quantile(w_eff, 0.10)
            p50 = self._safe_quantile(w_eff, 0.50)
            p90 = self._safe_quantile(w_eff, 0.90)

            # top1 / bottom1（只记录一个 id，避免 CSV 太乱）
            if w_eff.numel() > 0:
                top_val, top_idx = torch.max(w_eff, dim=0)
                bot_val, bot_idx = torch.min(w_eff, dim=0)
                token_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)

                top_token_id = int(token_ids[top_idx].item())
                bot_token_id = int(token_ids[bot_idx].item())
                top_w = float(top_val.item())
                bot_w = float(bot_val.item())
            else:
                top_token_id = -1
                bot_token_id = -1
                top_w = float("nan")
                bot_w = float("nan")

            return {
                "eval_gate_prior_tokens": float(mask.sum().item()),
                "eval_gate_g_min": float(wte.g_min),
                "eval_gate_mean": mean,
                "eval_gate_std": std,
                "eval_gate_min": mn,
                "eval_gate_max": mx,
                "eval_gate_p10": p10,
                "eval_gate_p50": p50,
                "eval_gate_p90": p90,
                "eval_gate_top1_id": float(top_token_id),
                "eval_gate_top1_w": top_w,
                "eval_gate_bot1_id": float(bot_token_id),
                "eval_gate_bot1_w": bot_w,
            }

    def _cancellation_metrics(self, wte):
        """
        抵消倾向：看 delta = base_now - base_init 是否更沿着 -prior。
        统计：
          cos(delta, prior) / cos(base, prior)
          frac_neg / frac_strong
          proj a = <delta, prior>/||prior||^2
        """
        if self._base_init is None:
            return {"eval_cancel_available": 0.0}

        with torch.no_grad():
            base_now = wte.base.weight.detach().to(torch.float32)            # [V,D]
            prior = wte.prior_matrix.detach().to(torch.float32)              # [V,D]
            prior_norm = prior.norm(dim=-1)
            mask = prior_norm > 0
            if mask.sum().item() == 0:
                return {"eval_cancel_available": 0.0}

            delta = (base_now - self._base_init)[mask]                       # [N,D]
            prior_m = prior[mask]                                            # [N,D]
            base_m  = base_now[mask]                                         # [N,D]

            cos_dp = _cosine_rows(delta, prior_m)                            # [N]
            cos_bp = _cosine_rows(base_m, prior_m)                           # [N]

            cosdp_mean, cosdp_std, cosdp_min, cosdp_max = self._safe_mean_std_min_max(cos_dp)
            cosbp_mean, cosbp_std, cosbp_min, cosbp_max = self._safe_mean_std_min_max(cos_bp)

            frac_neg = float((cos_dp < 0).float().mean().item())
            frac_strong = float((cos_dp < -0.5).float().mean().item())

            denom = (prior_m.norm(dim=-1) ** 2).clamp_min(1e-8)
            a = (delta * prior_m).sum(dim=-1) / denom                        # [N]

            a_mean, a_std, a_min, a_max = self._safe_mean_std_min_max(a)
            a_p10 = self._safe_quantile(a, 0.10)
            a_p50 = self._safe_quantile(a, 0.50)
            a_p90 = self._safe_quantile(a, 0.90)

            # 가장 negative a (最像抵消 prior 的 token id)
            min_val, min_idx = torch.min(a, dim=0)
            token_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            mostneg_id = int(token_ids[min_idx].item())

            return {
                "eval_cancel_available": 1.0,
                "eval_cancel_cos_dp_mean": cosdp_mean,
                "eval_cancel_cos_dp_std": cosdp_std,
                "eval_cancel_cos_dp_min": cosdp_min,
                "eval_cancel_cos_dp_max": cosdp_max,
                "eval_cancel_cos_bp_mean": cosbp_mean,
                "eval_cancel_cos_bp_std": cosbp_std,
                "eval_cancel_cos_bp_min": cosbp_min,
                "eval_cancel_cos_bp_max": cosbp_max,
                "eval_cancel_frac_neg": frac_neg,
                "eval_cancel_frac_strong": frac_strong,
                "eval_cancel_proj_a_mean": a_mean,
                "eval_cancel_proj_a_std": a_std,
                "eval_cancel_proj_a_min": a_min,
                "eval_cancel_proj_a_max": a_max,
                "eval_cancel_proj_a_p10": a_p10,
                "eval_cancel_proj_a_p50": a_p50,
                "eval_cancel_proj_a_p90": a_p90,
                "eval_cancel_mostneg_id": float(mostneg_id),
                "eval_cancel_mostneg_a": float(min_val.item()),
            }

    # ---------- printing ----------
    def _print_gate_stats(self, model):
        wte = getattr(getattr(model, "transformer", None), "wte", None)
        if wte is None:
            return
        if wte.__class__.__name__ != "GatedPriorEmbedding" and "GatedPriorEmbedding" not in str(type(wte)):
            return

        with torch.no_grad():
            w = wte.g_min + (1.0 - wte.g_min) * torch.sigmoid(wte.gate_logits)  # [V]
            prior_norm = wte.prior_matrix.norm(dim=-1)                          # [V]
            mask = prior_norm > 0
            w_eff = w[mask]

            print(f"[gate] g_min={wte.g_min}, prior_nonzero_tokens={mask.sum().item()}")
            if w_eff.numel() == 0:
                print("[gate] (no prior tokens)")
                return

            print(
                f"[gate] w mean={w_eff.mean().item():.4f}, std={w_eff.std(unbiased=False).item():.4f}, "
                f"min={w_eff.min().item():.4f}, max={w_eff.max().item():.4f}"
            )

            k = min(self.topk, w_eff.numel())
            vals, idxs = torch.topk(w_eff, k=k)
            token_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)[idxs]
            print("[gate] top token_ids:", token_ids.tolist())
            print("[gate] top weights:", [float(x) for x in vals.cpu().tolist()])

            vals2, idxs2 = torch.topk(-w_eff, k=k)
            token_ids2 = torch.nonzero(mask, as_tuple=False).squeeze(-1)[idxs2]
            print("[gate] bottom token_ids:", token_ids2.tolist())
            print("[gate] bottom weights:", [float((-x)) for x in vals2.cpu().tolist()])

    # ---------- main hooks ----------
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None:
            return
        wte = getattr(getattr(model, "transformer", None), "wte", None)
        if wte is None:
            return
        if wte.__class__.__name__ != "GatedPriorEmbedding" and "GatedPriorEmbedding" not in str(type(wte)):
            return

        # 每轮 eval 开始先重置，确保 ablation 本轮会做一次
        self._did_ablate = False
        self._last_ablation = None

        print(f"\n[PriorDiagnostics] step={state.global_step}")
        self._print_gate_stats(model)

        diag = {}
        diag.update(self._gate_metrics(wte))
        diag.update(self._cancellation_metrics(wte))

        # 注意：ablation 往往在 on_prediction_step 才会算到，所以这里先不写
        if metrics is not None:
            metrics.update(diag)
        self._log(diag)

        # 防止某些情况下同一步重复 log（可选）
        self._last_logged_step = state.global_step

    def on_prediction_step(self, args, state, control, model=None, inputs=None, **kwargs):
        """
        eval 时每个 batch 都会触发；只在第一个 batch 做一次 ablation：
          loss_on  = 正常（开 prior）
          loss_off = with prior_disabled()（关 prior）
        """
        if (not self.do_ablation) or self._did_ablate:
            return
        if model is None or inputs is None:
            return

        wte = getattr(getattr(model, "transformer", None), "wte", None)
        if wte is None:
            return
        if wte.__class__.__name__ != "GatedPriorEmbedding" and "GatedPriorEmbedding" not in str(type(wte)):
            return

        self._did_ablate = True

        batch = {k: v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}
        batch = _ensure_labels(batch, pad_token_id=self.tokenizer.pad_token_id)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            loss_on = float(model(**batch).loss.detach().float().cpu().item())
            with wte.prior_disabled():
                loss_off = float(model(**batch).loss.detach().float().cpu().item())

        if was_training:
            model.train()

        self._last_ablation = {"loss_on": loss_on, "loss_off": loss_off, "delta": (loss_off - loss_on)}

        print(
            f"[ablation] loss_on(prior)= {loss_on:.4f} | "
            f"loss_off(no_prior)= {loss_off:.4f} | Δ(off-on)= {(loss_off - loss_on):.4f}"
        )

        # ✅ 关键：ablation 结果算出来的当下就写入 log_history/CSV（最稳）
        ab = {
            "eval_ablation_loss_on": loss_on,
            "eval_ablation_loss_off": loss_off,
            "eval_ablation_delta_off_minus_on": (loss_off - loss_on),
        }
        self._log(ab)






# 查看 先验在输入层到底“有多大声” 的统计信息

def _norm_stats(x: torch.Tensor):
    x = x.float()
    return {
        "count": int(x.numel()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "min": float(x.min().item()),
        "p10": float(torch.quantile(x, 0.10).item()) if x.numel() > 1 else float(x.item()),
        "p50": float(torch.quantile(x, 0.50).item()) if x.numel() > 1 else float(x.item()),
        "p90": float(torch.quantile(x, 0.90).item()) if x.numel() > 1 else float(x.item()),
        "max": float(x.max().item()),
    }

@torch.no_grad()

def check_prior_vs_base_norms(model, only_prior_nonzero: bool = True):
    """
    对比：
      - base: model.transformer.wte.base.weight 的行范数
      - prior: model.transformer.wte.prior_matrix 的行范数（默认只看非零行）
    并给出一个建议的全局缩放因子 scale（用 p50 对齐）。
    要求 wte 是 GatedPriorEmbedding。
    """
    wte = model.transformer.wte
    if not hasattr(wte, "prior_matrix") or not hasattr(wte, "base"):
        raise ValueError("model.transformer.wte is not GatedPriorEmbedding-like.")

    base_w = wte.base.weight.detach()
    prior = wte.prior_matrix.detach()

    base_norm = base_w.norm(dim=-1)
    prior_norm = prior.norm(dim=-1)

    if only_prior_nonzero:
        mask = prior_norm > 0
        prior_norm_eff = prior_norm[mask]
        base_norm_eff = base_norm[mask]
    else:
        prior_norm_eff = prior_norm
        base_norm_eff = base_norm

    base_s = _norm_stats(base_norm_eff)
    prior_s = _norm_stats(prior_norm_eff)

    # 用 p50 对齐的 scale 建议（稳健）
    scale = base_s["p50"] / max(prior_s["p50"], 1e-12)

    print("[norm] base  stats:", base_s)
    print("[norm] prior stats:", prior_s)
    print(f"[norm] suggested prior_scale (p50 align) = {scale:.4f}")

    return {"base": base_s, "prior": prior_s, "suggested_scale": float(scale)}
