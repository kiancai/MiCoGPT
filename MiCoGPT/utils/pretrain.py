import math
import torch
import numpy as np
import torch.nn as nn

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



# PART2: Embedding Wrapper
class GatedPriorEmbedding(nn.Module):
    # 自定义了一个 GatedPriorEmbedding 类，用于替换 GPT2 模型的 embedding 层
    # E_eff = E_train + w(token) * E_prior
    # w(token) = g_min + (1-g_min)*sigmoid(gate_logits[token])

    def __init__(self, base: nn.Embedding, prior_matrix: torch.Tensor, g_min: float = 0.0, init_w: float = 0.1):
        super().__init__()

        # 简单参数检查
        if not (0.0 <= g_min < 1.0 and 0.0 < init_w < 1.0 and init_w >= g_min):
            raise ValueError(f"Invalid parameters: g_min、init_w")

        # base 为 GPT2 的 wte 层
        self.base = base
        # 先验向量，固定不训练
        self.register_buffer("prior_matrix", prior_matrix)
        self.g_min = float(g_min)

        vocab_size = prior_matrix.shape[0]

        # 让 w 的初值约等于 init_w
        # 将 init_w 映射回 gate_logits 的初值：logit( (init_w-g_min)/(1-g_min) )
        inner = (init_w - self.g_min) / max(1e-6, (1.0 - self.g_min))
        inner = min(max(inner, 1e-6), 1 - 1e-6)  # 防止出现 log(0) 或 log(∞)
        logit0 = math.log(inner / (1.0 - inner))

        # 每个 token 一个 gate_logits，初值为 logit0
        self.gate_logits = nn.Parameter(torch.full((vocab_size,), logit0, dtype=torch.float32))

    @property
    def weight(self):
        # 为了保持 GPT2 的 tie_weights 正常工作
        return self.base.weight

    def forward(self, input_ids: torch.LongTensor):
        # 前向传播：计算 E_eff = E_train + w(token) * E_prior
        # base_emb 为 GPT2 的 wte 层输出，即 E_train
        # prior_emb 为先验向量，根据 input_ids 索引取出
        # logits 为每个 token 的 gate_logits，根据 input_ids 索引取出
        # w 为每个 token 的门控权重，通过 sigmoid 映射到 [g_min, 1]
        # 返回 E_eff = E_train + w(token) * E_prior
        base_emb = self.base(input_ids)                        # [B, T, D]
        prior_emb = self.prior_matrix[input_ids]               # [B, T, D]
        w = self.g_min + (1.0 - self.g_min) * torch.sigmoid(self.gate_logits[input_ids])  # [B, T]
        return base_emb + w.unsqueeze(-1) * prior_emb



# PART3: 挂到 GPT2 模型上
import torch

def attach_gated_prior_to_gpt2(
    model, tokenizer, npz_path,
    g_min: float = 0.0, init_w: float = 0.1,
    shuffle_prior: bool = False, shuffle_seed: int = 42,
    prior_scale=None,   # None/"p50":自动对齐；float:手动缩放；1.0:不缩放
):
    vocab_size = model.config.vocab_size
    n_embd = model.config.n_embd

    prior_matrix, genus_token_ids, missing = build_prior_matrix_from_npz(
        tokenizer=tokenizer,
        npz_path=str(npz_path),
        vocab_size=vocab_size,
        n_embd=n_embd,
    )

    # 随机打乱先验向量
    if shuffle_prior and len(genus_token_ids) > 1:
        g = torch.Generator(device="cpu")
        g.manual_seed(shuffle_seed)
        ids = torch.tensor(genus_token_ids, dtype=torch.long)
        perm = torch.randperm(ids.numel(), generator=g)
        prior_matrix[ids] = prior_matrix[ids[perm]]
        print(f"[prior] shuffled prior vectors among {len(genus_token_ids)} genus tokens (seed={shuffle_seed})")

    # 先拿到 base embedding
    base_wte = model.transformer.wte

    # ====== 对齐 / 手动缩放 prior ======
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
                s = torch.quantile(base_norm[mask], 0.50) / torch.clamp_min(torch.quantile(prior_norm[mask], 0.50), 1e-12)
                prior_matrix = prior_matrix * s
                print(f"[prior] applied AUTO scale s={float(s.item()):.4f} to prior_matrix (p50 align)")
            else:
                print("[prior] no nonzero prior rows found, skip scaling")

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



