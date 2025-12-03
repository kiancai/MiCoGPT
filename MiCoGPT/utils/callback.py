import torch
from transformers.trainer_callback import TrainerCallback

class PriorGateStatsOnEvalLogCallback(TrainerCallback):
    """
    只在 eval 产生 log（logs 里包含 'eval_loss'）时计算并把指标塞进 logs，
    从而保证 notebook 表格每次验证都会多出这些列。
    """

    def __init__(self, token_ids=None, prefix="gp"):
        self.prefix = prefix
        self._ids = None
        if token_ids is not None:
            self._ids = torch.tensor(list(token_ids), dtype=torch.long)

    @staticmethod
    def _q(x: torch.Tensor, q: float) -> float:
        return torch.quantile(x, q).item()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # 只在验证日志那一行插入（避免训练loss每步都算）
        if "eval_loss" not in logs:
            return
        if not getattr(state, "is_local_process_zero", True):
            return

        model = kwargs.get("model", None)
        if model is None:
            return

        transformer = getattr(model, "transformer", None)
        if transformer is None:
            return
        wte = getattr(transformer, "wte", None)
        if wte is None:
            return
        if (not hasattr(wte, "prior_matrix")) or (not hasattr(wte, "gate_logits")) or (not hasattr(wte, "base")):
            return
        p = self.prefix

        ########################
        # 1) base 是否解冻
        logs[f"{p}_base_req"] = float(wte.base.weight.requires_grad)
        with torch.no_grad():
            base_w = wte.base.weight.detach()       # [V, D]
            # 2) base 是否真的在更新（相邻两次 eval 的变化）
            if not hasattr(self, "_prev_base_w"):
                self._prev_base_w = None
            if self._prev_base_w is None:
                logs[f"{p}_bdelta50"] = 0.0
                logs[f"{p}_bdmax"] = 0.0
            else:
                if self._ids is not None:
                    ids = self._ids.to(device=base_w.device)
                    cur = base_w.index_select(0, ids).float()
                    prv = self._prev_base_w.index_select(0, ids).float()
                else:
                    cur = base_w.float()
                    prv = self._prev_base_w.float()

                delta = (cur - prv).abs()
                logs[f"{p}_bdelta50"] = torch.quantile(delta, 0.50).item()
                logs[f"{p}_bdmax"] = delta.max().item()
            self._prev_base_w = base_w.detach().clone()
        ########################
        
        with torch.no_grad():
            base_w = wte.base.weight.detach()       # [V, D]
            prior  = wte.prior_matrix.detach()      # [V, D]
            gate_logits = wte.gate_logits.detach()
            g_min = float(getattr(wte, "g_min", 0.0))
            eps = 1e-12

            # w：标量门控 [V] 或向量门控 [V,D]
            if gate_logits.dim() == 1:
                w_vocab = g_min + (1.0 - g_min) * torch.sigmoid(gate_logits)        # [V]
                w_token_scalar = w_vocab                                            # [V]
                gated_prior = prior * w_vocab.unsqueeze(-1)                         # [V,D]
            elif gate_logits.dim() == 2:
                w_vocab = g_min + (1.0 - g_min) * torch.sigmoid(gate_logits)        # [V,D]
                w_token_scalar = w_vocab.mean(dim=-1)                               # [V]（统计用）
                gated_prior = prior * w_vocab                                       # [V,D]
            else:
                return

            # 选统计子集：优先用 genus_token_ids；否则用 prior_norm>0
            if self._ids is not None:
                ids = self._ids.to(device=base_w.device)
                base_sel  = base_w.index_select(0, ids)
                prior_sel = prior.index_select(0, ids)
                gp_sel    = gated_prior.index_select(0, ids)
                w_sel     = w_token_scalar.index_select(0, ids)
            else:
                mask = (prior.norm(dim=-1) > 0)
                if mask.any():
                    base_sel, prior_sel, gp_sel, w_sel = base_w[mask], prior[mask], gated_prior[mask], w_token_scalar[mask]
                else:
                    base_sel, prior_sel, gp_sel, w_sel = base_w, prior, gated_prior, w_token_scalar

            # 统计用 float32
            base_sel  = base_sel.float()
            prior_sel = prior_sel.float()
            gp_sel    = gp_sel.float()
            w_sel     = w_sel.float()

            base_norm  = base_sel.norm(dim=-1)
            prior_norm = prior_sel.norm(dim=-1)
            gp_norm    = gp_sel.norm(dim=-1)
            ratio      = gp_norm / (base_norm + eps)

            cos = (base_sel * gp_sel).sum(dim=-1) / ((base_norm + eps) * (gp_norm + eps))
            cos = cos.clamp(-1.0, 1.0)

            p = self.prefix
            logs[f"{p}_w50"]   = self._q(w_sel, 0.50)
            logs[f"{p}_w90"]   = self._q(w_sel, 0.90)
            logs[f"{p}_wm"]    = w_sel.mean().item()

            logs[f"{p}_bn50"]  = self._q(base_norm, 0.50)
            logs[f"{p}_pn50"]  = self._q(prior_norm, 0.50)
            logs[f"{p}_gn50"]  = self._q(gp_norm, 0.50)
            logs[f"{p}_r50"]   = self._q(ratio, 0.50)
            logs[f"{p}_cos50"] = self._q(cos, 0.50)
