import math
import torch
from torch.optim import AdamW
from transformers import Trainer
from transformers.optimization import get_scheduler
from transformers.trainer_callback import TrainerCallback


def freeze_wte_base(model, freeze: bool = True):
    """
    冻结/解冻输入 embedding 的 base 权重（注意：tied weights 时也等价冻结/解冻 lm_head.base）
    """
    wte = model.transformer.wte
    assert hasattr(wte, "base"), "wte 不是 GatedPriorEmbedding，找不到 .base"
    wte.base.weight.requires_grad = (not freeze)


def build_optimizer_no_filter(model, args):
    """
    关键：自己创建 optimizer，并且【不按 requires_grad 过滤】参数。
    否则你冻住 base 时，Trainer 可能直接把它从 optimizer 里剔除，后面解冻也更新不到。

    同时去重：tied weights 会导致同一个 Parameter 出现多次，必须用 id 去重。
    """
    no_decay_keys = ("bias", "ln_", "ln_f", "LayerNorm.weight", "layer_norm")

    decay_params = []
    nodecay_params = []
    seen = set()

    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))

        # 这里故意【不】检查 p.requires_grad
        if any(k in name for k in no_decay_keys):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    optim = AdamW(
        [
            {"params": decay_params,   "weight_decay": args.weight_decay, "lr": args.learning_rate},
            {"params": nodecay_params, "weight_decay": 0.0,              "lr": args.learning_rate},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )
    return optim


def compute_num_training_steps(trainer: Trainer) -> int:
    """
    用 Trainer 的 train dataloader 计算总步数（global_step 的上限）。
    """
    args = trainer.args
    if args.max_steps and args.max_steps > 0:
        return int(args.max_steps)

    dl = trainer.get_train_dataloader()
    steps_per_epoch = math.ceil(len(dl) / args.gradient_accumulation_steps)
    return int(steps_per_epoch * args.num_train_epochs)


class UnfreezeWteBaseAtStepCallback(TrainerCallback):
    """
    到某个 global_step 把 wte.base.weight 解冻。
    配合“自建 optimizer（不按 requires_grad 过滤）”才能真正更新到 base。
    """
    def __init__(self, unfreeze_step: int, verbose: bool = True):
        self.unfreeze_step = int(unfreeze_step)
        self.verbose = verbose
        self._done = False

    def on_step_begin(self, args, state, control, **kwargs):
        if self._done:
            return
        if state.global_step < self.unfreeze_step:
            return

        model = kwargs.get("model", None)
        if model is None:
            return

        wte = model.transformer.wte
        wte.base.weight.requires_grad = True
        self._done = True

        if self.verbose and getattr(state, "is_local_process_zero", True):
            print(f"\n[freeze] Unfroze wte.base.weight at global_step={state.global_step}\n")
