import torch
from torch.nn.utils.rnn import pad_sequence

class MiCoGPTDataCollator:
    """
    MiCoGPT v2.0 自定义数据整理器 (Data Collator)
    
    主要功能：
    1. 接收 dataset __getitem__ 返回的样本列表。
    2. 将 input_ids (物种Token序列) 进行 padding 对齐。
    3. 将 value_ids (丰度等级序列) 进行 padding 对齐。
    4. 将 condition_ids (环境元数据) 堆叠成 Tensor。
    5. 生成 attention_mask，标记 padding 位置。
    6. 生成 labels，用于计算 Loss (通常 labels = input_ids，padding 处设为 -100)。
    
    Args:
        tokenizer: 分词器，用于获取 pad_token_id
        max_length: 最大序列长度 (可选)
    """
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 获取 padding token id，如果未设置则默认为 0
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, examples):
        """
        Args:
            examples: list of dict, 每个 dict 包含:
                - input_ids: list[int]
                - value_ids: list[int]
                - condition_ids: list[int]
                - length: int
        
        Returns:
            batch: dict of tensors
        """
        # 1. 提取各个字段 (避免不必要的 Tensor 转换)
        # dataset __getitem__ 已经返回了 Tensor (clone 过的)，这里直接用
        # 如果不是 Tensor (例如 raw list)，再转
        def ensure_tensor(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)

        input_ids_list = [ensure_tensor(e["input_ids"]) for e in examples]
        value_ids_list = [ensure_tensor(e["value_ids"]) for e in examples]
        condition_ids_list = [ensure_tensor(e["condition_ids"]) for e in examples]
        
        # 2. Padding 对齐序列 (batch_first=True)
        # input_ids 使用 pad_token_id 填充
        if self.max_length:
            # 静态 Padding: 强制填充/截断到 max_length
            def pad_and_truncate(t, max_len, pad_val):
                cur_len = t.size(0)
                if cur_len == max_len:
                    return t
                elif cur_len > max_len:
                    return t[:max_len]
                else:
                    padding = torch.full((max_len - cur_len,), pad_val, dtype=t.dtype, device=t.device)
                    return torch.cat([t, padding])

            # 优化：如果所有 Tensor 长度都已经是 max_length，直接 stack (极快)
            if all(t.size(0) == self.max_length for t in input_ids_list):
                input_ids = torch.stack(input_ids_list)
                value_ids = torch.stack(value_ids_list)
            else:
                input_ids = torch.stack([pad_and_truncate(t, self.max_length, self.pad_token_id) for t in input_ids_list])
                value_ids = torch.stack([pad_and_truncate(t, self.max_length, 0) for t in value_ids_list])
        else:
            # 动态 Padding: 填充到当前 Batch 的最大长度
            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
            value_ids = pad_sequence(value_ids_list, batch_first=True, padding_value=0)
        
        # 3. 堆叠环境信息 (Metadata)
        # condition_ids 是定长的 (num_conditions)，直接 stack
        condition_ids = torch.stack(condition_ids_list)
        
        # 4. 生成 Attention Mask
        # 非 padding 部分为 1，padding 部分为 0
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # 5. 生成 Labels (用于 Causal Language Modeling)
        # 通常 labels 就是 input_ids，但是 padding 部分需要设为 -100 (PyTorch Loss 默认忽略 -100)
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        
        # 6. [Optimization] 动态裁剪 (已禁用)
        # 为了避免 Trainer 在 Evaluation 时因 Batch shape 不一致导致 RuntimeError，
        # 我们暂时禁用动态裁剪，保持 input_ids 为固定的 max_length (如果设定了)。
        pass
            
        return {
            "input_ids": input_ids,
            "value_ids": value_ids,
            "condition_ids": condition_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class MiCoGPTClassificationCollator(MiCoGPTDataCollator):
    """
    MiCoGPT v2.0 序列分类数据整理器
    
    继承自 MiCoGPTDataCollator，但处理 labels 的方式不同：
    - CLM (Pretraining): labels = input_ids (shifted)
    - Classification: labels = target class (0/1/...)
    """
    def __call__(self, examples):
        # 1. 提取并移除 examples 中的 labels (如果有)
        # 我们的 SubsetWithLabels 会在 __getitem__ 中添加 "labels" 字段
        classification_labels = None
        # 检查第一个样本是否有 labels 字段
        if examples and "labels" in examples[0]:
            # 注意：这里我们通过 pop 将 labels 从 input examples 中移除
            # 这样父类 MiCoGPTDataCollator 就不会受到干扰 (虽然父类目前也不读取 labels)
            classification_labels = [e.pop("labels") for e in examples]
            classification_labels = torch.tensor(classification_labels, dtype=torch.long)
            
        # 2. 调用父类处理 input_ids, value_ids, condition_ids
        # 父类会生成 input_ids, value_ids, condition_ids, attention_mask, 以及 CLM 用的 labels
        batch = super().__call__(examples)
        
        # 3. 覆盖 labels
        if classification_labels is not None:
            # 将分类标签放入 batch，覆盖掉父类生成的 CLM labels
            batch["labels"] = classification_labels
        else:
            # 如果没有提供分类标签 (例如预测阶段)，则删除父类生成的 CLM labels
            # 以免模型 forward 时误计算 CLM Loss
            if "labels" in batch:
                del batch["labels"]
                
        return batch
