from transformers import GPT2LMHeadModel, GPT2PreTrainedModel, GPT2Model, GPT2Config, GPT2ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
import numpy as np

class MiCoGPTConfig(GPT2Config):
    def __init__(
        self,
        num_bins=51,
        condition_vocab_sizes=None,
        prior_matrix_path=None,
        **kwargs
    ):
        """
        Args:
            num_bins: 丰度分箱数 (默认 51)
            condition_vocab_sizes: List[int], 每个环境因子的类别数
            prior_matrix_path: str, 先验矩阵文件路径
        """
        super().__init__(**kwargs)
        self.num_bins = num_bins
        self.condition_vocab_sizes = condition_vocab_sizes if condition_vocab_sizes is not None else []
        # 确保 prior_matrix_path 是字符串 (JSON serializable)
        self.prior_matrix_path = str(prior_matrix_path) if prior_matrix_path is not None else None
        
        # [vCross] Cross-Attention 默认为 True，但允许用户覆盖
        if "add_cross_attention" not in kwargs:
            self.add_cross_attention = True
        else:
            self.add_cross_attention = kwargs["add_cross_attention"]


class MiCoGPTForCausalLM(GPT2LMHeadModel):
    """
    MiCoGPT v2.0 模型主体 (vCross 版)
    继承自 GPT2LMHeadModel，增加了多模态 Embedding 融合逻辑。
    """
    def __init__(self, config):
        super().__init__(config)
        
        # --- 多模态 Embedding 层 ---
        # 1. 丰度 Value Embedding
        # input_ids 对应 Species Embedding (GPT2 原有的 wte)
        # value_ids 对应 Value Embedding
        self.value_embedding = nn.Embedding(config.num_bins, config.n_embd)
        
        # 2. 环境 Metadata Embeddings (支持多个环境因子)
        # condition_ids[:, i] 对应第 i 个环境因子
        self.condition_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, config.n_embd) 
            for vocab_size in config.condition_vocab_sizes
        ])
        
        # 3. 先验知识矩阵 (用于 Cross-Attention)
        # 初始化为一个很小的占位符，后续通过 set_prior_matrix 加载
        self.register_buffer("prior_matrix", torch.zeros(1, config.n_embd))
        
        # 初始化权重 (不仅是 head，也包括新增的 embedding)
        self.post_init()

    def set_prior_matrix(self, matrix_tensor):
        """加载预训练好的先验矩阵 (例如 Genus Embeddings)"""
        self.register_buffer("prior_matrix", matrix_tensor)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        # --- v2.0 新增输入 ---
        value_ids=None,
        condition_ids=None,
    ):
        """
        重写 forward 函数，实现多模态融合和先验注入
        """
        
        # 1. 构建 Input Embeddings (如果未提供 inputs_embeds)
        if inputs_embeds is None:
            # (1) 基础物种 Embedding (来自 GPT2 原有的 wte)
            inputs_embeds = self.transformer.wte(input_ids)
            
            # (2) 加上丰度 Value Embedding
            if value_ids is not None:
                value_embeds = self.value_embedding(value_ids)
                inputs_embeds = inputs_embeds + value_embeds
            
            # (3) 加上环境 Metadata Embedding
            # condition_ids shape: [Batch, Num_Conditions]
            # 我们将其广播加到序列的每个 Token 上
            if condition_ids is not None:
                for i, emb_layer in enumerate(self.condition_embeddings):
                    # 获取第 i 个环境因子的 Embedding: [Batch, Dim]
                    cond_vec = emb_layer(condition_ids[:, i])
                    # 广播到序列长度: [Batch, 1, Dim] -> 自动加到 [Batch, Seq_Len, Dim]
                    inputs_embeds = inputs_embeds + cond_vec.unsqueeze(1)
        
        # 2. 准备 Cross-Attention 的 Key/Value (即 Prior Matrix)
        # encoder_hidden_states 在 GPT2 (add_cross_attention=True) 中充当 Cross-Attention 的记忆库
        # 我们将全局的 prior_matrix 扩展到当前 Batch 大小
        # prior_matrix shape: [Num_Priors, Dim]
        # target shape: [Batch, Num_Priors, Dim]
        if encoder_hidden_states is None and self.prior_matrix.shape[0] > 1:
            batch_size = inputs_embeds.shape[0]
            # [Fix] 使用 repeat 而不是 expand，确保 tensor 在内存中是连续的 (contiguous)。
            # Transformers 的 Conv1D 层内部使用了 .view()，如果输入 tensor 是 expand 出来的（stride=0），会报错：
            # RuntimeError: view size is not compatible with input tensor's size and stride
            encoder_hidden_states = self.prior_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
            
        # 3. 调用父类 (GPT2LMHeadModel) 的 forward
        return super().forward(
            input_ids=None, # 我们已经计算了 inputs_embeds，所以这里传 None
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds, # 传入融合后的 Embedding
            encoder_hidden_states=encoder_hidden_states, # 传入先验库
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class MiCoGPTForSequenceClassification(GPT2ForSequenceClassification):
    """
    MiCoGPT v2.0 序列分类模型 (vCross 版)
    继承自 GPT2ForSequenceClassification，增加了多模态 Embedding 融合逻辑。
    """
    def __init__(self, config):
        super().__init__(config)
        
        # --- 多模态 Embedding 层 (与 MiCoGPTForCausalLM 保持一致) ---
        # 1. 丰度 Value Embedding
        self.value_embedding = nn.Embedding(config.num_bins, config.n_embd)
        
        # 2. 环境 Metadata Embeddings (支持多个环境因子)
        self.condition_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, config.n_embd) 
            for vocab_size in config.condition_vocab_sizes
        ])
        
        # 3. 先验知识矩阵 (用于 Cross-Attention)
        # 初始化为一个很小的占位符，后续通过 set_prior_matrix 加载
        self.register_buffer("prior_matrix", torch.zeros(1, config.n_embd))
        
        # 初始化权重 (不仅是 head，也包括新增的 embedding)
        self.post_init()

    def set_prior_matrix(self, matrix_tensor):
        """加载预训练好的先验矩阵 (例如 Genus Embeddings)"""
        self.register_buffer("prior_matrix", matrix_tensor)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        # --- v2.0 新增输入 ---
        value_ids=None,
        condition_ids=None,
    ):
        # 1. 构建 Input Embeddings (逻辑同 MiCoGPTForCausalLM)
        if inputs_embeds is None:
            # (1) 基础物种 Embedding
            inputs_embeds = self.transformer.wte(input_ids)
            
            # (2) 加上丰度 Value Embedding
            if value_ids is not None:
                value_embeds = self.value_embedding(value_ids)
                inputs_embeds = inputs_embeds + value_embeds
            
            # (3) 加上环境 Metadata Embedding
            if condition_ids is not None:
                for i, emb_layer in enumerate(self.condition_embeddings):
                    cond_vec = emb_layer(condition_ids[:, i])
                    inputs_embeds = inputs_embeds + cond_vec.unsqueeze(1)
        
        # 2. 准备 Cross-Attention 的 Key/Value
        if encoder_hidden_states is None and self.prior_matrix.shape[0] > 1:
            batch_size = inputs_embeds.shape[0]
            encoder_hidden_states = self.prior_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
            
        # 3. 手动调用 Transformer (GPT2Model)
        # 无法直接调用 super().forward()，因为它不支持 encoder_hidden_states
        # [Fix] 显式设置 use_cache=False，节省显存 (分类任务不需要 cache)
        # [Fix] 显式关闭 output_hidden_states/attentions，防止 Trainer 意外收集导致 OOM
        transformer_outputs = self.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                # 当 input_ids 为 None (只有 inputs_embeds) 时，我们无法直接通过 token id 判断 pad
                # 但在 MiCoGPT 中，我们通常不手动传入 inputs_embeds 进行分类微调，除非是在 forward 内部生成
                # 在上面的代码中，input_ids 确实传给了 inputs_embeds 生成逻辑，但 self.transformer 传的是 input_ids=None
                # 所以这里我们需要用原始的 input_ids 来计算 sequence_lengths
                # 注意：函数的输入参数里有 input_ids
                if input_ids is not None:
                     sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                     sequence_lengths = sequence_lengths % input_ids.shape[-1]
                     sequence_lengths = sequence_lengths.to(logits.device)
                else:
                    sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=None,
            attentions=None,
        )
