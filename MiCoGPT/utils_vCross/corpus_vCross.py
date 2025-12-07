import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from importlib.resources import files
from MiCoGPT.utils.tools import extract_taxon
from sklearn.preprocessing import LabelEncoder
import joblib

class MiCoGPTCorpusVCross(Dataset):
    def __init__(self, 
                 tokenizer,                # PreTrainedTokenizer
                 data_path: str,
                 metadata: pd.DataFrame | None = None,  # 样本的 metadata
                 key: str = "genus",
                 max_len: int = 512,
                 return_metadata: bool = False,
                 # [vCross] 参数: 指定需要注入的 metadata 列
                 use_meta_cols: list[str] | None = None,
                 # [vCross] 参数: 分箱数 (默认 51, scGPT 标准)
                 num_bins: int = 51,
                 # [vCross] 新增参数: 是否使用 phylogeny 过滤 OTU
                 phylogeny_path: str | None = None,
                 # [vCross] 参数: 随机种子 (用于 binning 的随机扰动)
                 seed: int = 42,
                 # [vCross] 参数: 是否进行 Log1p 转换 (scGPT 默认 True)
                 log1p: bool = True,
                 # [vCross] 参数: 归一化目标总数
                 # 如果为 None，则自动使用所有样本 Library Size 的中位数 (推荐)
                 # scGPT 默认 1e4，但在微生物组中建议自适应
                 normalize_total: float | None = None,
                 ):

        # 注意，这里读取丰度表后进行了转置，使得样本在行，OTU 在列
        self.data = pd.read_csv(data_path, sep=',', index_col=0).T
        self.tokenizer = tokenizer
        self.seed = seed
        self.log1p = log1p
        self.normalize_total = normalize_total
        
        # [vCross] Phylogeny 变为可选，仅用于过滤 OTU
        if phylogeny_path is not None:
            self.phylogeny = pd.read_csv(phylogeny_path, index_col=0)
        else:
            self.phylogeny = None

        self.max_len = max_len
        self.key = key
        self.return_metadata = return_metadata
        self.use_meta_cols = use_meta_cols if use_meta_cols is not None else []
        self.num_bins = num_bins

        # 预处理：合并 genus，保留相对丰度
        # 注意：这里我们不再做 z-score，因为我们要用相对丰度做 Ranking 和 Binning
        self.data = self._preprocess(self.data)

        # 记录样本顺序
        self.sample_ids = list(self.data.index)

        # 处理 Metadata
        if metadata is not None:
            if not isinstance(metadata, pd.DataFrame):
                raise TypeError("metadata 必须是 pandas.DataFrame")
            if not metadata.index.is_unique:
                raise ValueError("metadata.index (sample_id) 必须唯一")
            try:
                self.metadata = metadata.loc[self.sample_ids].copy()
            except KeyError as e:
                raise KeyError(
                    "metadata 中缺少某些样本的记录，请检查 metadata.index 是否包含所有样本。\n"
                    f"缺失信息：{e}"
                )
        else:
            self.metadata = None

        if self.use_meta_cols and self.metadata is None:
            raise ValueError("指定了 use_meta_cols，但没有提供 metadata (或 metadata 为 None)。")

        # [vCross] 构建 Label Encoders
        self.meta_encoders = {}
        if self.use_meta_cols:
            self._build_meta_encoders()

        self.input_ids_list = []
        self.value_ids_list = []
        self.condition_ids_list = []
        self.length_list = []
        
        # 逐样本处理
        print("Converting samples to tokens (vCross)...")
        for sample_id in tqdm(self.data.index):
            # 获取该样本的丰度数据 (Series)
            sample_abundance = self.data.loc[sample_id]
            
            # 核心转换函数
            input_ids, value_ids, condition_ids, length = self._convert_to_token(
                sample_abundance, sample_id
            )
            
            self.input_ids_list.append(input_ids)
            self.value_ids_list.append(value_ids)
            self.condition_ids_list.append(condition_ids)
            self.length_list.append(length)
            
        print(f'Total {len(self.input_ids_list)} samples.\n\
            Max length is {max(self.length_list)}.\n\
            Average length is {np.mean(self.length_list)}.\n\
            Min length is {min(self.length_list)}.')
            
        self.input_ids = torch.LongTensor(self.input_ids_list)
        self.value_ids = torch.LongTensor(self.value_ids_list)
        # condition_ids 是样本级的，不需要 padding，所以可以直接转
        self.condition_ids = torch.LongTensor(self.condition_ids_list)

    def _build_meta_encoders(self):
        """
        为每个选定的 Metadata 列构建 LabelEncoder。
        处理空值：空值会被填充为 "Unknown_{col}" 并作为一个类别。
        """
        print("[vCross] Building Metadata Encoders...")
        for col in self.use_meta_cols:
            le = LabelEncoder()
            # 填充空值，统一转为字符串
            # [vCross] 修改：为了区分不同列的 Unknown，加上列名后缀
            # 虽然 LabelEncoder 是独立的，但这样在查看 labels 时更清晰
            unknown_label = f"Unknown_{col}"
            col_values = self.metadata[col].fillna(unknown_label).astype(str).values
            
            # 还要处理可能是 "nan" 字符串的情况
            col_values = [v if v.lower() != 'nan' else unknown_label for v in col_values]
            
            le.fit(col_values)
            self.meta_encoders[col] = le
            print(f"  -> Encoder for '{col}': {len(le.classes_)} classes")
            
    def save_encoders(self, path):
        """保存 Encoders 以便推理时使用"""
        joblib.dump(self.meta_encoders, path)

    def subset_by_metadata(self, condition):
        """
        根据 metadata 条件划分子集
        Args:
            condition: 一个函数，接收 metadata DataFrame，返回布尔 Series/Array
        Returns:
            torch.utils.data.Subset
        """
        if self.metadata is None:
            raise ValueError("当前 corpus 没有关联 metadata，无法按 metadata 划分子集。")
        
        # 1. 生成布尔 mask
        # condition(self.metadata) 应该返回一个 boolean Series，index 与 metadata 一致
        try:
            mask = condition(self.metadata)
        except Exception as e:
            raise ValueError(f"应用筛选条件失败: {e}")

        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != len(self):
            # 有可能 metadata 长度和 corpus 不一致 (理论上 init 里检查过)
            # 或者 condition 返回的长度不对
            raise ValueError(
                f"condition 返回的 mask 长度为 {mask.shape[0]}，"
                f"但 corpus 样本数为 {len(self)}。"
            )
            
        # 2. 将 True 的位置转成 indices
        indices = np.where(mask)[0].tolist()
        
        # 3. 返回 PyTorch 自带的 Subset
        print(f"[subset_by_metadata] Selected {len(indices)} samples out of {len(self)}.")
        return Subset(self, indices)

    def subset_by_ids(self, ids):
        """
        根据 sample_id 列表划分子集
        """
        # 构建 sample_id -> index 的映射
        id2idx = {sid: i for i, sid in enumerate(self.sample_ids)}

        indices = []
        missing = []
        for sid in ids:
            if sid in id2idx:
                indices.append(id2idx[sid])
            else:
                missing.append(sid)

        if len(missing) > 0:
            print(f"[subset_by_ids] 警告：以下 sample_id 未在 corpus 中找到，将被忽略：{len(missing)} 个")

        print(f"[subset_by_ids] Selected {len(indices)} samples.")
        return Subset(self, indices)

    def __len__(self):
        return len(self.input_ids)        

    def __getitem__(self, index):
        # 1. Input IDs Mask
        attention_mask = torch.ones(self.input_ids[index].shape)
        attention_mask[self.input_ids[index] == self.tokenizer.pad_token_id] = 0
        
        # 2. Clone Tensors
        input_ids = self.input_ids[index].clone()
        value_ids = self.value_ids[index].clone()
        condition_ids = self.condition_ids[index].clone()

        return {
            'input_ids': input_ids,
            'value_ids': value_ids,
            'condition_ids': condition_ids,
            'attention_mask': attention_mask
        }

    def _digitize(self, x: np.ndarray, bins: np.ndarray, seed_modifier: int = 0) -> np.ndarray:
        """
        仿照 scGPT 的 _digitize 实现。
        当数值落在 bin edges 上（即 duplicates）时，通过随机扰动将其均匀分散到相邻 bin 中。
        
        Args:
            x: 数据数组 (values)
            bins: 分位点数组
            seed_modifier: 用于调整随机种子的修饰符 (通常使用 sample_id 的 hash)
        """
        # 确保使用确定性的随机数生成器
        # 组合 base seed 和 sample 特异性 modifier
        # 注意：seed 必须是 uint32
        full_seed = (self.seed + seed_modifier) % (2**32)
        rng = np.random.default_rng(full_seed)

        left_digits = np.digitize(x, bins)
        right_digits = np.digitize(x, bins, right=True)
        
        # 生成 0~1 之间的随机数
        rands = rng.random(len(x))
        
        # 在 left 和 right 之间插值
        # 如果 left == right，则 digits = left
        # 如果 left != right (说明 x 恰好等于某个 bin edge，且该 edge 可能重复)，
        # 则随机分配到 [left, right] 范围内的 bin
        digits = rands * (right_digits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

    def _convert_to_token(self, sample, sample_id):
        # 1. 过滤掉 0 值
        # sample 已经是 Normalized + Log1p 后的值
        sample = sample[sample > 0]
        
        # 2. [vCross] Ranking: 按数值降序排列
        # 无论是否 Log1p，单调性不变，Ranking 结果不变
        sample = sample.sort_values(ascending=False)
        
        # 3. [vCross] Binning: 动态分箱 (Value-based with Seeded Noise)
        # 采用 scGPT 的策略：基于数值的分位数分箱。
        # 优点：保留了数值的相对大小信息 (magnitude)。
        # 解决重复值问题：使用 seeded random noise 将重复值分散到相邻 bin。
        if len(sample) > 0:
            try:
                # 生成 Hash 作为 seed modifier，确保同一个样本每次处理都一样
                if isinstance(sample_id, str):
                    seed_mod = int(hash(sample_id)) % (10**8) 
                else:
                    seed_mod = int(sample_id)
                
                # 计算 Edges
                # scGPT 原版是用 np.linspace(0, 1, n_bins - 1)
                # 这意味着它包含了 0% (min) 和 100% (max) 作为 edge。
                # q = np.linspace(0, 1, self.num_bins - 1)
                # 我们使用标准分位点，得到 n_bins - 1 个内部分割点
                q = np.linspace(0, 1, self.num_bins - 1)
                bins = np.quantile(sample.values, q)
                
                # 调用带随机扰动的 digitize
                bin_ids_values = self._digitize(sample.values, bins, seed_modifier=seed_mod)
                
                # scGPT 的 bin id 从 1 开始 (0 是 padding)
                # digitize 返回的 0 表示 x < bins[0] (最小区间) -> 映射到 Bin 1
                # digitize 返回的 n_bins-1 表示 x >= bins[-1] (最大区间) -> 映射到 Bin n_bins
                # 所以我们需要 +1
                bin_ids = pd.Series(bin_ids_values + 1, index=sample.index)
                
            except Exception as e:
                print(f"Warning: Binning failed for sample {sample_id}: {e}")
                bin_ids = pd.Series(1, index=sample.index)
        else:
            bin_ids = pd.Series([], dtype=int)

        # 4. 构建 Input IDs (Taxon Token)
        # 结构: <bos> + [Taxon1, Taxon2...] + <eos>
        taxa_list = sample.index.tolist()
        sent = ['<bos>'] + taxa_list + ['<eos>']
        input_ids = self.tokenizer.encode(sent)
        length = len(input_ids)

        # 5. 构建 Value IDs (Bin Token)
        # <bos>/<eos> 的 Bin ID 设为 0 (Padding/Special)
        # Taxon 对应的 Bin ID 从 bin_ids 获取
        # 注意：bin_ids 的顺序已经和 taxa_list 一致（因为 sample 已经排过序）
        value_list = bin_ids.values.tolist()
        # [0] for <bos>, value_list for taxa, [0] for <eos>
        value_ids_raw = [0] + value_list + [0]
        
        # 6. 构建 Condition IDs (Metadata)
        condition_ids_raw = []
        if self.use_meta_cols:
            row = self.metadata.loc[sample_id]
            for col in self.use_meta_cols:
                val = row[col]
                # 处理空值
                if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
                    val_str = "Unknown"
                else:
                    val_str = str(val).strip()
                
                # 编码
                le = self.meta_encoders[col]
                # 处理未见过的 label (虽然在 construct 时应该都见过，但在 inference 时要注意)
                # 这里假设 construct 覆盖全集
                if val_str in le.classes_:
                    code = le.transform([val_str])[0]
                else:
                    # fallback to Unknown if possible, or 0
                    code = 0 
                condition_ids_raw.append(code)
        
        # 7. Padding & Truncate
        # 对 input_ids 和 value_ids 进行 padding
        if len(input_ids) > self.max_len:
            # Truncate
            input_ids = input_ids[:self.max_len-1] + [input_ids[-1]]
            value_ids_raw = value_ids_raw[:self.max_len-1] + [value_ids_raw[-1]]
        else:
            # Padding
            pad_len = self.max_len - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            value_ids_raw.extend([0] * pad_len) # Bin 0 for padding

        return input_ids, value_ids_raw, condition_ids_raw, length
    
    def _preprocess(self, data):
        # 提取 Genus
        genus = data.columns.to_series().apply(
            lambda name: extract_taxon(name, rank="Genus")
        )
        data.columns = genus.values
        # 合并同属
        data = data.groupby(data.columns, axis=1).sum()

        before = data.shape[0]

        # 对齐到 Phylogeny (如果存在)
        if self.phylogeny is not None:
            target_df = pd.DataFrame(index=self.phylogeny.index)
            # 仅保留在 Phylogeny 中存在的 OTU，缺失的补 0
            data = target_df.merge(data.T, left_index=True, right_index=True, how='left').fillna(0).T
        
        # 删除全 0 样本
        data = data.loc[(data != 0).any(axis=1)]
        print(f'{before - data.shape[0]} samples are dropped for all zeroes')

        # 计算相对丰度 (Total Sum Scaling)
        # data.div(data.sum(axis=1), axis=0) 得到的是 0-1 之间的相对丰度
        # 为了让 Log1p 有意义，我们需要将其缩放到 counts 级别 (Normalized Counts)
        
        # 1. 确定 Scaling Factor
        library_sizes = data.sum(axis=1)
        if self.normalize_total is not None:
            target_sum = self.normalize_total
            print(f"[vCross] Normalizing to fixed target sum: {target_sum}")
        else:
            target_sum = library_sizes.median()
            print(f"[vCross] Normalizing to median library size: {target_sum:.2f}")
            
        # 2. 执行归一化 (CPM-like)
        data = data.div(library_sizes, axis=0) * target_sum
        
        # Log1p 转换
        if self.log1p:
            print("[vCross] Applying Log1p transform...")
            data = np.log1p(data)
        
        return data
