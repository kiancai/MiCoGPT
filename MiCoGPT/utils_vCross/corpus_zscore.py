import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
from importlib.resources import files
from MiCoGPT.utils.tools import extract_taxon

class MiCoGPTCorpusZScore(Dataset):
    def __init__(self, 
                 tokenizer,                # PreTrainedTokenizer
                 data_path: str,
                 phylogeny_path = None,
                 metadata: pd.DataFrame | None = None,  # 样本的 metadata
                 key: str = "genus",
                 max_len: int = 512,
                 return_metadata: bool = False,
                 ):
        
        if phylogeny_path is None:
            # Default path if not provided
            phylogeny_path = str(files("MiCoGPT") / "resources/phylogeny.csv")

        # 注意，这里读取丰度表后进行了转置，使得样本在行，OTU 在列
        self.data = pd.read_csv(data_path, sep=',', index_col=0).T
        self.tokenizer = tokenizer
        self.phylogeny = pd.read_csv(phylogeny_path, index_col=0)
        self.max_len = max_len
        self.key = key
        self.return_metadata = return_metadata

        # 合并 genus，转相对丰度，做 z-score
        self.data, self.zero_values = self._preprocess(self.data)

        # 记录样本顺序：self.data 的行索引就是 sample_id
        #  以后 tokens[i] <-> sample_ids[i] <-> metadata.iloc[i]
        self.sample_ids = list(self.data.index)

        # 如果传入了 metadata，则对齐到当前样本顺序
        # 要求 metadata 的 index 就是 sample_id
        if metadata is not None:
            # 确保是 DataFrame
            if not isinstance(metadata, pd.DataFrame):
                raise TypeError("metadata 必须是 pandas.DataFrame")
            # 确保 index 是唯一的（sample_id 唯一）
            if not metadata.index.is_unique:
                raise ValueError("metadata.index (sample_id) 必须唯一")
            # 按 self.sample_ids 的顺序对齐 metadata
            # 如果缺少某些 sample_id，这里会抛 KeyError
            try:
                self.metadata = metadata.loc[self.sample_ids].copy()
            except KeyError as e:
                # 尝试忽略缺失的样本，只保留交集
                print(f"Warning: metadata 中缺少某些样本的记录。将只保留交集。")
                common_ids = list(set(self.sample_ids) & set(metadata.index))
                self.metadata = metadata.loc[common_ids].copy()
                # 同时也需要更新 data 和 sample_ids
                self.data = self.data.loc[common_ids]
                self.sample_ids = common_ids
        else:
            self.metadata = None

        tokens_list = []
        length_list = []
        
        # 对于每个 sample，提取其它那一行的 taxa 与标准化丰度
        # 送入 _convert_to_token 函数进行处理。tqdm 显示进度
        print("Converting samples to tokens (Z-Score sorting)...")
        for sample in tqdm(self.data.index):
            tokens, length = self._convert_to_token(self.data.loc[sample])
            tokens_list.append(tokens)
            length_list.append(length)
        
        print(f'Total {len(tokens_list)} samples.\n\
            Max length is {max(length_list)}.\n\
            Average length is {np.mean(length_list)}.\n\
            Min length is {min(length_list)}.')
        self.tokens = torch.LongTensor(tokens_list)
    
    def __len__(self):
        return len(self.tokens)        
    
    def __getitem__(self, index):
        # 1. Input IDs Mask
        attention_mask = torch.ones(self.tokens[index].shape)
        attention_mask[self.tokens[index] == self.tokenizer.pad_token_id] = 0
        tokens = self.tokens[index].clone()

        # 返回 vCross 兼容格式
        # value_ids 和 condition_ids 设为 0 (padding/dummy)
        # 具体的 Collator (BaselineClassificationCollator) 可能会把它们设为 None
        value_ids = torch.zeros_like(tokens)
        condition_ids = torch.zeros(1, dtype=torch.long) # Dummy condition

        return {
            'input_ids': tokens,
            'value_ids': value_ids,
            'condition_ids': condition_ids,
            'attention_mask': attention_mask
        }
    
    def subset_by_metadata(self, condition):
        if self.metadata is None:
            raise ValueError("当前 corpus 没有关联 metadata，无法按 metadata 划分子集。")
        # 1. 生成布尔 mask
        mask = condition(self.metadata)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != len(self):
            # 重新对齐 metadata 和 self.data 可能导致长度变化，需要检查
             raise ValueError(
                f"condition 返回的 mask 长度为 {mask.shape[0]}，"
                f"但 corpus 样本数为 {len(self)}，请检查 condition 的实现。"
            )
        # 2. 将 True 的位置转成 indices
        indices = np.where(mask)[0].tolist()
        # 3. 返回 PyTorch 自带的 Subset
        print(f"[subset_by_metadata] Selected {len(indices)} samples out of {len(self)}.")
        return Subset(self, indices)
    
    def subset_by_ids(self, ids):
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

        return Subset(self, indices)
    
    def _convert_to_token(self, sample):
        # 删除 z-score 低于 0 的 OTU，然后降序排序
        sample = sample[sample > self.zero_values]
        sample = sample.sort_values(ascending=False)

        # add bos & eos
        sent = ['<bos>'] + sample.index.tolist() + ['<eos>']

        # convert to token
        tokens = self.tokenizer.encode(sent)
        length = len(tokens)
        
        # padding and truncate
        if len(tokens) > self.max_len:
            # tokens = tokens[:self.max_len]
            # 保留结尾的 eos
            tokens = tokens[:self.max_len-1] + [tokens[-1]]
        else:
            # padding
            tokens.extend([self.tokenizer.pad_token_id] * (self.max_len - len(tokens)))
        
        return tokens, length
    
    def _preprocess(self, data):
        # 提取 Genus
        genus = data.columns.to_series().apply(
            lambda name: extract_taxon(name, rank="Genus")
        )
        # 把列名替换成提取到的 Genus（g__XXX，或者 None/NaN）
        data.columns = genus.values
        # 按列名分组，把同属的列加总
        data = data.groupby(data.columns, axis=1).sum()

        # 记录当前样本数（行数）
        before = data.shape[0]

        # 参考 phylogeny 中的 OTU，删除不在 phylogeny 中的 OTU
        # 这里对 data 进行转置，使得 OTU 在行，样本在列
        # 接着以 left 即 target_df 为准，合并 data，缺失的 OTU 用 0 填充
        # 这样就得到了一个 OTU 在行，样本在列的丰度表，且 OTU 按照 phylogeny 中的顺序排列
        # 最后再转置，使得样本在行，OTU 在列
        target_df = pd.DataFrame(index=self.phylogeny.index)
        data = target_df.merge(data.T, left_index=True, right_index=True, how='left').fillna(0).T
        
        # 如果一个 OTU 至少有一个 sample 不为 0，则保留该 OTU。否则，删除该 OTU
        data = data.loc[(data != 0).any(axis=1)]
        print(f'{before - data.shape[0]} samples are dropped for all zeroes')

        # 对每个 sample 进行相对丰度计算
        data = data.div(data.sum(axis=1), axis=0)

        print("Your data will be normalized with the phylogeny mean and std.")
        
        # z-score normalize
        data.loc['zero'] = 0  # 设置一个虚拟样本所有 taxa 为 0
        data = (data - self.phylogeny['mean']) / self.phylogeny['std']
        zero_values = data.loc['zero']
        data = data.drop('zero')  # 删除 0 样本，并返回对应 OTU 的零值
        return data, zero_values
