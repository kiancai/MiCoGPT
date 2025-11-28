import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from torch.utils.data import Dataset
from importlib.resources import files

from MiCoGPT.utils.tools import extract_taxon

class MiCoGPTCorpus(Dataset):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 data_path: str,
                 phylogeny_path = files("MiCoGPT")/"resources/phylogeny.csv",
                 key='genus',
                 max_len=512):

        # 注意，这里读取丰度表后进行了转置，使得样本在行，OTU 在列
        self.data = pd.read_csv(data_path, sep=',', index_col=0).T
        self.tokenizer = tokenizer
        self.phylogeny = pd.read_csv(phylogeny_path, index_col=0)
        self.max_len = max_len
        
        # 合并 genus，转相对丰度，做 z-score
        self.data, self.zero_values = self._preprocess(self.data)
    
        tokens_list = []
        length_list = []
        
        # 对于每个 sample，提取其它那一行的 taxa 与标准化丰度
        # 送入 _convert_to_token 函数进行处理。tqdm 显示进度
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
        attention_mask = torch.ones(self.tokens[index].shape)
        attention_mask[self.tokens[index] == self.tokenizer.pad_token_id] = 0
        tokens = self.tokens[index].clone()

        return {'input_ids': torch.tensor(tokens),
                'attention_mask': attention_mask}

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

        # z-score normalize
        data.loc['zero'] = 0  # 设置一个虚拟样本所有 taxa 为 0
        data = (data - self.phylogeny['mean']) / self.phylogeny['std']
        zero_values = data.loc['zero']
        data = data.drop('zero')  # 删除 0 样本，并返回对应 OTU 的零值
        return data, zero_values


class SequenceClassificationDataset(Dataset):
    def __init__(self, seq, mask, labels):
        self.seq = seq
        self.mask = mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.seq[idx]),
            "attention_mask": torch.tensor(self.mask[idx]),
            "labels": torch.tensor(self.labels[idx])
        }
        
class MicroCorpusWithLabelTokens(Dataset):
    def __init__(self, tokens, labels, tokenizer):
        self.tokens = tokens
        self.tokenizer = tokenizer
        self.labels = torch.tensor(self.tokenizer.encode(labels)).view(-1, 1)
        # insert label tokens after <bos> 
        self.tokens = torch.cat((self.tokens[:, :1], self.labels, self.tokens[:, 1:-1]), dim=1)
        
    def __len__(self):
        return self.tokens.shape[0]
    
    def __getitem__(self, idx):
        attention_mask = torch.ones(self.tokens[idx].shape)
        attention_mask[self.tokens[idx] == self.tokenizer.pad_token_id] = 0
        tokens = self.tokens[idx].clone()

        return {'input_ids': torch.tensor(tokens),
                'attention_mask': attention_mask}
