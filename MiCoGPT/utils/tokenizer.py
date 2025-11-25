from transformers import PreTrainedTokenizer
from typing import List

class MiCoGPTokenizer(PreTrainedTokenizer):

    def __init__(self, tokens, **kwargs):
        super().__init__(**kwargs)
        self.tokens = list(tokens)
        self.vocab = {v: i for i, v in enumerate(self.tokens)}
        self.ids_to_tokens = {i: v for i, v in enumerate(self.tokens)}
        self.add_special_tokens({
            'pad_token': '<pad>',
            'mask_token': '<mask>',
            'bos_token': '<bos>',
            'eos_token': '<eos>',
        })

    # 这里这个函数的作用是？
    def _tokenize(self, text):
        return [text]
    # def _tokenize(self, text):
    #     return list(text)

    
    def _add_tokens(self, new_tokens: List[str], special_tokens: bool = False) -> int:

        new_tokens = [token for token in new_tokens if token not in self.vocab]
        # 若过滤后为空，说明没有任何新 token 被添加
        if not new_tokens:
            return 0
    
        self.tokens.extend(new_tokens)
        self.vocab = {v: i for i, v in enumerate(self.tokens)}
        self.ids_to_tokens = {i: v for i, v in enumerate(self.tokens)}

        return len(new_tokens)
    
    def _convert_token_to_id(self, token):
        return self.vocab[token]
    
    def _convert_id_to_token(self, index):
        return self.ids_to_tokens[index]
    
    def get_vocab(self):
        return self.vocab
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    @property
    def vocab_size(self):
        return len(self.vocab)