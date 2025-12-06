# [v2] 引用新的 corpus_v2，增加了对 metadata token 的支持

import pickle
from MiCoGPT.utils.corpus_v2 import MiCoGPTCorpus
from importlib.resources import files
import pandas as pd

def construct(
    input_path: str,
    output_path: str,
    meta_path: str,
    key: str = "genus",
    # [v2] 新增参数: 指定使用哪些 metadata 列
    use_meta_cols: list[str] | None = None,
):

    # [v2] 强制使用 MiCoGPTokenizer_v2.pkl
    tokenizer_path = files("MiCoGPT")/"resources"/"MiCoGPTokenizer_v2.pkl"
    print("[construct] Using MiCoGPTokenizer_v2.pkl")
    
    max_len = 512
    print(f"max_len = {max_len}")

    # 加载 tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"tokenizer vocab size = {len(tokenizer.vocab)}")

    meta_df = pd.read_csv(meta_path, sep="\t", index_col="Run", low_memory=False)
    print(f"metadata shape = {meta_df.shape}")

    # [v2] 传递 use_meta_cols 参数
    corpus = MiCoGPTCorpus(
        data_path=str(input_path),
        metadata=meta_df,
        tokenizer=tokenizer,
        key=key,
        max_len=max_len,
        use_meta_cols=use_meta_cols
    )

    print(f"corpus length: {len(corpus)}")

    with open(output_path, "wb") as f:
        pickle.dump(corpus, f)

    return corpus
