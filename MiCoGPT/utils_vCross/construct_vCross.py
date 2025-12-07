import pickle
from MiCoGPT.utils_vCross.corpus_vCross import MiCoGPTCorpusVCross
from importlib.resources import files
import pandas as pd
import os

def construct(
    input_path: str,
    output_path: str,
    meta_path: str,
    key: str = "genus",
    use_meta_cols: list[str] | None = None,
    num_bins: int = 51,
    log1p: bool = True,
    normalize_total: float | None = None,
):

    # [vCross] 使用纯净版的 Tokenizer vCross
    tokenizer_path = files("MiCoGPT")/"resources"/"MiCoGPTokenizer_vCross.pkl"
    print("[construct_vCross] Using MiCoGPTokenizer_vCross.pkl")
    
    max_len = 512
    print(f"max_len = {max_len}")

    # 加载 tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"tokenizer vocab size = {len(tokenizer.vocab)}")

    meta_df = pd.read_csv(meta_path, sep="\t", index_col="Run", low_memory=False)
    print(f"metadata shape = {meta_df.shape}")

    corpus = MiCoGPTCorpusVCross(
        data_path=str(input_path),
        metadata=meta_df,
        tokenizer=tokenizer,
        key=key,
        max_len=max_len,
        use_meta_cols=use_meta_cols,
        num_bins=num_bins,
        log1p=log1p,
        normalize_total=normalize_total,
    )

    print(f"corpus length: {len(corpus)}")

    # 保存 Corpus
    with open(output_path, "wb") as f:
        pickle.dump(corpus, f)
    
    # [vCross] 额外保存 Metadata Encoders
    # 路径规则：output_path 的目录 + "meta_encoders.joblib"
    encoder_path = os.path.join(os.path.dirname(output_path), "meta_encoders.joblib")
    corpus.save_encoders(encoder_path)
    print(f"Metadata Encoders saved to: {encoder_path}")

    return corpus
