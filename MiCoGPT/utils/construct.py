import pickle
from MiCoGPT.utils.corpus import MiCoGPTCorpus
from importlib.resources import files
import pandas as pd

def construct(
    input_path: str,
    output_path: str,
    meta_path: str,
    key: str = "genus",
):

    tokenizer_path = files("MiCoGPT")/"resources"/"MiCoGPTokenizer.pkl"

    max_len = 512
    print(f"max_len = {max_len}")

    # 加载 tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"tokenizer vocab size = {len(tokenizer.vocab)}")

    meta_df = pd.read_csv(meta_path, sep="\t", index_col="Run")
    print(f"metadata shape = {meta_df.shape}")

    corpus = MiCoGPTCorpus(
        data_path=str(input_path),
        metadata=meta_df,
        tokenizer=tokenizer,
        key=key,
        max_len=max_len
    )

    print(f"corpus length: {len(corpus)}")

    with open(output_path, "wb") as f:
        pickle.dump(corpus, f)

    return corpus