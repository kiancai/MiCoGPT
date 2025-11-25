import pickle
from configparser import ConfigParser
from MiCoGPT.utils.corpus import MiCoGPTCorpus
from importlib.resources import files

def construct(
    input_path: str,             # 输入丰度表文件
    output_path: str,            # 输出的 corpus pkl 路径
    key: str = "genus",
    no_normalize: bool = False,  # 是否跳过归一化
):

    res_dir = files("MiCoGPT")/"resources"
    tokenizer_path = res_dir/"MiCoGPTokenizer.pkl"
    config_path = res_dir/"config.ini"

    # 获取 max_len 配置
    cfg = ConfigParser()
    cfg.read(config_path)
    max_len = cfg.getint("construct", "max_len")
    print(f"max_len = {max_len}")

    # 加载 tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"tokenizer vocab size = {len(tokenizer.vocab)}")

    # 打印归一化提示
    if not no_normalize:
        print("Your data will be normalized with the phylogeny mean and std.")

    corpus = MiCoGPTCorpus(
        data_path=str(input_path),
        tokenizer=tokenizer,
        key=key,
        max_len=max_len,
        preprocess=not no_normalize,
    )

    print(f"corpus length: {len(corpus)}")

    with open(output_path, "wb") as f:
        pickle.dump(corpus, f)

    return corpus