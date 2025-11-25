import re
from typing import Optional

RANK_PREFIX_MAP = {
    "Kingdom": "k__",
    "Phylum": "p__",
    "Class": "c__",
    "Order": "o__",
    "Family": "f__",
    "Genus": "g__",
    "Species": "s__",
}

def extract_taxon(raw_name: str, rank: str) -> Optional[str]:

    if rank not in RANK_PREFIX_MAP:
        raise ValueError(f"Unknown rank: {rank!r}.")

    name = str(raw_name).strip()   # 统一成字符串并去掉首尾空白
    name = name.replace("; ", ";")
    prefix = RANK_PREFIX_MAP[rank]
    pattern = rf"{re.escape(prefix)}[^;]*"  # 匹配从前缀后到分号或字符串结束的内容
    m = re.search(pattern, name)
    if m:
        return m.group(0)
    print(
        f"[warning] Could not find {rank} (prefix {prefix!r}) in: {raw_name!r}"
    )
    return None
