# coding: utf-8
"""从大 CSV 中为指定物种各抽取最多 200 条，保存为单独 CSV。

默认物种（可按需修改）：
    Homo sapiens
    Mus musculus
    Drosophila melanogaster
    Caenorhabditis elegans
    Danio rerio
    Arabidopsis thaliana
    Zea mays
    Saccharomyces cerevisiae S288C
"""

import csv
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

# 允许超长字段
csv.field_size_limit(sys.maxsize)


DEFAULT_SPECIES = [
    "Homo sapiens",                  # 人
    "Mus musculus",                  # 小鼠
    "Drosophila melanogaster",       # 果蝇
    "Caenorhabditis elegans",        # 线虫
    "Danio rerio",                   # 斑马鱼
    "Arabidopsis thaliana",          # 拟南芥
    "Zea mays",                      # 玉米
    "Saccharomyces cerevisiae S288C",# 啤酒酵母 S288C
]


def sample_species(
    csv_path: str,
    species_list=None,
    per_species: int = 200,
    chunksize: int = 50_000,
    output_dir: Optional[Path] = None,
):
    if species_list is None:
        species_list = DEFAULT_SPECIES

    species_set = set(species_list)
    collected = {sp: [] for sp in species_list}
    counts = {sp: 0 for sp in species_list}  # 已采样行数
    required_cols = None  # infer later

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 先仅读取必要列，其他列全部保留原样输出（需知道所有列名）
    try:
        preview = pd.read_csv(csv_path, sep=",", engine="python", nrows=5)
        required_cols = list(preview.columns)
    except Exception as e:
        print(f"无法读取文件: {csv_path}")
        print(e)
        return

    total_rows = 0
    done_species = set()

    try:
        reader = pd.read_csv(
            csv_path,
            sep=",",
            engine="python",
            chunksize=chunksize,
        )
    except Exception as e:
        print(f"无法读取文件: {csv_path}")
        print(e)
        return

    for chunk in reader:
        total_rows += len(chunk)

        # 只处理仍未满额的物种
        remaining = species_set - done_species
        if not remaining:
            break

        filt = chunk[chunk["ORGANISM"].isin(remaining)]
        if filt.empty:
            continue

        for sp, subdf in filt.groupby("ORGANISM"):
            have = counts[sp]
            need = per_species - have
            if need <= 0:
                done_species.add(sp)
                continue
            take = subdf.iloc[:need]
            collected[sp].append(take)
            counts[sp] += len(take)
            if counts[sp] >= per_species:
                done_species.add(sp)

    # 输出每个物种的 CSV
    for sp in species_list:
        if collected[sp]:
            df_out = pd.concat(collected[sp], axis=0)
        else:
            df_out = pd.DataFrame(columns=required_cols)

        out_name = sp.replace(" ", "_").replace("/", "_") + "_sample.csv"
        out_path = output_dir / out_name
        df_out.to_csv(out_path, index=False)
        print(f"{sp}: 采样 {len(df_out)} 条 -> {out_path}")

    print(f"扫描完成，总读取行数: {total_rows}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "/share/datasets/mRNA/refseq/complete/complete_NM_mRNA_filtered.csv"

    sample_species(csv_path)
