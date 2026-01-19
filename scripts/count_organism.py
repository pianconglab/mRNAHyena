import csv
import sys
import os
from collections import Counter
from pathlib import Path
import pandas as pd

# 允许超长字段
csv.field_size_limit(sys.maxsize)


def count_organism(csv_path: str, chunksize: int = 100_000):
    """
    统计 CSV 中 ORGANISM 的种类及各自数量（分块读取，避免内存不足）。
    """
    col = "ORGANISM"
    counter = Counter()

    try:
        reader = pd.read_csv(
            csv_path,
            sep=",",
            engine="python",
            usecols=[col],
            chunksize=chunksize,
        )
    except Exception as e:
        print(f"无法读取文件: {csv_path}")
        print(e)
        return

    total = 0
    for chunk in reader:
        if col not in chunk.columns:
            print(f"文件缺少 {col} 列")
            return
        counter.update(chunk[col].astype(str).str.strip())
        total += len(chunk)

    print(f"文件: {csv_path}")
    print(f"总记录数: {total}")
    print(f"ORGANISM 种类数: {len(counter)}\n")
    print("Top 50 (按数量降序):")
    for name, cnt in counter.most_common(50):
        print(f"{cnt}\t{name}")

    # 保存全部结果到 scripts 目录下的 CSV
    scripts_dir = Path(__file__).resolve().parent
    out_csv = scripts_dir / "organism_counts.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["organism", "count"])
        for name, cnt in counter.most_common():
            writer.writerow([name, cnt])

    print(f"\n已保存全部统计到: {out_csv}")

    # 如果超过 50 种，提示剩余
    if len(counter) > 50:
        print(f"... 还有 {len(counter) - 50} 种未显示")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "/share/datasets/mRNA/refseq/complete/complete_NM_mRNA_filtered.csv"
    count_organism(csv_path)
