import pandas as pd
import csv, sys

# 允许超长字段（默认 ~128KB 会触发错误）
csv.field_size_limit(sys.maxsize)

# =============================================================================
# 配置部分 —— 修改为你的实际路径
# =============================================================================
csv_files = [
    # "/share/datasets/mRNA/refseq/vertebrate_mammalian/NM_mRNA/human_mRNA_filtered.csv",
    # "/share/datasets/mRNA/refseq/vertebrate_mammalian/NM_mRNA/human_mRNA.csv",
    # "/share/datasets/mRNA/refseq/vertebrate_mammalian/NM_mRNA/vertebrate_mammalian_mRNA_filtered.csv",
    # "/share/datasets/mRNA/refseq/vertebrate_mammalian/NM_mRNA/vertebrate_mammalian_mRNA.csv",
    # "/share/datasets/mRNA/refseq/vertebrate_mammalian/XM_mRNA/human_mRNA_filtered.csv",
    # "/share/datasets/mRNA/refseq/vertebrate_mammalian/XM_mRNA/human_mRNA.csv",
    # "/share/datasets/mRNA/refseq/vertebrate_mammalian/XM_mRNA/vertebrate_mammalian_mRNA_filtered.csv",
    "/share/datasets/mRNA/refseq/vertebrate_mammalian/XM_mRNA/vertebrate_mammalian_mRNA.csv",
    # "/share/datasets/mRNA/refseq/complete/complete_NM_mRNA_filtered.csv",
    # "/share/datasets/mRNA/refseq/complete/complete_XM_mRNA_filtered.csv",
]

# output_fasta = "data/mRNA_data/vertebrate_mammalian_NM_human_mRNA_filtered_separators.fasta"
# output_fasta = "data/mRNA_data/vertebrate_mammalian_NM_human_mRNA_separators.fasta"
# output_fasta = "data/mRNA_data/vertebrate_mammalian_NM_vertebrate_mammalian_mRNA_filtered_separators.fasta"
# output_fasta = "data/mRNA_data/vertebrate_mammalian_NM_vertebrate_mammalian_mRNA_separators.fasta"
# output_fasta = "data/mRNA_data/vertebrate_mammalian_XM_human_mRNA_filtered_separators.fasta"
# output_fasta = "data/mRNA_data/vertebrate_mammalian_XM_human_mRNA_separators.fasta"
# output_fasta = "data/mRNA_data/vertebrate_mammalian_XM_vertebrate_mammalian_mRNA_filtered_separators.fasta"
output_fasta = "data/mRNA_data/vertebrate_mammalian_XM_vertebrate_mammalian_mRNA_separators.fasta"
# output_fasta = "data/mRNA_data/complete_NM_mRNA_filtered_separators.fasta"
# output_fasta = "data/mRNA_data/complete_XM_mRNA_filtered_separators.fasta"

# ★ 分隔符（可自行修改，如 "|"、"NNN"、"<CDS>" 等）
SEPARATOR = "|"

# =============================================================================
# 主处理代码
# =============================================================================

def csv_to_fasta(csv_files, output_fasta, chunksize=50_000):
    """
    将多个 mRNA CSV 文件转成一个全长 mRNA FASTA 文件（流式处理，避免 OOM）。
    FASTA 格式：
        >RefSeq_ID Organism
        5UTR | CDS | 3UTR
    """
    required_cols = ["RefSeq_ID", "ORGANISM", "5UTR", "CDS", "3UTR"]
    total = 0

    with open(output_fasta, "w") as out:
        for file in csv_files:
            try:
                reader = pd.read_csv(
                    file,
                    sep=",",
                    engine="python",
                    chunksize=chunksize,
                    usecols=required_cols,
                )
            except Exception as e:
                print(f"无法读取文件: {file}")
                print(e)
                continue

            for chunk in reader:
                # 列检查
                if not set(required_cols).issubset(chunk.columns):
                    print(f"文件 {file} 缺少必要列, 跳过该块...")
                    print("需要的列:", required_cols)
                    print("找到的列:", chunk.columns.tolist())
                    continue

                for _, row in chunk.iterrows():
                    refseq_id = str(row["RefSeq_ID"]).strip()
                    organism = str(row["ORGANISM"]).strip()

                    utr5 = str(row["5UTR"]).strip().upper()
                    cds = str(row["CDS"]).strip().upper()
                    utr3 = str(row["3UTR"]).strip().upper()

                    full_mrna = SEPARATOR.join([utr5, cds, utr3])
                    header = f">{refseq_id} {organism}"

                    out.write(header + "\n")
                    for i in range(0, len(full_mrna), 60):
                        out.write(full_mrna[i:i + 60] + "\n")
                    total += 1

    print(f"写入完成: {output_fasta}")
    print(f"共写入 {total} 条记录")


if __name__ == "__main__":
    csv_to_fasta(csv_files, output_fasta)
