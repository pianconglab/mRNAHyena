import pandas as pd

# =============================================================================
# 配置部分 —— 修改为你的实际路径
# =============================================================================
csv_files = [
    "/share/datasets/mRNA/refseq/viral/host_human.csv",
]

output_fasta = "data/mRNA_data/viral_host_human_separators.fasta"

# ★ 分隔符（可自行修改，如 "|"、"NNN"、"<CDS>" 等）
SEPARATOR = "<SEP>"

# =============================================================================
# 主处理代码
# =============================================================================

def csv_to_fasta(csv_files, output_fasta):
    """
    将多个 mRNA CSV 文件转成一个全长 mRNA FASTA 文件。
    FASTA 格式：
        >RefSeq_ID Organism
        5UTR | CDS | 3UTR
    """
    all_records = []

    for file in csv_files:
        try:
            df = pd.read_csv(file, sep=None, engine="python")
        except Exception as e:
            print(f"无法读取文件: {file}")
            print(e)
            continue

        required_cols = {"RefSeq_ID", "ORGANISM", "5UTR", "CDS", "3UTR"}
        if not required_cols.issubset(df.columns):
            print(f"文件 {file} 缺少必要列, 跳过...")
            print("需要的列:", required_cols)
            print("找到的列:", df.columns.tolist())
            continue

        for _, row in df.iterrows():
            refseq_id = str(row["RefSeq_ID"]).strip()
            organism = str(row["ORGANISM"]).strip()

            utr5 = str(row["5UTR"]).strip().upper()
            cds  = str(row["CDS"]).strip().upper()
            utr3 = str(row["3UTR"]).strip().upper()

            # ★ 在 5UTR / CDS / 3UTR 之间加入分隔符
            full_mrna = SEPARATOR.join([utr5, cds, utr3])

            header = f">{refseq_id} {organism}"
            all_records.append((header, full_mrna))

    # 写入 FASTA
    with open(output_fasta, "w") as out:
        for header, seq in all_records:
            out.write(header + "\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i:i + 60] + "\n")

    print(f"写入完成: {output_fasta}")
    print(f"共写入 {len(all_records)} 条记录")


if __name__ == "__main__":
    csv_to_fasta(csv_files, output_fasta)
