import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
import torch
import umap

# 加载两个 batch 的 embeddings
emb0 = torch.load("./checkpoint/cache/emb_batch_0.pt", weights_only=True)
emb1 = torch.load("./checkpoint/cache/emb_batch_1.pt", weights_only=True)

# Mean pooling: (batch, seq_len, d_model) -> (batch, d_model)
emb0_pooled = emb0.mean(dim=1)
emb1_pooled = emb1.mean(dim=1)

# 合并 embeddings 并转为 numpy
embeddings = torch.cat([emb0_pooled, emb1_pooled], dim=0).cpu().numpy()
labels = np.array([0] * len(emb0_pooled) + [1] * len(emb1_pooled))
print(f"Pooled embeddings shape: {embeddings.shape}")

# Step 1: PCA 降到 50 维
pca_50 = PCA(n_components=50)
emb_pca50 = pca_50.fit_transform(embeddings)
print(f"After PCA (50D): {emb_pca50.shape}")

# Step 2: UMAP 降到 2 维（从原始 50 维 PCA 结果）
reducer = umap.UMAP(n_components=2)
emb_umap = reducer.fit_transform(emb_pca50)
print(f"After UMAP (2D): {emb_umap.shape}")

# 用 UMAP 2D 做简单聚类，计算 ARI / FMI
kmeans = KMeans(n_clusters=2, n_init="auto")
pred_labels = kmeans.fit_predict(emb_umap)

ari_umap = adjusted_rand_score(labels, pred_labels)
fmi_umap = fowlkes_mallows_score(labels, pred_labels)
print(f"ARI_UMAP: {ari_umap:.4f}")
print(f"FMI_UMAP: {fmi_umap:.4f}")

# Step 3: PCA 降到 2 维（从原始 50 维 PCA 结果）
pca_2 = PCA(n_components=2)
emb_pca2 = pca_2.fit_transform(emb_pca50)
print(f"After PCA (2D): {emb_pca2.shape}")

# 用 PCA 2D 做简单聚类，计算 ARI / FMI
kmeans = KMeans(n_clusters=2, n_init="auto")
pred_labels = kmeans.fit_predict(emb_pca2)

ari_pca = adjusted_rand_score(labels, pred_labels)
fmi_pca = fowlkes_mallows_score(labels, pred_labels)
print(f"ARI_PCA: {ari_pca:.4f}")
print(f"FMI_PCA: {fmi_pca:.4f}")

# 画图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# UMAP 图
axes[0].scatter(emb_umap[labels == 0, 0], emb_umap[labels == 0, 1], s=5, alpha=0.6, label="batch_0")
axes[0].scatter(emb_umap[labels == 1, 0], emb_umap[labels == 1, 1], s=5, alpha=0.6, label="batch_1")
axes[0].set_title("UMAP (from PCA-50D)")
axes[0].set_xlabel("UMAP-1")
axes[0].set_ylabel("UMAP-2")
axes[0].text(0.02, 0.98, f"ARI_UMAP={ari_umap:.4f}\nFMI_UMAP={fmi_umap:.4f}",
             transform=axes[0].transAxes, va="top", ha="left",
             fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
axes[0].legend()

# PCA 2D 图
axes[1].scatter(emb_pca2[labels == 0, 0], emb_pca2[labels == 0, 1], s=5, alpha=0.6, label="batch_0")
axes[1].scatter(emb_pca2[labels == 1, 0], emb_pca2[labels == 1, 1], s=5, alpha=0.6, label="batch_1")
axes[1].set_title("PCA (2D from PCA-50D)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].text(0.02, 0.98, f"ARI_PCA={ari_pca:.4f}\nFMI_PCA={fmi_pca:.4f}",
             transform=axes[1].transAxes, va="top", ha="left",
             fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
axes[1].legend()

plt.tight_layout()
plt.savefig("/home_zfs/wanghanyi/mRNAHyena/scripts/pca_umap_plot.png", dpi=150)
plt.show()
print("Figure saved to /home_zfs/wanghanyi/mRNAHyena/scripts/pca_umap_plot.png")
