import torch

path = "/home_zfs/wanghanyi/mRNAHyena/checkpoint/cache/emb_batch_0.pt"
emb = torch.load(path, weights_only=True)
print("type:", type(emb))
print("shape:", tuple(emb.shape))
print("dtype:", emb.dtype)
print("device:", emb.device)
print("ndim:", emb.ndim)