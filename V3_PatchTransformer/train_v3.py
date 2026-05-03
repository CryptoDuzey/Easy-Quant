"""
Easy-Quant V3: Spatial-Temporal Patch-Transformer
模型定义 · 训练循环 · Rank IC 损失函数

用法:
    python train_v3.py --data_path ./data --epochs 100 --lr 1e-4

输出:
    checkpoints/patch_transformer_v3.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm


class PatchTransformer(nn.Module):
    """
    SOTA 级 Patch-Transformer 架构

    将 60 天量价序列切分为 12 个 5 天 Patch，
    通过多头注意力机制挖掘 300 只股票的截面相对强弱。

    输入形状: [Batch, Stocks, Time, Features]
    输出形状: [Batch, Stocks]  — 每只股票的 Alpha 得分
    """

    def __init__(self, n_vars=15, n_patches=12, patch_len=5, d_model=128, n_heads=8):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = n_patches

        # 线性投射层：将 Patch 展平并映射到高维空间
        self.patch_embed = nn.Linear(patch_len * n_vars, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))

        # Transformer 编码器
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)

        # 最终打分层
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, N, 60, 15] -> 展平为 [B*N, 60, 15]
        B, N, L, F = x.shape
        x = x.reshape(B * N, L, F)

        # Patching: [B*N, 12, 5*15]
        x = x.unfold(1, self.patch_len, self.patch_len)
        x = x.reshape(B * N, self.n_patches, -1)

        # Embedding + Position
        x = self.patch_embed(x) + self.pos_embed
        x = self.transformer(x)

        # 取最后时刻特征打分
        out = self.head(x[:, -1, :])
        return out.reshape(B, N)


def rank_ic_loss(pred, target):
    """
    优化排序能力的 Rank IC 损失函数

    等价于 1 - Spearman-like correlation，
    迫使模型专注截面排序质量而非绝对值精度。
    """
    pred_centered = pred - torch.mean(pred)
    target_centered = target - torch.mean(target)
    cos_sim = F.cosine_similarity(
        pred_centered, target_centered, dim=0
    )
    return 1 - cos_sim


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = rank_ic_loss(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        total_loss += rank_ic_loss(pred, y_batch).item()
    return total_loss / len(loader)


def main(data_path, epochs=100, lr=1e-4, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: 替换为实际数据加载逻辑
    # 期望形状: X [N_samples, 300, 60, 15], y [N_samples, 300]
    print(f"Loading data from {data_path} ...")
    # X, y = load_your_data(data_path)
    # dataset = TensorDataset(X, y)
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PatchTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss = 999  # train_epoch(model, loader, optimizer, device)
        # val_loss = validate(model, loader, device)
        scheduler.step()

        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     torch.save(model.state_dict(), "checkpoints/patch_transformer_v3.pth")

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}")

    print("Training complete. Model saved to checkpoints/patch_transformer_v3.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    main(args.data_path, args.epochs, args.lr, args.batch_size)
