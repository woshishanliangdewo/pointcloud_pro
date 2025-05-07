import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        # 空间特征编码（网页1的Embedding思想[1](@ref)）
        self.pos_embed = nn.Linear(3, d_model)  # 空间坐标编码
        self.vel_embed = nn.Linear(3, d_model)  # 速度编码
        
        # 时空位置编码（网页5的多头注意力扩展[5](@ref)）
        self.temporal_pe = nn.Parameter(torch.randn(10, 1024,  2 * d_model)) # 时间位置编码(seq_len=10)
        self.spatial_pe = nn.Parameter(torch.randn(1, 1024,  2 * d_model)) # 空间位置编码(points=1024)
        
        # Transformer编码层（网页3的时空注意力融合[3](@ref)）
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model*2,  # 空间+速度特征拼接
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # 异常检测头（网页7的Dropout应用[7](@ref)）
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model*2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        
    def forward(self, pos, vel):
        """
        输入维度：
        pos: [batch, seq, points, 3]
        vel: [batch, seq, points, 3]
        输出维度：
        anomaly_score: [batch, points]
        """
        batch_size, seq_len, num_points, _ = pos.shape
        print(pos.size())
        # 特征嵌入（空间+速度）
        pos_feat = self.pos_embed(pos)  # [batch, seq, points, d_model]
        vel_feat = self.vel_embed(vel)  # [batch, seq, points, d_model]
        combined = torch.cat([pos_feat, vel_feat], dim=-1)  # [batch, seq, points, 2*d_model]
        
        # 时空位置编码（网页5的多头注意力扩展[5](@ref)）
        # combined = combined.permute(1, 2, 0, 3)  # [seq, points, batch, 2d]
        # print(combined.size())
        # print(self.temporal_pe[:seq_len].size())

        # combined += self.temporal_pe[:seq_len]    # 时间编码
        # combined += self.spatial_pe[:, :num_points] # 空间编码
        
         # 添加时间编码（形状：[S, 1, 2D]）
        time_enc = self.temporal_pe[:seq_len].unsqueeze(1)  # [S, 1, 2D]
        combined += time_enc.expand_as(combined)
    
        # 添加空间编码（形状：[1, P, 2D]）
        space_enc = self.spatial_pe.unsqueeze(0)  # [1, P, 2D]
        combined += space_enc.expand(batch_size, -1, -1, -1)

        
        # Transformer处理（网页6的编码器结构[6](@ref)）
        combined = combined.flatten(1,2)  # [seq, points*batch, 2d]
        encoded = self.transformer_encoder(combined)  # [seq, points*batch, 2d]
        encoded = encoded.view(seq_len, num_points, batch_size, -1)
        
        # 时间维度聚合（网页3的时序特征提取[3](@ref)）
        time_avg = encoded.mean(dim=0)  # [points, batch, 2d]
        scores = self.anomaly_head(time_avg)  # [points, batch, 1]
        return scores.squeeze(-1).permute(1,0)  # [batch, points]



def train_model(model, dataloader, epochs=50):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for pos, vel, label in dataloader:
            optimizer.zero_grad()
            outputs = model(pos, vel)
            loss = criterion(outputs, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 网页7的梯度裁剪[7](@ref)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


def inference(model, pos_sequence, vel_sequence):
    """
    输入单一样本的时空序列:
    pos_sequence: [seq_len, num_points, 3]
    vel_sequence: [seq_len, num_points, 3]
    返回异常得分矩阵: [num_points]
    """
    model.eval()
    with torch.no_grad():
        scores = model(pos_sequence.unsqueeze(0), vel_sequence.unsqueeze(0))
    return scores.squeeze(0).cpu().numpy()


model = SpatioTemporalTransformer(d_model=128, nhead=4, num_layers=3)
train_model(model, dataloader, epochs=30)


# 模拟测试数据
test_pos = torch.randn(10, 1024, 3)  # 10帧，1024个点
test_vel = torch.randn(10, 1024, 3)

# 推理检测
anomaly_scores = inference(model, test_pos, test_vel)

# 标记异常点（阈值=0.8）
anomaly_mask = anomaly_scores > 0.8
print(f"Detected {anomaly_mask.sum()} anomalies")