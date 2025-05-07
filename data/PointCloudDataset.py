import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PointCloudDataset(Dataset):
    def __init__(self, seq_length=10, point_num=1024):
        """
        模拟点云时空序列数据
        :param seq_length: 时间序列长度（默认10帧）
        :param point_num: 单帧点云数量（默认1024点）
        """
        # 生成正态分布的空间坐标 (batch, seq, points, 3)
        self.positions = np.random.randn(100, seq_length, point_num, 3)  # 100个样本
        # 生成带有随机异常的速度场 (batch, seq, points, 3)
        self.velocities = np.random.randn(100, seq_length, point_num, 3) * 0.1
        # 添加5%的异常点（速度突变）
        anomaly_mask = np.random.rand(*self.velocities.shape) < 0.05
        self.velocities[anomaly_mask] += np.random.randn(anomaly_mask.sum()) * 2
        
    def __len__(self):
        return self.positions.shape[0]
    
    def __getitem__(self, idx):
        """
        返回数据格式：
        pos_tensor: [seq_len, num_points, 3] 空间坐标序列
        vel_tensor: [seq_len, num_points, 3] 速度序列
        anomaly_label: [num_points] 异常点标签
        """
        # 归一化处理（参考网页6的位置编码思想[6](@ref)）
        pos = (self.positions[idx] - self.positions[idx].mean(axis=(0,1))) / self.positions[idx].std(axis=(0,1))
        vel = (self.velocities[idx] - self.velocities[idx].mean(axis=(0,1))) / self.velocities[idx].std(axis=(0,1))
        
        # 计算速度突变标签（网页3的时空特征思想[3](@ref)）
        vel_diff = np.abs(vel[1:] - vel[:-1]).mean(axis=(0,2))
        label = (vel_diff > np.quantile(vel_diff, 0.95)).astype(np.float32)
        
        return (
            torch.FloatTensor(pos),  # [seq, points, 3]
            torch.FloatTensor(vel),  # [seq, points, 3]
            torch.FloatTensor(label) # [points]
        )