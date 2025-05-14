import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import sys

pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(pythonpath)
sys.path.insert(0,pythonpath)
from utils.pointnet_utils import PointNetEncoder  # 需要自定义点云编码器

# 自监督数据生成器
class WindFieldSSL(Dataset):

    def __init__(self, num_samples=1000, num_points=2048):
        self.num_points = num_points
        self.data = []
        for _ in range(num_samples):
            coords = self.generate_obstacle_field(num_points)
            velocities = self.physically_constrained_flow(coords)
            velocities += self.add_shear_perturbation(coords)
            self.data.append((coords.astype(np.float32), velocities.astype(np.float32)))

    def generate_obstacle_field(self, n):
        # 生成带有随机障碍物的三维场景
        coords = np.random.rand(n, 3)*10 - 5  # 范围[-5,5]
        
        # 添加圆柱体障碍物
        cylinder_center = np.random.rand(3)*8-4
        cylinder_radius = np.random.uniform(0.5, 2.0)
        cylinder_mask = np.linalg.norm(coords[:,:2]-cylinder_center[:2], axis=1) < cylinder_radius
        coords[cylinder_mask] += np.random.normal(scale=0.1, size=(cylinder_mask.sum(),3))
        
        # 添加立方体障碍物
        cube_min = np.random.rand(3)*6-3
        cube_size = np.random.uniform(1.0, 3.0, 3)
        cube_mask = np.all((coords > cube_min) & (coords < cube_min+cube_size), axis=1)
        coords[cube_mask] += np.random.normal(scale=0.1, size=(cube_mask.sum(),3))
        print(coords)
        return coords


    def physically_constrained_flow(self, coords):
        # 基于势流理论生成物理合理速度场
        velocities = np.zeros_like(coords)
        
        # 基础流速
        base_flow = np.array([2.0, -1.5, 0.5])
        
        # 添加障碍物绕流效应
        for i in range(coords.shape[0]):
            x, y, z = coords[i]
            
            # 圆柱绕流效应 (x-y平面)
            dx = x - 0.0  # 假设圆柱在原点
            dy = y - 0.0
            r = np.sqrt(dx ** 2 + dy ** 2)
            if r > 0.5:
                theta = np.arctan2(dy, dx)
                vr = (1 - (0.5/r) ** 2) * np.cos(theta)
                vtheta = -(1 + (0.5/r) ** 2) * np.sin(theta)
                velocities[i,0] += vr * np.cos(theta) - vtheta * np.sin(theta)
                velocities[i,1] += vr * np.sin(theta) + vtheta * np.cos(theta)
            
            # 垂直方向分层流
            velocities[i,2] += 0.3 * np.sin(z * np.pi/2)
            
        velocities += base_flow
        return velocities * 0.5  # 缩放速度场


    def add_shear_perturbation(self, coords):
        # 在障碍物下游添加剪切扰动
        perturbations = np.zeros_like(coords)
        
        # 圆柱后方剪切层
        cylinder_mask = (coords[:,0] > 0) & (coords[:,0] < 3) & \
                       (np.abs(coords[:,1]) < 1.5) & \
                       (coords[:,2] > -2)
        if np.any(cylinder_mask):
            perturbations[cylinder_mask] += np.random.normal(
                scale=np.array([0.5, 1.2, 0.3]), 
                size=(cylinder_mask.sum(),3)
            )
        
        # 随机点强剪切
        shear_points = np.random.choice(coords.shape[0], int(coords.shape[0]*0.02), replace=False)
        perturbations[shear_points] += np.random.normal(scale=3.0, size=(len(shear_points),3))
        
        return perturbations
    

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        coords, velocities = self.data[idx]
        return torch.from_numpy(coords), torch.from_numpy(velocities)

if __name__== "__main__":
    wind = WindFieldSSL()
    wind.generate_obstacle_field()
        