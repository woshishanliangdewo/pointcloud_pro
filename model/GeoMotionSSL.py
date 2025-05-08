import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.pointnet_utils import PointNetEncoder

# 几何-运动一致性自监督模型
class GeoMotionSSL(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        # 共享特征编码器
        self.geo_encoder = PointNetEncoder(global_feat=True, 
                                         feature_transform=True, 
                                         channel=3)
        self.flow_encoder = PointNetEncoder(global_feat=True,
                                          feature_transform=True,
                                          channel=3)
        
        # 跨模态融合模块
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )
        
        # 自监督任务头
        self.recon_geo = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.Linear(feat_dim//2, 3)  # 重建坐标偏移量
        )
        self.recon_flow = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.Linear(feat_dim//2, 3)  # 重建速度场
        )
        
        # 对比学习投影头
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, geo, flow):
        # 几何特征提取
        geo_feat, _, _ = self.geo_encoder(geo)
        # 运动特征提取
        flow_feat, _, _ = self.flow_encoder(flow)
        
        # 特征融合
        fused_feat = self.fusion(torch.cat([geo_feat, flow_feat], dim=1))
        
        # 自监督重建任务
        delta_geo = self.recon_geo(fused_feat)
        recon_flow = self.recon_flow(fused_feat)
        
        # 对比学习特征
        proj_feat = self.proj_head(fused_feat)
        
        return delta_geo, recon_flow, proj_feat


# 多任务自监督损失
class SSLMultiLoss(nn.Module):
    def __init__(self, alpha=0.7, temp=0.1):
        super().__init__()
        self.alpha = alpha
        self.temp = temp


    def compute_curvature(geo):
    # 基于局部邻域协方差分析计算点云曲率
        B, N, _ = geo.size()
        curvatures = torch.zeros(B, N).to(geo.device)
        
        for b in range(B):
            points = geo[b]  # [N, 3]
            
            # 构建KDTree
            kdtree = KDTree(points.cpu().numpy())
            
            # 查询最近邻
            _, idx = kdtree.query(points.cpu().numpy(), k=10)
            
            for i in range(N):
                neighbors = points[idx[i]]  # [10,3]
                centroid = torch.mean(neighbors, dim=0)
                cov = torch.cov((neighbors - centroid).T)
                
                # 特征值分解
                eigenvalues = torch.linalg.eigvalsh(cov)
                lambda_sum = eigenvalues.sum()
                if lambda_sum > 1e-6:
                    curvatures[b,i] = eigenvalues[0] / lambda_sum  # 最小特征值占比作为曲率
                    
        return curvatures


    def geometric_consistency(self, pred_delta, geo, flow):
        # 基于几何曲率的运动一致性约束
        curvature = compute_curvature(geo)  # 几何曲率计算函数
        return torch.mean(curvature * torch.norm(pred_delta, dim=-1))
    
    def contrastive_loss(self, feat1, feat2):
        # 增强样本对的对比损失
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        logits = torch.mm(feat1, feat2.T) / self.temp
        labels = torch.arange(logits.size(0)).to(logits.device)
        return F.cross_entropy(logits, labels)
    
    def forward(self, delta_geo, recon_flow, proj_feat, geo, flow, aug_geo, aug_flow):
        # 重建损失
        flow_loss = F.mse_loss(recon_flow, flow)
        # 几何一致性损失
        geo_loss = self.geometric_consistency(delta_geo, geo, flow)
        # 对比损失
        contrast_loss = self.contrastive_loss(proj_feat, self.proj_head(aug_geo))
        
        total_loss = self.alpha*(flow_loss + geo_loss) + (1-self.alpha)*contrast_loss
        return total_loss



class KDTree:
    """简化的KDTree实现"""
    def __init__(self, data):
        self.data = data
        self.n_samples = data.shape[0]
        
    def query(self, points, k=1):
        indices = np.zeros((points.shape[0], k), dtype=int)
        for i, p in enumerate(points):
            dists = np.linalg.norm(self.data - p, axis=1)
            indices[i] = np.argpartition(dists, k)[:k]
        return None, indices


def augment_batch(geo, flow):
    """物理合理的数据增强"""
    B, N, _ = geo.size()
    device = geo.device
    
    # 随机旋转
    angles = torch.rand(3, device=device) * 2*np.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    Rx = torch.tensor([[1, 0, 0],
                       [0, cos_a[0], -sin_a[0]],
                       [0, sin_a[0], cos_a[0]]], device=device)
    Ry = torch.tensoriiiiii                                                                                                                                                                                               ([[cos_a[1], 0, sin_a[1]],
                       [0, 1, 0],
                       [-sin_a[1], 0, cos_a[1]]], device=device)
    Rz = torch.tensor([[cos_a[2], -sin_a[2], 0],
                       [sin_a[2], cos_a[2], 0],
                       [0, 0, 1]], device=device)
    R = torch.mm(torch.mm(Rz, Ry), Rx)
    
    # 应用旋转
    geo_rot = torch.matmul(geo, R.T)
    flow_rot = torch.matmul(flow, R.T)
    
    # 添加噪声
    geo_noise = (geo_rot + torch.randn_like(geo_rot) * 0.01).requires_grad_(True)
    flow_noise = (flow_rot + torch.randn_like(flow_rot) * 0.02).requires_grad_(True)
    
    geo_noise = geo_noise.clone().detach().requires_grad_(True)
    flow_noise = flow_noise.clone().detach().requires_grad_(True)
    # 保持不可压缩性约束
    # divergence = torch.mean(torch.sum(torch.autograd.grad(
    #     flow_noise.sum(), geo_noise, create_graph=False,allow_unused=True)[0], dim=-1))
    
    grads = torch.autograd.grad(
        flow_noise.sum(), geo_noise, create_graph=False, allow_unused=True)[0]
    if grads is not None:
        divergence = torch.mean(torch.sum(grads, dim=-1))
    else:
        divergence = torch.tensor(0., device=device)

    # 修正速度场保持div(u)=0
    if divergence.abs() > 0.1:
        flow_corrected = flow_noise - divergence * geo_noise / 3
    else:
        flow_corrected = flow_noise
        
    return geo_noise, flow_corrected

# 训练流程
def train_ssl():
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WindFieldSSL()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = GeoMotionSSL().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = SSLMultiLoss(alpha=0.7)
    
    # 训练循环
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for geo, flow in dataloader:
            geo, flow = geo.to(device), flow.to(device)
            
            # 数据增强
            aug_geo, aug_flow = augment_batch(geo, flow)
            
            # 前向传播
            delta_geo, recon_flow, proj_feat = model(geo, flow)
            _, _, aug_feat = model(aug_geo, aug_flow)
            
            # 计算损失
            loss = criterion(delta_geo, recon_flow, proj_feat,
                            geo, flow, aug_geo, aug_flow)
            
            # 反向传播
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# 下游任务微调
def finetune_shear_detection(pretrained_model):
    # 冻结编码层
    for param in pretrained_model.geo_encoder.parameters():
        param.requires_grad = True
    for param in pretrained_model.flow_encoder.parameters():
        param.requires_grad = True
    
    # 添加分类头
    classifier = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )
    
    # 微调训练流程...

train_ssl()