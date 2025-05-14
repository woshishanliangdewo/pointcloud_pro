import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.pointnet_utils import PointNetEncoder
from utils.KdTree import KDTree

# 几何-运动一致性自监督模型
class GeoMotionSSL(nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        # 共享特征编码器(使用pointnet网络，进行点云特征提取)
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
        
        # 自监督任务头?(What's this)
        self.recon_geo = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.Linear(feat_dim//2, 3)  # 重建坐标偏移量
        )
    
        self.recon_flow = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.Linear(feat_dim//2, 3)  # 重建速度场
        )
        
        # 对比学习投影头?(What's this)
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, geo, flow):
        # 几何特征提取
        geo =  geo.transpose(2,1)
        geo_feat, _, _ = self.geo_encoder(geo)
        # 运动特征提取
        flow =  flow.transpose(2,1)
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

    def compute_curvature(self):
    # 基于局部邻域协方差分析计算点云曲率
        B, N, _ = self.size()
        curvatures = torch.zeros(B, N).to(self.device)
        
        for b in range(B):
            points = self[b]  # [N, 3]
            
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
        curvature = self.compute_curvature()  # 几何曲率计算函数
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







