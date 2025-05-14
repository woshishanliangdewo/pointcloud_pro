import torch
import numpy as np

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
    
    Ry = torch.tensor([[cos_a[1], 0, sin_a[1]],
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
