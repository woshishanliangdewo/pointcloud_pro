import torch
from model.GeoMotionSSL import GeoMotionSSL,SSLMultiLoss
from torch.utils.data import DataLoader
from dataset.windfieldDataset import WindFieldSSL
from utils.argument_batch import augment_batch

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


if __name__ == '__main__':
    train_ssl()