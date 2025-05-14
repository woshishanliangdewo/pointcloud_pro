import torch.nn as nn
# 未完成，只是一个模型
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
    
