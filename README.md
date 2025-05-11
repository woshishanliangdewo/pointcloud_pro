# WindPointCloud: 基于Transformer自监督学习的风场点云异常检测


​**NXU AlphaLab实验室**​ | [实验室主页](#) (待补充) | [论文链接](#) (待补充)

本项目提出了一种基于 ​**Transformer自监督机制**​ 的风场点云（LiDAR/Radar）异常检测方法，适用于机场大气湍流异常点识别等场景。

## 核心创新点
✅ ​**自监督预训练**​：采用Masked Point Modeling (MPM) 策略，无需人工标注即可学习点云特征  
✅ ​**多尺度Transformer**​：融合局部-全局点云几何特征，提升细小异常检测能力  
✅ ​**动态风场适应**​：通过时间序列建模处理动态风场中的噪声干扰  

## 快速开始

### 环境配置
```bash
git clone https://github.com/woshishanliangdewo/WindPointCloud.git
cd WindPointCloud
conda create -n windpc python=3.10
pip install -r requirements.txt
conda activate windpc
```