import numpy as np

# 生成三维风场数据（示例数据）
def generate_3d_wind_field(x_size=30, y_size=30, z_size=30):
    x, y, z = np.mgrid[-5:5:x_size*1j, -5:5:y_size*1j, -5:5:z_size*1j]
    
    # 速度场函数（可修改为实际数据）
    u = 1.5 * np.sin(np.pi * x/5) * np.cos(np.pi * y/5) * np.cos(np.pi * z/5)
    v = -1.5 * np.cos(np.pi * x/5) * np.sin(np.pi * y/5) * np.cos(np.pi * z/5)
    w = 0.6 * np.cos(np.pi * x/5) * np.cos(np.pi * y/5) * np.sin(np.pi * z/5)
    return x, y, z, u, v, w