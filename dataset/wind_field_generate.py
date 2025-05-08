import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


# 生成三维风场数据（示例数据）
def generate_3d_wind_field(x_size=30, y_size=30, z_size=30):
    x, y, z = np.mgrid[-5:5:x_size*1j, -5:5:y_size*1j, -5:5:z_size*1j]
    
    # 速度场函数（可修改为实际数据）
    u = 1.5 * np.sin(np.pi * x/5) * np.cos(np.pi * y/5) * np.cos(np.pi * z/5)
    v = -1.5 * np.cos(np.pi * x/5) * np.sin(np.pi * y/5) * np.cos(np.pi * z/5)
    w = 0.6 * np.cos(np.pi * x/5) * np.cos(np.pi * y/5) * np.sin(np.pi * z/5)
    return x, y, z, u, v, w


def generate_wind_field(shape=(64, 64, 64), frames=10, anomaly_ratio=0.02):
    """生成带异常的三维模拟风场数据"""
    u = np.zeros((frames,) + shape)
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    
    # 生成基础风场（高斯平滑）
    for t in range(frames):
        u[t] = gaussian_filter(np.random.randn(*shape), sigma=3)
        v[t] = gaussian_filter(np.random.randn(*shape), sigma=3)
        w[t] = gaussian_filter(np.random.randn(*shape), sigma=3)
    
    # 添加随机异常（突发性速度变化）
    anomalies = np.random.rand(*u.shape) < anomaly_ratio
    u[anomalies] += np.random.normal(scale=5, size=anomalies.sum())
    v[anomalies] += np.random.normal(scale=5, size=anomalies.sum())
    w[anomalies] += np.random.normal(scale=5, size=anomalies.sum())
    
    return np.stack([u, v, w], axis=-1)  # shape: (T, X, Y, Z, 3)
