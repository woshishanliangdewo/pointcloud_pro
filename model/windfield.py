import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from utils.Convutil import convolve3d
from data.windfieldDataset import generate_wind_field
class WindFieldModel:

    def horn_schunck_3d(flow_sequence, alpha=0.2, iterations=100):
        """
        三维Horn-Schunck光流法计算平滑性损失
        :param flow_sequence: 输入光流序列 (T, X, Y, Z, 3)
        :param alpha: 平滑性权重系数
        :param iterations: 迭代次数
        :return: 每个空间点的损失矩阵 (X, Y, Z)
        """
        T, X, Y, Z, _ = flow_sequence.shape
        u = flow_sequence[..., 0]
        v = flow_sequence[..., 1]
        w = flow_sequence[..., 2]
        
        # 计算时空梯度
        grad_x = np.gradient(u, axis=1)
        grad_y = np.gradient(u, axis=2)
        grad_z = np.gradient(u, axis=3)
        grad_t = np.gradient(u, axis=0)
        
        # 初始化光流场
        u_avg = np.zeros_like(u)
        v_avg = np.zeros_like(v)
        w_avg = np.zeros_like(w)
        
        # 迭代求解
        for _ in range(iterations):
            # 计算邻域平均（三维卷积核）
            kernel = np.ones((3,3,3)) / 27
            for t in range(T):
                u_avg[t] = convolve3d(u[t], kernel)
                v_avg[t] = convolve3d(v[t], kernel)
                w_avg[t] = convolve3d(w[t], kernel)
            # 更新公式（三维扩展）
            denominator = (alpha ** 2 + grad_x**2 + grad_y **2 + grad_z **2)
            u = u_avg - grad_x * (grad_x * u_avg + grad_y * v_avg + grad_z * w_avg + grad_t) / denominator
            v = v_avg - grad_y * (grad_x * u_avg + grad_y * v_avg + grad_z * w_avg + grad_t) / denominator
            w = w_avg - grad_z * (grad_x * u_avg + grad_y * v_avg + grad_z * w_avg + grad_t) / denominator
        # 计算最终损失（数据项 + 平滑项）
        data_term = (grad_x*u + grad_y*v + grad_z*w + grad_t) **2
        smooth_term = alpha ** 2 * (np.gradient(u, axis=1) ** 2 + 
                                np.gradient(v, axis=2) ** 2 + 
                                np.gradient(w, axis=3) ** 2)
        return np.mean(data_term + smooth_term, axis=0)  # 沿时间轴平均



    def multi_frame_accumulation(loss_sequence, window_size=5):
        """
        多帧损失累积（滑动窗口平均）
        对各个窗口进行平均，然后进行归一化得到一个损失矩阵

        :param loss_sequence: 各帧损失矩阵列表
        :param window_size: 滑动窗口大小
        :return: 累积后的增强损失矩阵
        """
        accumulated = np.zeros_like(loss_sequence[0])
        for i in range(len(loss_sequence)):
            start = max(0, i - window_size + 1)
            window = loss_sequence[start:i+1]
            accumulated += np.mean(window, axis=0)
        return accumulated / len(loss_sequence)


    if __name__ == "__main__":
        # 生成模拟数据
        wind_data = generate_wind_field(shape=(32, 32, 32), frames=20)
        
        # 计算各帧光流损失
        loss_sequence = []
        for t in tqdm(range(1, wind_data.shape[0])):
            flow_pair = wind_data[t-1:t+1]  # 两帧计算光流
            loss = horn_schunck_3d(flow_pair, iterations=50)
            loss_sequence.append(loss)
        
        # 多帧累积增强
        accumulated_loss = multi_frame_accumulation(loss_sequence)
        
        # 异常检测（基于动态阈值）
        mean_loss = np.mean(accumulated_loss)
        std_loss = np.std(accumulated_loss)
        anomaly_mask = accumulated_loss > mean_loss + 3*std_loss
        
        print(f"异常区域占比: {np.mean(anomaly_mask)*100:.2f}%")
