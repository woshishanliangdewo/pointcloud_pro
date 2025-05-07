import numpy as np

def convolve3d(matrix, kernel):
    """手动实现三维卷积（边界处理采用镜像填充）"""
    pad = 1
    padded = np.pad(matrix, pad, mode='reflect')
    result = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                result[i,j,k] = np.sum(padded[i:i+3, j:j+3, k:k+3] * kernel)
    return result