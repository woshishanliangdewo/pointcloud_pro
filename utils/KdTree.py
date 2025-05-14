import numpy as np
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
