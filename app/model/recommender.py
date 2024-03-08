import numpy as np

class Candidate:
    def __init__(self, topk: int = 10):
        self.topk: int = topk

    def generate(self, matrix: np.ndarray):
        candidates = matrix.argsort()[:, ::-1][:, 1:self.topk+1]
        return candidates