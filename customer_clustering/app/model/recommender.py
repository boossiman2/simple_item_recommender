import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Similarity():
    def calculate_cos_sim(self, vectors: np.ndarray) -> tuple:
        matrix = cosine_similarity(vectors, vectors)
        return matrix, matrix.shape

    def calculate_jaccard_sim(self):
        pass


class Candidate():
    def __init__(self, topk: int = 20):
        self.topk: int = topk

    def generate(self, matrix: np.ndarray):
        candidates = matrix.argsort()[:, ::-1][:, 1:self.topk+1]
        return candidates