import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Similarity:
    def calculate_cos_sim(self, vectors: np.ndarray) -> tuple:
        matrix = cosine_similarity(vectors, vectors)
        return matrix, matrix.shape

    def calculate_jaccard_sim(self) -> object:
        pass
