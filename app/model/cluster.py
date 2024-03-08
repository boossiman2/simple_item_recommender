import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score
from abc import ABC, abstractmethod

DATA_PATH = '../data/'
MODEL_PATH = './model/models/'

class ClusterModel(ABC):
    def load_dataset(self, data_path, file_name) -> np.ndarray:
        return pd.read_csv(data_path + file_name)[
            ['Recency_log_scaled', 'Frequency_log_scaled', 'Frequency_log_scaled']
        ]

    @abstractmethod
    def train_model(self, **kwargs) -> None:
        pass

    @abstractmethod
    def load_model(self, **kwargs) -> None:
        pass

    @abstractmethod
    def inference(self, **kwargs) -> np.ndarray:
        pass

    def evaluate_silhouette_score(self, labels: np.ndarray) -> float:
        return silhouette_score(self.x, labels).astype(float)

    def agglomerative_clustering(self, n_clusters: int = 2) -> np.ndarray:
        model = AgglomerativeClustering(n_clusters=n_clusters, random_state=self.random_statem)
        return model.fit_predict(self.x)


class KMeansModel(ClusterModel):
    def __init__(self, data_path: str, file_name: str):
        self.model = None
        self.random_state = 0
        self.x = super().load_dataset(data_path, file_name)

    def train_model(self, n_cluster: int = 2) -> np.ndarray:
        self.model = KMeans(n_clusters=n_cluster, random_state=self.random_state)
        return self.model.fit_predict(self.x)

    def save_model(self, file_name: str) -> None:
        joblib.dump(self.model, MODEL_PATH + "KMeans_" + file_name + ".pkl")

    def load_model(self, file_name: str) -> None:
        self.model = joblib.load(MODEL_PATH + "KMeans_" + file_name + ".pkl")

    def inference(self) -> np.ndarray:
        return self.model.predict(self.x)


class SpectralClusteringModel(ClusterModel):
    def __init__(self, data_path: str, file_name: str):
        self.model = None
        self.random_state = 0
        self.x = super().load_dataset(data_path, file_name)

    def train_model(self, n_cluster: int = 2) -> np.ndarray:
        self.model = SpectralClustering(n_clusters=n_cluster, random_state=self.random_state)
        return self.model.fit_predict(self.x)

    def save_model(self, file_name: str) -> None:
        joblib.dump(self.model, MODEL_PATH + "SpectralClustering_" + file_name + ".pkl")

    def load_model(self, file_name: str) -> None:
        self.model = joblib.load(MODEL_PATH + "SpectralClustering_" + file_name + ".pkl")

    def inference(self) -> np.ndarray:
        return self.model.predict(self.x)


class DBSCANModel(ClusterModel):
    def __init__(self, data_path: str, file_name: str):
        self.model = None
        self.random_state = 0
        self.x = super().load_dataset(data_path, file_name)
        # self.metric = ['cosine', 'cityblock', 'cosine', 'euclidean', 'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean']

    def train_model(self, eps: float = 0.5, min_samples: int = 5, metric: str = 'cosine') -> np.ndarray:
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, random_state=self.random_state)
        return self.model.fit_predict(self.x)

    def save_model(self, file_name: str) -> None:
        joblib.dump(self.model, MODEL_PATH + "SpectralClustering_" + file_name + ".pkl")

    def load_model(self, file_name: str) -> None:
        self.model = joblib.load(MODEL_PATH + "SpectralClustering_" + file_name + ".pkl")

    def inference(self) -> np.ndarray:
        return self.model.predict(self.x)

    def dbscan(self, eps: float = 0.5, min_samples: int = 5, metric: str = 'euclidean') -> np.ndarray:
        model = DBSCAN(eps, min_samples=min_samples, metric=metric)
        return model.fit_predict(self.x)

'''
if __name__ == '__main__':
    data_path = './data/'
    file_name = 'Online_Retail_preprocessed.csv'
    clusterModel = ClusterModel(data_path=data_path, file_name=file_name)
    labels = clusterModel.kmeans()
    s_score = clusterModel.evaluate_silhouette_score(labels)
    print(f'{s_score:.3f}')
'''