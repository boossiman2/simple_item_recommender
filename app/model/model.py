from pydantic import BaseModel


class ClusterCustomerOutDto(BaseModel):
    silhouette_score: float | None


class ClusterCustomerInDto(BaseModel):
    user_name: str
    cluster_name: str = 'kmeans'
    n_clusters: int = 2

class SimilarProductInDto(BaseModel):
    product_id: int
    # product_name: str
    # product_category_1: str
    # product_category_2: str

class SimilarProductOutDto(BaseModel):
    product_id: int
    product_name: str
    product_category_1: str
    product_category_2: str