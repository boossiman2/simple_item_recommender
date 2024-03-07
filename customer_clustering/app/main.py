import uvicorn
from fastapi import FastAPI, HTTPException, status, WebSocket
from starlette.websockets import WebSocketDisconnect

from model.cluster import KMeansModel, SpectralClusteringModel, DBSCANModel
from model.model import ClusterCustomerInDto, ClusterCustomerOutDto
from settings import settings

app = FastAPI()

@app.get("/")
async def read_root():
    return None

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        await websocket.close()


@app.websocket('/train_custom_cluster')
def train_cluster_customer(in_dto: ClusterCustomerInDto):
    # TODO: inDto 따라서 switch case
    cluster_model = KMeansModel('./data/', 'Online_Retail_preprocessed.csv')
    # TODO: os error 체크
    labels = cluster_model.train_model(n_cluster=in_dto.n_clusters)
    cluster_model.save_model(file_name=in_dto.user_name+"_kmeans_test")

    silhouette_score = cluster_model.evaluate_silhouette_score(labels)
    result = ClusterCustomerOutDto(silhouette_score=silhouette_score)

    return result


@app.websocket('/cluster_customers')
async def cluster_customers(in_dto: ClusterCustomerInDto, websocket: WebSocket) -> ClusterCustomerOutDto:
    await websocket.accept()
    # TODO: inDto 따라서 switch case
    cluster_model = KMeansModel('./data/', 'Online_Retail_preprocessed.csv')
    # load model fail
    if cluster_model.load_model(file_name=in_dto.user_name+"_kmeans_test") is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="False")
    labels = cluster_model.inference(n_cluster=in_dto.n_clusters)
    # inference fail
    if labels is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="False")

    silhouette_score = cluster_model.evaluate_silhouette_score(labels)
    result = ClusterCustomerOutDto(silhouette_score=silhouette_score)

    return result


if __name__ == '__main__':
    uvicorn.run(app, host=settings.conf_host, port=settings.conf_port)