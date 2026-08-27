from fastapi import FastAPI
from app.api.v1.endpoints.media import router as media_router
from app.api.v1.endpoints.detection import router as detection_router
from app.api.v1.endpoints.investigation import router as investigation_router

app=FastAPI(
    title="AIMD",
    version="1.0.0"
)

app.include_router(
    media_router,
    prefix="/api/v1"
)

app.include_router(
    detection_router,
    prefix="/api/v1"
)

app.include_router(
    investigation_router,
    prefix="/api/v1"
)


@app.get("/health")
@app.get("/api/v1/health")
def health():
    return {
        "status":"ok"
    }