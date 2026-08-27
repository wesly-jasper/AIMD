from fastapi import APIRouter,UploadFile,File,HTTPException

from app.services.ingestion.media_ingestion import MediaIngestionService


router=APIRouter(
    prefix="/media",
    tags=["Media"]
)

service=MediaIngestionService()


media_store = {}

@router.post("/upload")
async def upload_media(file:UploadFile=File(...)):

    try:
        result=await service.ingest(file)
        media_store[result["media_id"]] = result
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


@router.get("/{media_id}")
def get_media(media_id: str):
    if media_id in media_store:
        return media_store[media_id]
    
    # Fallback to checking upload directory
    matches = list(service.upload_dir.glob(f"{media_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Media not found")
    
    file_path = matches[0]
    metadata = service.metadata_extractor.extract(file_path)
    fingerprint = service.fingerprint_service.generate(file_path)
    
    return {
        "media_id": media_id,
        "stored_filename": file_path.name,
        "file_path": str(file_path),
        "metadata": metadata,
        "fingerprint": fingerprint,
        "status": "STORED"
    }