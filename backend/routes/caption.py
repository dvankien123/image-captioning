from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/caption")
async def caption(image: UploadFile = File(...)):
    return {
        "filename": image.filename,
        "message": "Image received"
    }
