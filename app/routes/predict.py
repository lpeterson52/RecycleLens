from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.models.classifier import predict_recyclability
from app.utils.image_utils import load_image_from_bytes

router = APIRouter()

# 5 MB max file size
MAX_FILE_SIZE = 5 * 1024 * 1024


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # read file bytes (fastapi's UploadFile provides async read)
    image_bytes = await file.read()

    # enforce size limit
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {MAX_FILE_SIZE} bytes.",
        )

    prediction = predict_recyclability(load_image_from_bytes(image_bytes))
    return {"filename": file.filename, "prediction": prediction}