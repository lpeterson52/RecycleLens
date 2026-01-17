from fastapi import APIRouter, UploadFile, File
from app.models.classifier import predict_recyclability
from app.utils.image_utils import load_image_from_bytes
router = APIRouter()

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Placeholder for prediction logic
    image_bytes = await file.read()
    prediction = predict_recyclability(load_image_from_bytes(image_bytes))
    return {"filename": file.filename, "prediction": prediction}