from ultralytics import YOLO
from pathlib import Path
from app.utils.image_utils import load_image_from_bytes
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolov8n-waste-12cls-best.pt"

model = None

def load_model() -> None:
    """Load the YOLO model from the specified path."""
    global model
    model = YOLO(MODEL_PATH)

def predict_recyclability(image: np.array) -> str:
    """
    Docstring for predict_recyclability
    
    :param image: Image to be classified
    :type image: np.array
    :return: Returns a string indicating recyclability
    :rtype: str
    """
    global model
    if model is None:
        raise ValueError("Model is not loaded. Call load_model() before prediction.")
    results = model.predict(image, conf=0.25, save=False)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return "No objects detected"
    result = results[0]
    box = result.boxes[0]
    cls_id = int(box.cls)
    label = result.names[cls_id]
    return label
