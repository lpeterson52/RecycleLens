from pathlib import Path
from app.utils.image_utils import load_image_from_bytes
import numpy as np
import os
import requests
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "best.pt"
# Allow overriding model location via env vars (for Railway persistent volume)
# - MODEL_PATH: full path to the model file
# - MODEL_DIR: directory containing the model file (joined with MODEL_NAME)
env_model_path = os.environ.get("MODEL_PATH")
env_model_dir = os.environ.get("MODEL_DIR")
if env_model_path:
    MODEL_PATH = Path(env_model_path)
elif env_model_dir:
    MODEL_PATH = Path(env_model_dir) / MODEL_NAME
else:
    MODEL_PATH = BASE_DIR / MODEL_NAME

model = None

def load_model() -> None:
    """Load the YOLO model from the specified path."""
    global model
    # If a mounted persistent volume is provided (MODEL_PATH / MODEL_DIR), prefer it.
    # If the file is missing, fall back to downloading from MODEL_URL.
    model_url = os.environ.get("MODEL_URL")
    if not MODEL_PATH.exists():
        # If the parent dir exists and is writable, create the directory structure.
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if model_url:
            _download_model(model_url, MODEL_PATH)
        else:
            raise RuntimeError(f"Model not found at {MODEL_PATH} and MODEL_URL is not set")

    # import ultralytics lazily to avoid heavy imports at module import time
    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)


def _download_model(url: str, dest: Path, timeout: int = 60) -> None:
    """Download a file from `url` to `dest` streaming to disk.

    Raises requests.HTTPError on bad status.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

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
    
    results = model.predict(image, device="cpu", conf=0.25, save=False)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return "No objects detected"
    result = results[0]
    box = result.boxes[0]
    cls_id = int(box.cls)
    label = result.names[cls_id]
    return label
