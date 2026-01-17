from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routes import predict as predict_module
from PIL import Image
import io

def create_test_app():
    app = FastAPI()
    app.include_router(predict_module.router)
    return app

def test_predict_endpoint(monkeypatch):
    # Patch predict_recyclability so the test doesn't need the real model
    monkeypatch.setattr(predict_module, "predict_recyclability", lambda img: "recyclable")

    app = create_test_app()
    client = TestClient(app)

    # create a tiny in-memory JPEG image
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test.jpg", buf, "image/jpeg")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "test.jpg"
    assert data["prediction"] == "recyclable"