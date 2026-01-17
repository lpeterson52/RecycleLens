from fastapi import FastAPI
from app.routes import predict
app = FastAPI(title="Recycle Detector API", version="1.0.0")
app.include_router(predict.router)
@app.on_event("startup")
async def startup_event():
    """Event handler for application startup."""
    from app.models.classifier import load_model
    load_model()

@app.get("/")
async def read_root():
    """Root endpoint returning a welcome message."""
    return {"Hello": "World"}

