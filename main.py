from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import predict, history, admin
from utils.model_loader import get_model
import uvicorn
import logging

app = FastAPI(
    title="Cattle Disease Detection API",
    description="""
## Cattle Disease Detection — REST API

Upload an image of your cattle and receive an AI-powered disease diagnosis.

### Features
- **Disease Prediction** — detects FMD, Lumpy Skin, Mastitis, Healthy
- **PDF Reports** — downloadable diagnosis reports
- **History** — past prediction records
- **Model Management** — switch between model versions (admin only)

### Authentication
Admin endpoints require an `X-Admin-Token` header.
""",
    version="1.0.0",
    contact={"name": "Cattle Health Check Kenya"},
)

# Allow any frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(predict.router,  prefix="/api", tags=["Prediction"])
app.include_router(history.router,  prefix="/api", tags=["History"])
app.include_router(admin.router,    prefix="/api/admin", tags=["Admin"])


@app.on_event("startup")
def load_model_on_startup():
    """
    Attempt to load the TensorFlow model once at startup so first requests are fast.
    If loading fails (e.g. out of memory), log the error but still allow the app to start,
    so that health endpoints work and we can debug from logs.
    """
    logger = logging.getLogger("uvicorn.error")
    try:
        get_model()
        logger.info("Model loaded successfully at startup.")
    except Exception as exc:
        logger.error("Model failed to load at startup; will retry on first request.", exc_info=exc)


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Cattle Disease Detection API",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/api/health", tags=["Health"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)