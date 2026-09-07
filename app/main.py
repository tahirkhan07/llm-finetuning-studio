import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import models, datasets, hardware, training, evaluation, inference, experiments, export, settings

app = FastAPI(title="LLM Fine-Tuning Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(hardware.router, prefix="/api/hardware", tags=["hardware"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(evaluation.router, prefix="/api/evaluate", tags=["evaluation"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
app.include_router(export.router, prefix="/api/export", tags=["export"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {"message": "Frontend not found, API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
