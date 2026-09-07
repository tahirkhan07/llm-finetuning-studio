from fastapi import APIRouter
import asyncio
from core.experiments.tracker import ExperimentTracker

router = APIRouter()
tracker = ExperimentTracker()

@router.get("/")
async def list_experiments():
    return {"experiments": await asyncio.to_thread(tracker.list_experiments)}

@router.delete("/")
async def delete_all():
    await asyncio.to_thread(tracker.delete_all_experiments)
    return {"success": True}

@router.delete("/{exp_id}")
async def delete_one(exp_id: str):
    import os, shutil
    # Basic logic to delete an adapter directory based on ID
    path = os.path.join("outputs", exp_id)
    if os.path.exists(path):
        shutil.rmtree(path)
    tracker._load_or_create_db()
    tracker.db = tracker.db[tracker.db["Experiment ID"] != exp_id]
    tracker._save_db()
    return {"success": True}
