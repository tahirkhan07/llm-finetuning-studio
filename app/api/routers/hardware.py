from fastapi import APIRouter, HTTPException
import asyncio
from app.state import state
from core.hardware.detector import detect_hardware
from core.hardware.planner import recommend_config

router = APIRouter()

@router.get("/detect")
async def detect():
    hw = await asyncio.to_thread(detect_hardware)
    state.hardware = hw
    return hw.__dict__

@router.get("/recommend")
async def recommend():
    try:
        if not state.hardware:
            return {"error": "Run diagnostics first"}
        
        params = getattr(state, "model_params_billions", None) or 7.0
            
        cfg = await asyncio.to_thread(recommend_config, state.hardware, model_params_billions=params)
        return cfg
    except Exception as e:
        return {"error": str(e)}
