from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel
from typing import List, Dict, Any
from app.state import state
from core.inference.loader import InferenceLoader
from core.inference.generator import Generator

router = APIRouter()

class LoadAdapterRequest(BaseModel):
    adapter_path: str

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    temperature: float = 0.7
    max_tokens: int = 256

@router.post("/load")
async def load_adapter(req: LoadAdapterRequest):
    if not state.model_id:
        raise HTTPException(status_code=400, detail="Base model ID not set. Load a model first.")
        
    try:
        path = None if req.adapter_path in ("None", "", None) else req.adapter_path
        state.clear_vram()
        m, t = await asyncio.to_thread(InferenceLoader.load, state.model_id, path)
        state.inference_model = m
        state.inference_tokenizer = t
        state.inference_adapter_path = path
        return {"success": True, "message": f"Loaded adapter: {req.adapter_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_generation(req: ChatRequest):
    if not state.inference_model:
        raise HTTPException(status_code=400, detail="No model loaded for inference.")
        
    gen = Generator(state.inference_model, state.inference_tokenizer)
    
    def generate():
        for chunk in gen.generate_stream(req.messages, max_new_tokens=req.max_tokens, temperature=req.temperature):
            yield chunk
            
    return StreamingResponse(generate(), media_type="text/plain")
