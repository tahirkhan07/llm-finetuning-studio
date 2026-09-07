from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
from app.state import state
from core.models.loader import load_model_for_training
from core.models.inspector import inspect_model
from core.models.architecture import get_lora_targets

router = APIRouter()

class LoadModelRequest(BaseModel):
    model_id: str
    quantization: str

@router.post("/load")
async def load_and_inspect(request: LoadModelRequest):
    try:
        quant = request.quantization
        quant_cfg = {"bits": 4, "dtype": "nf4"}
        if quant == "8-bit":
            quant_cfg = {"bits": 8}
        elif quant == "16-bit (LoRA)":
            quant_cfg = {"bits": 16, "dtype": "bf16"}
            
        state.clear_vram()
        model, tokenizer = await asyncio.to_thread(load_model_for_training, request.model_id, quant_cfg)
        state.model_id = request.model_id
        state.model = model
        state.tokenizer = tokenizer
        
        props = await asyncio.to_thread(inspect_model, model, tokenizer)
        
        lora_targets = get_lora_targets(model)
        state.lora_targets = lora_targets
        
        has_template = props.get("has_chat_template", False)
        
        return {
            "success": True,
            "message": f"Successfully loaded: {request.model_id}",
            "properties": props,
            "lora_targets": lora_targets,
            "has_chat_template": has_template
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
