from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
from typing import Optional
from app.state import state
from core.datasets.loader import load_dataset
from core.datasets.mapper import map_to_canonical, _is_preformatted
from core.datasets.validator import validate_dataset

router = APIRouter()

class LoadDatasetRequest(BaseModel):
    source: str

class MapDatasetRequest(BaseModel):
    instruction_col: str
    input_col: Optional[str] = None
    output_col: str
    system_prompt: str = "You are a helpful AI assistant."

def _guess_column(cols, candidates, fallback_index=None):
    for c in candidates:
        if c in cols:
            return c
    if fallback_index is not None and len(cols) > fallback_index:
        return cols[fallback_index]
    return None

@router.post("/load")
async def load_ds(request: LoadDatasetRequest):
    try:
        ds = await asyncio.to_thread(load_dataset, request.source)
        state.raw_dataset = ds
        cols = ds.column_names if isinstance(ds.column_names, list) else list(ds.features.keys())
        
        if _is_preformatted(ds):
            return {
                "success": True,
                "is_preformatted": True,
                "message": f"Loaded dataset with {len(ds)} rows. Auto-detected pre-formatted chat dataset!",
                "columns": cols
            }
            
        instr_guess = _guess_column(cols, ["instruction", "question", "prompt", "input", "text", "quote", "sentence"], 0)
        in_guess = _guess_column(cols, ["input", "context"])
        if in_guess == instr_guess:
            in_guess = None
        out_guess = _guess_column(cols, ["output", "response", "answer", "completion", "tags", "label", "target"], -1)
        
        preview_lines = []
        for i in range(min(3, len(ds))):
            row = ds[i]
            preview_lines.append({k: repr(str(v)[:80]) for k, v in row.items()})
            
        return {
            "success": True,
            "is_preformatted": False,
            "message": f"Loaded dataset with {len(ds)} rows.",
            "columns": cols,
            "guesses": {
                "instruction": instr_guess,
                "input": in_guess,
                "output": out_guess
            },
            "preview": preview_lines
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/map")
async def map_ds(request: MapDatasetRequest):
    if state.raw_dataset is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
        
    try:
        inp_col = None if request.input_col in ("None", "", None) else request.input_col
        canonical = await asyncio.to_thread(
            map_to_canonical,
            state.raw_dataset, 
            request.instruction_col, 
            inp_col, 
            request.output_col, 
            request.system_prompt
        )
        
        report = await asyncio.to_thread(validate_dataset, canonical, state.tokenizer, max_length=2048)
        
        bad_indices = set(report.invalid_indices + report.duplicate_indices + report.too_long_indices)
        
        if bad_indices:
            clean = canonical.select([i for i in range(len(canonical)) if i not in bad_indices])
            state.canonical_dataset = clean
        else:
            clean = canonical
            state.canonical_dataset = canonical
            
        return {
            "success": True,
            "message": f"Dataset mapped & cleaned! {report.valid_samples} samples ready for training.",
            "report": report.model_dump()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
