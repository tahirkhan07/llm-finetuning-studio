from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, FileResponse
import asyncio
import os
from core.models.exporter import merge_adapter, convert_to_gguf, push_to_hub, list_merged_models

router = APIRouter()

class MergeRequest(BaseModel):
    adapter_path: str
    base_model_id: str
    output_dir: str

@router.post("/merge")
async def merge_model(req: MergeRequest):
    try:
        def on_prog(m): pass
        adapter_name = os.path.basename(os.path.dirname(req.adapter_path))
        full_output_dir = os.path.join(req.output_dir, adapter_name)
        result = await asyncio.to_thread(
            merge_adapter,
            req.adapter_path,
            full_output_dir,
            req.base_model_id or None,
            progress_fn=on_prog
        )
        return {"success": True, "output_dir": full_output_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class GGUFRequest(BaseModel):
    merged_model_dir: str
    quant_type: str

@router.post("/gguf")
async def make_gguf(req: GGUFRequest):
    try:
        def on_prog(m): pass
        model_name = os.path.basename(req.merged_model_dir)
        output_path = os.path.join(os.path.dirname(req.merged_model_dir), f"{model_name}-{req.quant_type}.gguf")
        result = await asyncio.to_thread(
            convert_to_gguf,
            req.merged_model_dir,
            output_path,
            req.quant_type,
            progress_fn=on_prog
        )
        return {"success": True, "gguf_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download")
async def download_file(file: str):
    if not os.path.exists(file):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file, filename=os.path.basename(file), media_type='application/octet-stream')

import tempfile
import shutil
from starlette.background import BackgroundTask

@router.get("/download_folder")
async def download_folder(path: str):
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail="Folder not found")
    
    # Create zip file in a temp directory
    temp_dir = tempfile.mkdtemp()
    zip_filename = os.path.basename(path.rstrip('/')) or "weights"
    zip_path = os.path.join(temp_dir, zip_filename)
    
    shutil.make_archive(zip_path, 'zip', path)
    
    final_zip = zip_path + ".zip"
    
    def cleanup():
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    return FileResponse(
        path=final_zip, 
        filename=f"{zip_filename}.zip", 
        media_type='application/zip',
        background=BackgroundTask(cleanup)
    )

@router.get("/merged")
async def get_merged_models():
    return {"models": await asyncio.to_thread(list_merged_models)}

class PushRequest(BaseModel):
    merged_model_dir: str
    repo_id: str
    private: bool = True

@router.post("/push")
async def push_model(req: PushRequest):
    try:
        def on_prog(m): pass
        result = await asyncio.to_thread(
            push_to_hub,
            model_dir=req.merged_model_dir,
            repo_id=req.repo_id,
            private=req.private,
            progress_fn=on_prog
        )
        return {"success": True, "url": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
