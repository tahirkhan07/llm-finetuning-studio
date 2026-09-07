from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from huggingface_hub import login, logout, whoami
import asyncio

router = APIRouter()

class LoginRequest(BaseModel):
    token: str

@router.post("/login")
async def do_login(req: LoginRequest):
    try:
        await asyncio.to_thread(login, token=req.token)
        return {"success": True, "message": "Logged into HuggingFace successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/logout")
async def do_logout():
    try:
        await asyncio.to_thread(logout)
        return {"success": True, "message": "Logged out of HuggingFace."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def do_check():
    try:
        user = await asyncio.to_thread(whoami)
        return {"success": True, "user": user["name"]}
    except:
        return {"success": False, "message": "Not logged in."}
