from fastapi import APIRouter, UploadFile
from utils import file_resize
from deeplearning.main import run_analysis

router = APIRouter()

@router.post("/x_ray")
async def reset_password(file: UploadFile):
    image_path = await file_resize(file)
    return run_analysis(image_path)