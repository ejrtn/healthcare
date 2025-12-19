from fastapi import APIRouter, UploadFile
from utils import file_resize
from deeplearning.main import run_analysis

router = APIRouter()

@router.post("/analyze")
async def reset_password(file: UploadFile):
    image_path,uuid_file_name = await file_resize(file)
    return run_analysis(image_path,uuid_file_name)