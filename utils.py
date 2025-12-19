from PIL import Image
from fastapi import UploadFile
import uuid
import os
from PIL import Image
import io

SAVE_DIR = 'D:/healthcare/healthcare/fastAPI/static/results/'
IMAGE_SIZE = 320

async def file_resize(file: UploadFile):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents))

    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

    extension = os.path.splitext(file.filename)[1]

    uuid_file_name = uuid.uuid4()
    save_path = os.path.join(SAVE_DIR, f"{uuid_file_name}{extension}")

    resized_image.convert("RGB").save(save_path)

    return save_path, uuid_file_name