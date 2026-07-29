from fastapi import UploadFile
from PIL import Image
import io

async def read_image(file: UploadFile):

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))

    return image