from fastapi import FastAPI, UploadFile, File
from backend.predict import read_image
from inference.predict import predict_image

app = FastAPI(
    title="AWS Image Classifier",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "AWS Image Classifier API Running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = await read_image(file)

    result = predict_image(image)

    return {
        "filename": file.filename,
        "width": image.size[0],
        "height": image.size[1],
        **result
    }