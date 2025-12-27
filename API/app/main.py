import io
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
import tensorflow as tf

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_path = "app/model/deepsign_model.keras"
        ml_models["deepsign"] = tf.keras.models.load_model(model_path)
        ml_models["labels"] = list("ABCDEFGHIKLMNOPQRSTUVWXY")
        print("Model loaded successfully.")
    except Exception as e:
        print("Error loading model:", e)
    yield
    ml_models.clear()


app = FastAPI(
    title="Deepsign API - Sign Language Recognition",
    description="Production ready API for recognizing Hand Sign",
    version="1.0.0",
    lifespan=lifespan
)  

def image_preprocessor(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.grayscale(img)
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        return img_array.astype(np.float32)
    except Exception as e:
        raise ValueError("Failed to Preprocess: ", str(e))
    
@app.get("/health", tags=["Monitoring"])
async def get_health():
    if "deepsign" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {"status": "ready", "model": "deepsign_model.keras"}