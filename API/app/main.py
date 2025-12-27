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