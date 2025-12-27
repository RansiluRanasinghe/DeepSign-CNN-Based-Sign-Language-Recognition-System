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
        ml_models["labels"] = [
            "A","B","C","D","E","F","G","H","I",
            "K","L","M","N","O","P","Q","R","S",
            "T","U","V","W","X","Y"
        ]

        print(ml_models["labels"])

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
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((28, 28))
        img_array = np.asarray(img, dtype=np.float32)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {str(e)}")


@app.get("/health", tags=["Monitoring"])
async def get_health():
    if "deepsign" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model": "deepsign_model.keras"}


@app.post("/predict", tags=["inference"])
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    content = await file.read()
    processed_image = image_preprocessor(content)

    prediction = ml_models["deepsign"].predict(processed_image, verbose=0)

    predict_idx = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))

    return {
        "prediction": ml_models["labels"][predict_idx],
        "confidence": round(confidence, 4),
        "metadata": {
            "input_shape": processed_image.shape,
            "model_name": "DeeSign_V1"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
