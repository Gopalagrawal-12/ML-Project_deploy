import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
app = FastAPI()

# 1. ALLOW CORS (Crucial for the HTML to talk to the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD MODEL
# Ensure "Alziemer_model.keras" is in the same folder as this script
from download import download_model

model = None

@app.on_event("startup")
def load_model_on_startup():
    global model
    try:
        download_model()
        model = tf.keras.models.load_model("model/Alziemer_model.keras")
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Error loading model:", str(e))
        model = None
# Replace these with your actual training class names in the correct order
CLASS_NAMES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded yet"}

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except:
        return {"error": "Invalid image"}

    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    import asyncio
    predictions = await asyncio.to_thread(model.predict, img_array)

    predicted_index = np.argmax(predictions[0])
    label = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    return {
        "prediction": label,
        "confidence": confidence,
        "raw_predictions": predictions.tolist()
    }
    
@app.get("/")
def health():
    return {
        "status": "ready" if model is not None else "loading"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
