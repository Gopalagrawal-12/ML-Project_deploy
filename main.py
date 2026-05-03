import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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
model = tf.keras.models.load_model("Alziemer_model.keras")

# Replace these with your actual training class names in the correct order
CLASS_NAMES = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Preprocess
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Inference
    predictions = model.predict(img_array)

    # Get the highest confidence score and label
    predicted_index = np.argmax(predictions[0])
    label = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    return {
        "prediction": label,
        "confidence": confidence,
        "raw_predictions": predictions.tolist()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
