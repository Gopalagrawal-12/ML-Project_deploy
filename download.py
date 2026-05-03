https://github.com/Gopalagrawal-12/ML-Project_deploy/releases/download/v1.0.0/Alziemer_model.keras
import os
import requests

MODEL_URL = "https://github.com/Gopalagrawal-12/ML-Project_deploy/releases/download/v1.0.0/Alziemer_model.keras"
MODEL_PATH = "model/Alziemer_model.keras"

def download_model():
    os.makedirs("model", exist_ok=True)  # ✅ create folder if not exists

    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Download complete!")
    else:
        print("Model already exists.")

if __name__ == "__main__":
    download_model()
