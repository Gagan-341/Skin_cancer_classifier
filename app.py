import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from utils import load_image_for_inference

app = Flask(__name__)

# --- Auto-load model and class names ---
MODEL_PATH = "models/final_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train first.")
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# Optional descriptions
descriptions = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma (Bowen’s disease). Precancerous lesions caused by sun exposure.",
    "bcc": "Basal Cell Carcinoma. A common, slow-growing skin cancer, rarely spreads but needs treatment.",
    "bkl": "Benign Keratosis. Non-cancerous lesions such as seborrheic keratoses, solar lentigines, lichen-planus–like keratoses.",
    "df": "Dermatofibroma. A benign skin lesion, usually firm nodules on the skin.",
    "mel": "Melanoma. The most serious type of skin cancer, can spread quickly if not treated early.",
    "nv": "Melanocytic Nevi (Moles). Common benign moles, but some can develop into melanoma.",
    "vasc": "Vascular lesions. Includes angiomas, angiokeratomas, pyogenic granulomas, hemorrhage-related skin issues."
}

IMG_SIZE = 224

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file selected")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        os.makedirs("uploads", exist_ok=True)
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        arr = load_image_for_inference(filepath, IMG_SIZE)
        preds = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))

        predicted_class = class_names[idx]
        probability = float(preds[idx])
        description = descriptions.get(predicted_class, "No description available.")
        image_url = f"/uploads/{file.filename}"

        return render_template(
            "result.html",
            class_name=predicted_class,
            probability=probability,
            description=description,
            image_url=image_url
        )
    return render_template("index.html", prediction=None)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory("uploads", filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
