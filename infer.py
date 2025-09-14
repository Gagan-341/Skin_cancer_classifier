import os
import json
import tensorflow as tf
import numpy as np
from utils import load_image_for_inference

def main():
    # --- Auto-detect model path ---
    model_path = "models/final_model.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

    model = tf.keras.models.load_model(model_path, compile=False)

    # --- Auto-detect class names ---
    class_names_path = os.path.join(os.path.dirname(model_path), "class_names.json")
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file not found at {class_names_path}")

    with open(class_names_path, "r") as f:
        class_names = json.load(f)

    print("Classes in order:", class_names)

    # --- Pick an image for testing ---
    test_image = "data/processed/vasc/ISIC_0034214.jpg"  # replace with your test image
    if not os.path.exists(test_image):
        raise FileNotFoundError(f"Test image not found: {test_image}")

    arr = load_image_for_inference(test_image, 224)

    # --- Predict ---
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx]) * 100

    print(f"\nPredicted class: {class_names[idx]} ({confidence:.2f}% confidence)")

    # --- Full class probabilities ---
    print("\nFull prediction probabilities:")
    for i, cls in enumerate(class_names):
        print(f"{cls}: {preds[i]*100:.2f}%")


if __name__ == "__main__":
    main()
