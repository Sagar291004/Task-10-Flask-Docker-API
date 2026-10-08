from flask import Flask, request, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")


@app.route("/")
def home():
    return jsonify({
        "message": "Flask Deep Learning API is running successfully!"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    image_path = "temp_image.jpg"
    image.save(image_path)

    # Load and preprocess image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    predictions = model.predict(img_array)

    results = decode_predictions(predictions, top=3)[0]

    output = []

    for _, label, confidence in results:
        output.append({
            "class": label,
            "confidence": float(confidence)
        })

    # Remove temporary image
    if os.path.exists(image_path):
        os.remove(image_path)

    return jsonify({
        "predictions": output
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)