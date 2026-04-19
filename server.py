import os
import io
import base64
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Flask, request, jsonify
from flask_cors import CORS

# -------------------------------
# Flask setup
# -------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------
# Config
# -------------------------------
DATA_DIR = Path("data/Training")
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE = 224
MODEL_PATH = Path("outputs/dl/inference_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -------------------------------
# Model Definition
# -------------------------------
class ResNet18MC(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)

# -------------------------------
# Load Model
# -------------------------------
print("Loading model...")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = ResNet18MC(num_classes=4, dropout_rate=0.3)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print(f"Model loaded successfully. Classes: {CLASSES}")

# -------------------------------
# Monte Carlo Dropout Prediction
# -------------------------------
def mc_predict(model, image_tensor, n_passes=20):
    model.eval()

    # Enable dropout during inference
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    image_tensor = image_tensor.to(DEVICE)

    all_probs = []

    with torch.no_grad():
        for _ in range(n_passes):
            outputs = model(image_tensor.unsqueeze(0))
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.array(all_probs)

    mean_probs = all_probs.mean(axis=0)[0]
    std_probs = all_probs.std(axis=0)[0].mean()

    entropy = -np.sum(
        mean_probs * np.log(np.clip(mean_probs, 1e-10, 1.0))
    )

    return mean_probs, std_probs, entropy

# -------------------------------
# Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = data["image"]

        # Remove base64 header if present
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_tensor = transform(image)

        probs, std, entropy = mc_predict(
            model,
            image_tensor,
            n_passes=20,
        )

        pred_idx = int(np.argmax(probs))

        # safer class retrieval
        model_classes = checkpoint.get("classes", CLASSES)
        predicted_class = model_classes[pred_idx]

        confidence = float(probs[pred_idx])

        return jsonify(
            {
                "predictedClass": predicted_class,
                "confidence": confidence,
                "entropy": float(entropy),
                "std": float(std),
                "probabilities": probs.tolist(),
            }
        )

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------
# Health Check
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": "ResNet-18",
            "classes": CLASSES,
        }
    )

# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)