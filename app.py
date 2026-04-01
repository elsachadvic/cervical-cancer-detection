from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import sys
import os
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

template_folder = os.path.join(base_path, 'templates')
static_folder = os.path.join(base_path, 'static')

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

# ---- Load ResNet model ----
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 5)

model.load_state_dict(torch.load("model_resnet.pth", map_location=torch.device("cpu")))
model.eval()

# ---- Image transformations (same as training) ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ---- Class labels (replace with your folder names) ----
classes = [
    'Dyskeratotic',
    'Koilocytotic',
    'Metaplastic',
    'Parabasal',
    'Superficial-Intermediate'
]

# ---- Prediction function ----
def predict_image(img_path):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, "static")

    # ---- IMAGE PROCESSING ----
    original_image = Image.open(img_path).convert('RGB')

    # save original
    original_path = os.path.join(static_dir, "original.png")
    original_image.save(original_path)

    # resize for model
    resized_image = original_image.resize((224,224))

    # depth
    gray = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2GRAY)
    depth_path = os.path.join(static_dir, "depth.png")
    cv2.imwrite(depth_path, gray)

    # processed
    processed = cv2.GaussianBlur(gray,(5,5),0)
    processed_path = os.path.join(static_dir, "processed.png")
    cv2.imwrite(processed_path, processed)

    # segmented
    _, segmented = cv2.threshold(processed,127,255,cv2.THRESH_BINARY)
    segmented_path = os.path.join(static_dir, "segmented.png")
    cv2.imwrite(segmented_path, segmented)

    # 🔥 ---- ADD MODEL PREDICTION HERE ---- 🔥

    image_tensor = transform(resized_image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    predicted_class = classes[predicted.item()]
    print("Predicted class:", predicted_class)  # debug

    # classification logic
    if predicted_class in ["Superficial-Intermediate", "Parabasal"]:
        final_result = "Normal Cell"

    elif predicted_class in ["Metaplastic", "Koilocytotic"]:
        final_result = "Abnormal Cell"

    elif predicted_class == "Dyskeratotic":
        final_result = "Cancer Cell"

    # ---- FINAL RETURN ----
    return final_result, predicted_class, "original.png", "depth.png", "processed.png", "segmented.png"

    with torch.no_grad():
     output = model(image_tensor)
    _, predicted = torch.max(output, 1)

    predicted_class = classes[predicted.item()]

    if predicted_class in ["Superficial-Intermediate", "Parabasal"]:
      final_result = "Normal Cell"
    elif predicted_class in ["Metaplastic", "Koilocytotic"]:
      final_result = "Abnormal Cell"

    elif predicted_class == "Dyskeratotic":
      final_result = "Cancer Cell"

    return final_result, predicted_class, "original.png", "depth.png", "processed.png", "segmented.png"
    

from flask import url_for

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, "static")

    filepath = os.path.join(static_dir, file.filename)
    file.save(filepath)

    result, cell_type, original, depth, processed, segmented = predict_image(filepath)

    return render_template(
        "index.html",
        prediction_class=result,
        cell_type=cell_type,
        confidence=90,
        original_image=original,
        depth_image=depth,
        processed_image=processed,
        segmented_image=segmented
    )

if __name__ == "__main__":
    app.run()
    