import os
import io
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
CORS(app)  # Allows your frontend (Vercel) to communicate with this backend

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
# U-Net for Vegetation Segmentation
veg_model = tf.keras.models.load_model('models/vegetation_model.h5')

# YOLOv8 for Soil Classification
# Using 'r' before the string to handle Windows backslashes correctly
soil_model = YOLO(r'models\best (1).pt')

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def preprocess_veg(img_bytes):
    """
    Preprocesses the image for the U-Net model.
    Resizes to 256x256 and normalizes pixel values.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = np.array(img)
    # Resize and normalize to match training data
    img_resized = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
    return np.expand_dims(img_resized, axis=0)

# ---------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------

@app.route('/')
def home():
    """Root endpoint to verify the server is running."""
    return "AI Analysis Server is Online. Use the web interface to upload images."

@app.route('/predict/vegetation', methods=['POST'])
def predict_veg():
    """Endpoint for Vegetation Segmentation."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file'].read()
        input_tensor = preprocess_veg(file)
        
        # Run prediction
        pred = veg_model.predict(input_tensor)[0]
        
        # Threshold the sigmoid output (0.5) to get binary mask
        mask = (pred > 0.5).astype(np.uint8)
        
        # Calculate coverage percentage
        coverage = (np.sum(mask == 1) / mask.size) * 100
        
        return jsonify({
            "status": "success",
            "coverage": round(float(coverage), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/soil', methods=['POST'])
def predict_soil():
    """Endpoint for Soil Type Detection."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file'].read()
        img = Image.open(io.BytesIO(file))
        
        # YOLOv8 Inference
        results = soil_model(img)[0]
        
        if len(results.boxes) > 0:
            # Get the detection with the highest confidence
            box = results.boxes[0]
            class_id = int(box.cls[0])
            label = soil_model.names[class_id]
            conf = float(box.conf[0])
            
            return jsonify({
                "status": "success",
                "label": label, 
                "confidence": round(conf, 4)
            })
        
        return jsonify({
            "status": "not_detected",
            "label": "Unknown", 
            "confidence": 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    # Render will pass the correct port via the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
