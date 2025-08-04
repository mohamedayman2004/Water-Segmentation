# 🌊 Water Body Segmentation using U-Net + ResNet

This project is part of my internship at **Cellula Technologies**, completed during **Week 5**.  
It focuses on segmenting water bodies from satellite images using a deep learning model and deploying the solution using Flask.

## 📌 Project Overview

- **Task:** Semantic Segmentation of Water Bodies
- **Model:** U-Net with ResNet Backbone (custom input for 12-channel TIFF images)
- **Accuracy Achieved:** 93%
- **Deployment:** Flask web application

## 🧠 Model Details

- Input shape: `(128, 128, 12)` for multi-band TIFF satellite imagery.
- Architecture: 
  - ResNet-based encoder (converted to accept 12 bands)
  - U-Net decoder for pixel-wise segmentation
- Output: Binary mask identifying water bodies.

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Flask
- NumPy / Pandas
- tifffile / Pillow (PIL)
- Matplotlib
- Scikit-learn

## 🖼️ Sample Results

> Add screenshots or examples here (before & after segmentation)

## 🚀 Deployment

The trained model is deployed as a Flask app with an upload interface.  
Users can submit `.tif` satellite images and receive water-body segmentation results in real-time.


# Run the Flask app
python app.py
