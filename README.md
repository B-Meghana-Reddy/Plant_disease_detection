# Plant Disease Detection System

## Overview
The Plant Disease Detection System is a machine learning-based solution designed to identify diseases in plants from images of their leaves. By leveraging advanced image processing techniques and machine learning models, this system provides accurate disease classification, helping farmers and gardeners take timely action to protect their crops.

## Features
- **Disease Detection:** Identify diseases from leaf images with high accuracy.
- **User-Friendly Interface:** Simple and intuitive web application built using Streamlit.
- **Real-Time Predictions:** Upload an image and get instant results.
- **Multi-Class Support:** Detects multiple plant diseases across various plant species.

## Tech Stack
### Machine Learning
- **Google Colab:** For training and testing the ML model.
- **TensorFlow/Keras:** To build and train the neural network.
- **OpenCV:** For image preprocessing and augmentation.

### Web Application
- **Streamlit:** For building the user interface.
- **Python:** Backend logic and API integration.

## Dataset
The model is trained on a comprehensive plant disease dataset containing labeled images of healthy and diseased leaves across various plant species.

Dataset source: [Kaggle Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## Model Architecture
- **Convolutional Neural Network (CNN):**
  - Input Layer: Resized images (e.g., 128x128 pixels).
  - Hidden Layers: Multiple convolutional and pooling layers for feature extraction.
  - Output Layer: Fully connected layers with softmax activation for multi-class classification.

## Results
The model achieved an accuracy of **96%** on the train dataset. 

## Future Enhancements
- Expand dataset to include more plant species and diseases.
- Improve the web application's UI/UX.
- Integrate real-time data collection through mobile apps.
- Provide localized treatment suggestions.
