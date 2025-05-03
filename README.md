
Forest Fire Detection using Deep Learning (In Progress)

This repository contains a deep learning-based project aimed at detecting forest fires using image data. The goal is to train a Convolutional Neural Network (CNN) to classify images as either fire or no fire. The project is currently **under development** as part of a Deep Learning course assessment.
Project Status

In Progress
We are currently working on:

* Finalizing the dataset
* Building and training the CNN model
* Implementing the prediction pipeline
* Creating a simple web-based demo (optional)

Overview

* Problem Statement: Early detection of forest fires from visual data.
* Approach: Use CNN-based image classification.
* Dataset* Open-source forest fire image datasets (to be finalized).

Planned Pipeline

 1. Data Collection

    * Collect images of forest fires and normal forest scenes.
    * Organize data into training, validation, and test sets.

2. Image Preprocessing

    * Resize to standard dimensions (e.g., 224x224)
    * Normalize pixel values
    * Convert to RGB or grayscale

 3. Data Augmentation

    * Rotation
    * Flip
    * Zoom
    * Brightness adjustment

 4. Model Design

    * Custom CNN or fine-tuned pre-trained model (e.g., MobileNetV2)
    * Output: Binary classification (fire / no fire)

 5. Model Training & Evaluation

    * Use metrics like accuracy, precision, recall, and F1-score
    * Visualize learning curves and confusion matrix

 6. (Optional) Web App

    * Build a simple interface using Flask or Streamlit



Requirements

    To be included once the codebase is complete.


    tensorflow
    numpy
    opencv-python
    matplotlib
    flask
    scikit-learn

How to Run (Planned)

# Train the model
python train.py

# Predict from a new image
python predict.py --image path/to/image.jpg

# Launch web app (if developed)
python app.py


