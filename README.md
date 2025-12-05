🐶🐱 Cat vs Dog Image Classification (PyTorch)

A complete beginner-friendly deep learning project for classifying images as Cat or Dog using PyTorch.
This repository is structured so even a new learner can understand the workflow end-to-end.

📌 1. Project Overview

This project demonstrates:

Loading image datasets

Preprocessing using transforms

Building a CNN model

Training & evaluation

Saving and loading model weights

Predicting a single image

Anyone can follow the steps and reproduce the results.

📁 2. Folder Structure
Image-classification/
│
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
│
└── src/
    ├── dataset.py
    ├── model.py
    ├── train.py
    └── model_evaluation.py


PyTorch automatically assigns labels based on folder names:

cats → 0

dogs → 1

⚙️ 3. Installation
Step 1 — Create Conda Environment
conda create -n pytorch-env python=3.10 -y
conda activate pytorch-env

Step 2 — Install PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Step 3 — Install Pillow
pip install pillow

📥 4. Dataset Setup

Organize dataset like this:

data/
  train/
    cats/
    dogs/
  test/
    cats/
    dogs/


Each folder must contain .jpg or .png images.

🧠 5. Pipeline Explanation (Simple)
dataset.py

Loads images

Applies transforms

Creates train & test dataloaders

model.py

Builds a simple CNN

Conv → ReLU → MaxPool → Fully Connected

train.py

Runs training loop

Computes loss

Evaluates accuracy

Saves model as model.pth

model_evaluation.py

Loads saved model

Predicts a single image

Prints true vs predicted label

🚀 6. Train the Model

Run:

python -m src.train


Expected example output:

Epoch [1/3], Loss: 0.69
Epoch [2/3], Loss: 0.65
Epoch [3/3], Loss: 0.59
Test Accuracy: 65.65%
Model saved as model.pth

🔍 7. Predict a Single Image

Run:

python -m src.model_evaluation


Change the image path inside the file:

image_path = "data/test/dogs/dog01.jpg"


Example output:

Image: data/test/dogs/dog01.jpg
True Label: dogs
Predicted: dog

📈 8. Improve Accuracy (Optional)

You can improve accuracy with:

Normalization

Data augmentation

More epochs

Better CNN

GPU training

Transfer Learning (ResNet18 ~95% accuracy)

🏗 9. Future Enhancements

FastAPI prediction API

Streamlit UI

Training plots

Confusion matrix

Docker container

MLOps workflow (MLflow, DVC)

👤 10. Author

Eswar Vardhan
Beginner-friendly PyTorch image classification project.