# 🐶🐱 Cat vs Dog Image Classification (PyTorch)

A complete beginner-friendly deep learning project for classifying images as **Cat** or **Dog** using PyTorch. This repository is structured so even a new learner can understand the workflow end-to-end.

---

## 📌 Project Overview

This project demonstrates:

- Loading image datasets
- Preprocessing using transforms
- Building a CNN model
- Training & evaluation
- Saving and loading model weights
- Predicting a single image

Anyone can follow the steps and reproduce the results.

---

## 📁 Folder Structure

```
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
```

> **Note:** PyTorch automatically assigns labels based on folder names:
> - `cats` → 0
> - `dogs` → 1

---

## ⚙️ Installation

### Step 1 — Create Conda Environment

```bash
conda create -n pytorch-env python=3.10 -y
conda activate pytorch-env
```

### Step 2 — Install PyTorch (CPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3 — Install Pillow

```bash
pip install pillow
```

---

## 📥 Dataset Setup

Organize your dataset like this:

```
data/
  train/
    cats/
    dogs/
  test/
    cats/
    dogs/
```

Each folder must contain `.jpg` or `.png` images.

---

## 🧠 Pipeline Explanation

| File | Description |
|------|-------------|
| `dataset.py` | Loads images, applies transforms, creates train & test dataloaders |
| `model.py` | Builds a simple CNN: Conv → ReLU → MaxPool → Fully Connected |
| `train.py` | Runs training loop, computes loss, evaluates accuracy, saves model as `model.pth` |
| `model_evaluation.py` | Loads saved model, predicts a single image, prints true vs predicted label |

---

## 🚀 Train the Model

Run the following command:

```bash
python -m src.train
```

**Expected output:**

```
Epoch [1/3], Loss: 0.69
Epoch [2/3], Loss: 0.65
Epoch [3/3], Loss: 0.59
Test Accuracy: 65.65%
Model saved as model.pth
```

---

## 🔍 Predict a Single Image

Run:

```bash
python -m src.model_evaluation
```

Change the image path inside the file:

```python
image_path = "data/test/dogs/dog01.jpg"
```

**Example output:**

```
Image: data/test/dogs/dog01.jpg
True Label: dogs
Predicted: dog
```

---

## 📈 Improve Accuracy (Optional)

You can improve accuracy with:

| Technique | Benefit |
|-----------|---------|
| Normalization | Stabilizes training |
| Data Augmentation | Increases dataset diversity |
| More Epochs | Longer training time |
| Better CNN Architecture | More expressive model |
| GPU Training | Faster computation |
| Transfer Learning (ResNet18) | ~95% accuracy |

---

## 🏗 Future Enhancements

- [ ] FastAPI prediction API
- [ ] Streamlit UI
- [ ] Training plots (loss/accuracy curves)
- [ ] Confusion matrix visualization
- [ ] Docker container
- [ ] MLOps workflow (MLflow, DVC)

---

## 👤 Author

**Eswar Vardhan**

Beginner-friendly PyTorch image classification project.

---

## 📄 License

This project is open source and available for learning purposes.