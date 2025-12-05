import torch
from PIL import Image
from torchvision import transforms
from src.model import SimpleCNN
import os

device = torch.device("cpu")

# Load model
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict image
def predict_image(image_path):
    model = load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor.to(device))
        _, pred = torch.max(outputs, 1)

    classes = ["cat", "dog"]
    return classes[pred.item()]

# SINGLE simple test
if __name__ == "__main__":
    # pick a file from test set
    image_path = "data/test/dogs/dog.4085.jpg"  # <-- change this to any test image

    prediction = predict_image(image_path)

    # true label from folder name
    true_label = os.path.basename(os.path.dirname(image_path))  # 'cats' or 'dogs'

    print("Image:", image_path)
    print("True Label:", true_label)
    print("Predicted:", prediction)
