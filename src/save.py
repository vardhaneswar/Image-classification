from src.model import SimpleCNN
import torch

model = SimpleCNN()
torch.save(model.state_dict(), "model.pth")
print("Model saved!")
