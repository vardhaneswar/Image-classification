import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    root="data/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="data/test",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)