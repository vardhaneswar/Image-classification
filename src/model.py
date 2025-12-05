import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 1st convolution layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        # 2nd convolution layer
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        # 3rd convolution layer
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # fully connected layer (final output)
        self.fc = nn.Linear(64 * 28 * 28, 2)  # 2 classes: cat and dog

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = SimpleCNN()
    x = torch.randn(1, 3, 224, 224)  # fake image
    out = model(x)
    print("Output shape:", out.shape)
