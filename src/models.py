import torch.nn.functional as F
import torch.nn as nn

def get_model(data):
    if data == 'cifar10':
        return CNN_CIFAR()  # Use the modified CNN model for CIFAR-10

class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)  # Batch normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(256 * 2 * 2, 128)  # Adjust for CIFAR-10 image dimensions
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 2 * 2)  # Flatten
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
