import torch.nn.functional as F
import torch.nn as nn

def get_model(data):
    if data == 'fmnist' or data == 'fedemnist':
        return CNN_MNIST()  # Use CNN model for Fashion-MNIST or FedEMNIST
    elif data == 'cifar10':
        return CNN_CIFAR()  # Use CNN model for CIFAR-10
               

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))  # For grayscale images (1 channel)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(9216, 128)  # 9216 comes from flattening the feature map after pooling
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])  # Flatten the feature map
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x


class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)  # For RGB images (3 channels)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(256 * 2 * 2, 128)  # Adjusting for CIFAR image dimensions
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 256)
        self.drop3 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(256, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 2 * 2)  # Flatten the feature map for CIFAR-10 images
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = self.fc3(x)
        return x
