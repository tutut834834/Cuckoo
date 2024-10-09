import torch
import torch.nn as nn
import torchvision.models as models

def get_model(data):
    """ Returns the appropriate model based on the dataset """
    if data == 'fmnist' or data == 'fedemnist':
        return ResNet18_FMNIST()  # Use ResNet18 for FashionMNIST
    elif data == 'cifar10':
        return ResNet18_CIFAR10()  # Use ResNet18 for CIFAR-10
    else:
        raise ValueError(f"Unsupported dataset: {data}")

class ResNet18_FMNIST(nn.Module):
    def __init__(self):
        super(ResNet18_FMNIST, self).__init__()
        # Load pre-trained ResNet-18 model
        self.model = models.resnet18(pretrained=False)
        
        # Modify the first layer to accept 1 channel (grayscale) instead of 3 (RGB)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Modify the output layer to classify 10 classes (for FashionMNIST)
        self.model.fc = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        return self.model(x)


class ResNet18_CIFAR10(nn.Module):
    def __init__(self):
        super(ResNet18_CIFAR10, self).__init__()
        # Load pre-trained ResNet-18 model
        self.model = models.resnet18(pretrained=False)
        
        # CIFAR-10 images are 3-channel RGB, so no need to modify the input layer
        # Modify the output layer to classify 10 classes (for CIFAR-10)
        self.model.fc = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        return self.model(x)


# If you want to keep CNN models as an option, you can also keep them below:

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.drop1(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x


class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 256)
        self.drop3 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.drop1(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop2(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.drop3(x)
        x = self.fc3(x)
        return x
