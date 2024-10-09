import torch.nn as nn
import torchvision.models as models

def get_model(data):
    if data == 'fmnist' or data == 'fedemnist':
        return CNN_MNIST()
    elif data == 'cifar10':
        return ResNet18()  # Returning ResNet18 for CIFAR-10

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
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.drop1(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return x

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)  # Use ResNet18 model
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # Adjust output layer to 10 classes for CIFAR-10

    def forward(self, x):
        return self.model(x)
