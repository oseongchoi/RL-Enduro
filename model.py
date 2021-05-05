import torch
from torch import nn


class Model(nn.Module):

    def __init__(self, n_action):
        """
        Build and initialize the network.
        """
        super(Model, self).__init__()

        # Convolution layers.
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # Fully connected layers.
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, n_action)

        # Hleper layers.
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Forward propagation.
        """
        # Normalize
        x = x / 255.0
        # Conv layer 1
        x = self.conv1(x)
        x = self.relu(x)
        # Conv layer 2
        x = self.conv2(x)
        x = self.relu(x)
        # Conv layer 3
        x = self.conv3(x)
        x = self.relu(x)
        # Flatten
        x = self.flatten(x)
        # FC layer 1
        x = self.fc1(x)
        x = self.relu(x)
        # FC layer 2
        x = self.fc2(x)
        return x

    def predict(self, x, with_value=False):
        """
        A special method for predicting the best next action.
        """
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(x, device='cuda')
            x = torch.unsqueeze(x, 0)
            x = self.forward(x)
            x = torch.argmax(x)
            x = x.cpu().numpy()
        return x


class VanilaModel(nn.Module):

    def __init__(self, n_action):
        """
        Build and initialize the network.
        """
        super(Model, self).__init__()

        # Convolution layers.
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)

        # Fully connected layers.
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, n_action)

        # Hleper layers.
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Forward propagation.
        """
        # Normalize
        x = x / 255.0
        # Conv layer 1
        x = self.conv1(x)
        x = self.relu(x)
        # Conv layer 2
        x = self.conv2(x)
        x = self.relu(x)
        # Flatten
        x = self.flatten(x)
        # FC layer 1
        x = self.fc1(x)
        x = self.relu(x)
        # FC layer 2
        x = self.fc2(x)
        return x

    def predict(self, x):
        """
        A special method for predicting the best next action.
        """
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(x, device='cuda')
            x = torch.unsqueeze(x, 0)
            x = self.forward(x)
            x = torch.argmax(x)
            x = x.cpu().numpy()
        return x
