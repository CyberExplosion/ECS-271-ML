import torch.nn as nn

# MLP but as a class inherit from nn.Module
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.fc1 = nn.Linear(in_features=8, out_features=32)
        self.fc1 = nn.Linear(in_features=4, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=3)
        # self.fc3 = nn.Linear(in_features=16, out_features=10)
        self.relu = nn.ReLU()
        self.name = "MLP"

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

# a simple shallow cnn model
# the image would be a 2x4 image
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=2)
        self.fc1 = nn.Linear(in_features=12 * 2 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.fc3 = nn.Linear(in_features=60, out_features=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.name = "CNN"

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 12 * 2 * 4)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# A simple shallow rnn model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=2, hidden_size=16, num_layers=3, batch_first=True)
        self.fc = nn.Linear(in_features=16, out_features=10)
        self.name = "RNN"

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x