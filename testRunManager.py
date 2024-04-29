import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from collections import OrderedDict

from models import MLP
from runBuilderManager import RunBuilder, RunManager

# Mock data
train_data = torch.randn(1000, 4)  # Mock training data
train_labels = torch.randint(0, 3, (1000,))  # Mock training labels
valid_data = torch.randn(200, 4)  # Mock validation data
valid_labels = torch.randint(0, 3, (200,))  # Mock validation labels

# Mock model
class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.name = "MockModel"

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Mock data loaders
train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=32, shuffle=True)
valid_loader = DataLoader(list(zip(valid_data, valid_labels)), batch_size=32, shuffle=False)

# Mock RunManager
paramsADAM = OrderedDict(
    epoch=[100],
    kfoldSplit=[10],
    lr=[0.001],
    batch_size=[256],
    num_worker=[0],
    model=[MockModel],
    optim=[torch.optim.Adam],
    criterion=[torch.nn.CrossEntropyLoss],
    autocast=[False]
)
run_builder = RunBuilder(paramsADAM)
run_manager = RunManager(statsFolderPath=".", statsFileName="test_stats")

# Mock RunManager initialization
run_manager.begin_run(run=run_builder.runs[0], model=MockModel(), trainLoader=train_loader, validLoader=valid_loader)

# Simulate training and validation epochs
for epoch in range(3):  # Simulate 3 epochs
    run_manager.begin_epoch()

    # Simulate training
    for data, labels in train_loader:
        preds = run_manager.model(data)
        loss = nn.CrossEntropyLoss()(preds, labels)
        run_manager.track_train_loss(loss)
        run_manager.track_numTrain_correct(preds, labels)

    # Simulate validation
    for data, labels in valid_loader:
        preds = run_manager.model(data)
        loss = nn.CrossEntropyLoss()(preds, labels)
        run_manager.track_valid_loss(loss)
        run_manager.track_numValid_correct(preds, labels)

    run_manager.end_epoch(checkptFolderPath=".")

# Mock RunManager end
run_manager.end_run(savePath="", save=False)

print("Mock training and validation completed.")
