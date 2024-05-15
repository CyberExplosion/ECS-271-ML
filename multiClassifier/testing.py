import csv
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import KFold
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange

###  Load data


def read_csv(file):
    with open(file, newline="") as f:
        reader = csv.reader(f)
        line_count = 0
        rows = []
        for row in reader:
            if line_count == 0:
                titles = row
            else:
                rows.append(row)
            line_count += 1
    rows_int = np.array([[int(r) for r in row] for row in rows])
    return titles, rows_int


titles, rows_train = read_csv("data/studentsdigits-train.csv")
assert titles[-1] == "Digit" and len(titles) == 9, "Not train set"
X_train = rows_train[:, 0 : len(titles) - 1]
Y_train = rows_train[:, -1]
print(X_train.shape)
print(Y_train.shape)

titles, rows_test = read_csv("data/studentsdigits-test.csv")
assert len(titles) == 8, "Not test set"
X_test = rows_test
print(X_test.shape)

# Build the MLP model
# TODO: Write summary writer

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        torch.manual_seed(0)  # reproducibility

        self.layers = nn.ModuleDict(
            OrderedDict(
                [
                    ("fc_1", nn.Linear(in_features=8, out_features=32)),
                    ("relu_1", nn.ReLU()),
                    ("fc_2", nn.Linear(in_features=32, out_features=16)),
                    ("relu_2", nn.ReLU()),
                    ("fc_3", nn.Linear(in_features=16, out_features=10)),
                    ("relu_3", nn.ReLU()),
                ]
            )
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.lossFunction = nn.CrossEntropyLoss()

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def train(self, X_train, Y_train):
        self.training = True

        kf = KFold(n_splits=5, shuffle=False)
        # k-fold cross validation
        for fold, (train_index, test_index) in enumerate(
            tqdm(kf.split(X_train), total=kf.get_n_splits())
        ):
            x_train_fold, x_evaluate_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_evaluate_fold = Y_train[train_index], Y_train[test_index]

            for epoch in trange(1000, desc=f"Fold {fold+1}", leave=False):
                self.optimizer.zero_grad()
                fold_pred = self.forward(torch.FloatTensor(np.array(x_train_fold)))
                fold_true = torch.LongTensor(np.array(y_train_fold))
                loss = self.lossFunction(fold_pred, fold_true)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Evaluate using this fold
            fold_evaluate_pred = self.forward(
                torch.FloatTensor(np.array(x_evaluate_fold))
            )
            fold_evaluate_true = torch.LongTensor(np.array(y_evaluate_fold))

            # Metrics
            acc = accuracy_score(fold_evaluate_true, fold_evaluate_pred.argmax(dim=1))
            lossItem = loss.item()
            print(f"Epoch: {epoch}, Loss: {lossItem}, Accuracy: {acc}")


# Train model
model = MLP()
model.train(X_train, Y_train)
