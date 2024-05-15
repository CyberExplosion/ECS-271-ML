from collections import OrderedDict
import time
from numpy import ndarray
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.utils
import torch.utils.data
from tqdm import trange
from loader import CoordinateDataset, Loader, ReshapeCoordinateDataset
from models import CNN, MLP, RNN
from runBuilderManager import RunBuilder, RunManager


SAVE_MODEL_PATH = "./savedModels"
STATISTIC_PATH = "./savedStatistics"
# Format the time.local() to be used as a string for filename that tells the time
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

paramsADAM = OrderedDict(
    epoch=[100],
    kfoldSplit=[10],
    lr=[0.001],
    batch_size=[256],
    num_worker=[0],
    model=[MLP],
    optim=[torch.optim.Adam],
    criterion=[torch.nn.CrossEntropyLoss],
    autocast=[False]
)
# paramsADAM = OrderedDict(
#     epoch=[100, 200],
#     kfoldSplit=[5, 10],
#     lr=[0.001, 0.01],
#     batch_size=[512],
#     num_worker=[8],
#     model=[MLP],
#     optim=[torch.optim.Adam],
#     criterion=[torch.nn.CrossEntropyLoss],
#     autocast=[False]
# )


class Train:
    def __init__(self, params, trainData: torch.utils.data.Dataset, valData: torch.utils.data.Dataset):
        self.manager = RunManager(STATISTIC_PATH, TIMESTAMP, earlyStop=False)
        self.runBuilder = RunBuilder(params)
        self.trainData = trainData
        self.valData = valData

    def run(self):
        for k, run in enumerate(self.runBuilder.runs):
            print(f"Run: {k}")
            print(run)
            # loader = Loader(
            #     self.trainData, run.batch_size, run.num_worker, run.kfoldSplit
            # ).get_data_loaders()
            train_loader = torch.utils.data.DataLoader(self.trainData, batch_size=run.batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(self.valData, batch_size=run.batch_size, shuffle=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {device}")
            model = run.model().to(device)
            criterion = run.criterion()
            optimizer = run.optim(model.parameters(), lr=run.lr)

            self.manager.begin_run(
                run, model, train_loader, val_loader
            )  # This mean we track each fold run -> maybe too much

            for _ in trange(
                run.epoch,
                desc=f"Run {k+1}/{len(self.runBuilder)}'s epoch progress",
            ):
                self.manager.begin_epoch()
                model.train()
                for coordinates, labels in train_loader:
                    coordinates, labels = coordinates.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=run.autocast):
                        predictions = model(coordinates)
                        loss = criterion(predictions, labels)
                    loss.backward()
                    optimizer.step()
                    self.manager.track_train_loss(loss)
                    self.manager.track_numTrain_correct(predictions, labels)

                model.eval()
                for val_coordinates, val_labels in val_loader:
                    with torch.no_grad():
                        val_coordinates, val_labels = (
                            val_coordinates.to(device),
                            val_labels.to(device),
                        )
                        with torch.cuda.amp.autocast(enabled=run.autocast):
                            val_predictions = model(val_coordinates)
                            val_loss = criterion(val_predictions, val_labels)
                        self.manager.track_valid_loss(val_loss)
                        self.manager.track_numValid_correct(
                            val_predictions, val_labels
                        )

                self.manager.end_epoch(SAVE_MODEL_PATH)
                if self.manager.stop:
                    break
                self.manager.end_run()


class IrisDataset(torch.utils.data.Dataset):
    def __init__(self, X: ndarray, y: ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Test the train
if __name__ == "__main__":
    # data = CoordinateDataset("data/studentsdigits-modified.csv")
    # data = ReshapeCoordinateDataset("data/studentsdigits-modified.csv")

    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    trainData = IrisDataset(X_train, y_train)
    valData = IrisDataset(X_test, y_test)
    print(f"Length of trainData: {len(trainData)}")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f'First data: {trainData[0]}')

    train = Train(paramsADAM, trainData, valData).run()
