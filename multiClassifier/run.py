from collections import OrderedDict
import time
from typing import List
from sklearn.base import accuracy_score
from sklearn.metrics import classification_report
import torch
import torch.utils
import torch.utils.data
from tqdm import trange
from loader import CoordinateDataset, Loader
from models import MLP
from runBuilderManager import RunBuilder, RunManager

SAVE_MODEL_PATH = "./savedModels"
STATISTIC_PATH = "./savedStatistics"
# Format the time.local() to be used as a string for filename that tells the time
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

paramsADAM = OrderedDict(
    epoch=[500],
    kfoldSplit=[5],
    lr=[0.001],
    batch_size=[500],
    num_worker=[4],
    model=[MLP],
    optim=[torch.optim.Adam],
    criterion=[torch.nn.CrossEntropyLoss],
)

class Train:
    def __init__(self, params, trainData, testData):
        self.manager = RunManager(STATISTIC_PATH, TIMESTAMP)
        self.runBuilder = RunBuilder(params)
        self.trainData = trainData
        self.testData = testData

    def run(self):
        for k, run in enumerate(self.runBuilder.runs):
            print(f"Run: {k}")
            print(run)
            loader = Loader(
                self.trainData, run.batch_size, run.num_worker, run.kfoldSplit
            ).get_data_loaders()

            model = run.model()
            criterion = run.criterion()
            optimizer = run.optim(model.parameters(), lr=run.lr)

            for fold, (train_loader, val_loader) in loader:
                self.manager.begin_run(
                    run, model, train_loader, val_loader
                )  # This mean we track each fold run -> maybe too much
                for _ in trange(
                    run.epoch,
                    desc=f"Run {k+1}/{len(self.runBuilder)} - fold {fold}/{run.kfoldSplit}'s epoch progress",
                ):
                    self.manager.begin_epoch()
                    model.train()
                    for coordinates, labels in train_loader:
                        optimizer.zero_grad()
                        predictions = model(coordinates)
                        loss = criterion(predictions, labels)
                        loss.backward()
                        optimizer.step()
                        self.manager.track_train_loss(loss)
                        self.manager.track_numTrain_correct(predictions, labels)

                    model.eval()
                    for val_coordinates, val_labels in val_loader:
                        with torch.no_grad():
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

    def afterTrainEval(self, evalModel: torch.nn.Module, batch_size, num_workers):
        evalLoader = torch.utils.data.DataLoader(
            self.trainData, batch_size=batch_size, num_workers=num_workers
        )
        predList = []
        trueList = []
        with torch.no_grad():
            evalModel.eval()
            for coordinates, labels in evalLoader:
                predictions = evalModel(coordinates)
                predList.append(predictions)
                trueList.append(labels)
            acc = accuracy_score(trueList, predList)
            print(classification_report(trueList, predList))
        return acc

    def test(self, coordinates: List[torch.Tensor], testModel: torch.nn.Module) -> list:
        testResult = []
        with torch.no_grad():
            testModel.eval()
            for coordinate in coordinates:
                testResult.append(testModel(coordinate))
        return testResult


# Test the train
if __name__ == "__main__":
    data = CoordinateDataset("data/studentsdigits-train.csv")
    testData = CoordinateDataset("data/studentsdigits-test.csv")
    train = Train(paramsADAM, data, testData).run()
