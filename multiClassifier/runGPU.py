from collections import OrderedDict
import time
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
    epoch=[300],
    kfoldSplit=[10],
    lr=[0.001],
    batch_size=[512],
    num_worker=[0],
    model=[RNN],
    optim=[torch.optim.Adam],
    criterion=[torch.nn.CrossEntropyLoss],
    autocast=[False]
)

class Train:
    def __init__(self, params, trainData: torch.utils.data.Dataset):
        self.manager = RunManager(STATISTIC_PATH, TIMESTAMP)
        self.runBuilder = RunBuilder(params)
        self.trainData = trainData

    def run(self):
        for k, run in enumerate(self.runBuilder.runs):
            print(f"Run: {k}")
            print(run)
            loader = Loader(
                self.trainData, run.batch_size, run.num_worker, run.kfoldSplit
            ).get_data_loaders()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {device}")
            model = run.model().to(device)
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


# Test the train
if __name__ == "__main__":
    # data = CoordinateDataset("data/studentsdigits-modified.csv")
    # data = ReshapeCoordinateDataset("data/studentsdigits-modified.csv")
    data = ReshapeCoordinateDataset("data/studentsdigits-train.csv")
    train = Train(paramsADAM, data).run()
