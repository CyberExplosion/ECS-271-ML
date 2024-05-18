from collections import namedtuple
from itertools import product
from pathlib import Path
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from EarlyStopping import EarlyStopping


class RunBuilder:
    def __init__(self, params) -> None:
        self.runs = self._get_runs(params)

    def __len__(self):
        return len(self.runs)

    def _get_runs(self, params):
        Run = namedtuple("Run", params.keys())

        runs = []
        for v in product(*params.values()):
            # print(f"value {v} and {Run(*v)}")
            runs.append(Run(*v))
        return runs


class RunManager:
    def __init__(self, statsFolderPath, statsFileName, earlyStop=True):
        self.epoch_count = 0
        self.epoch_train_loss = 0
        self.epoch_valid_loss = 0
        self.epoch_numTrain_correct = 0
        self.epoch_numValid_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.model = None
        self.train_loader = None
        self.valid_loader = None
        self.tb = None

        self.useEarlyStop = earlyStop
        self.earlyStop = None
        self.stop = False

        self.statsFolderPath = Path(statsFolderPath)
        Path.mkdir(self.statsFolderPath, exist_ok=True, parents=True)
        self.statsFileCSV = Path(self.statsFolderPath, f"{statsFileName}.csv")
        self.errorPath = Path(statsFolderPath, "error.txt")
        open(self.errorPath, "w")  # restart the error file

    def begin_run(self, run, model, trainLoader, validLoader):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.model = model
        self.train_loader = trainLoader
        self.valid_loader = validLoader
        comment = f"-{ {k: v if k != 'model' else v.__class__.__name__ for k,v in run._asdict().items()} }"
        self.tb = SummaryWriter(comment=self.sanitize_param_name(comment))

        if self.run_count == 1:
            self.tb.add_graph(self.model, next(iter(self.train_loader))[0].to(self.model.device), use_strict_trace=False)
        # images, labels = next(iter(self.train_loader))
        # grid = torchvision.utils.make_grid(images)

        # self.tb.add_image("images", grid)  # Add images and graph when begin one run
        self.earlyStop = EarlyStopping(patience=15)
        self.stop = False

    def end_run(self, savePath="", save=False):
        self.tb.close()
        self.epoch_count = 0
        self.writeToCSV()
        if save:
            self.saveModel(savePath, result=self.run_data[-1])

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_train_loss = 0
        self.epoch_numTrain_correct = 0
        self.epoch_valid_loss = 0
        self.epoch_numValid_correct = 0

    def end_epoch(self, checkptFolderPath):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        train_loss = self.epoch_train_loss / len(self.train_loader.dataset)
        train_accuracy = self.epoch_numTrain_correct / len(self.train_loader.dataset)

        valid_loss = self.epoch_valid_loss / len(self.valid_loader.dataset)
        valid_accuracy = self.epoch_numValid_correct / len(self.valid_loader.dataset)

        self.tb.add_scalars(
            "Loss", {"trainLoss": train_loss, "validLoss": valid_loss}, self.epoch_count
        )
        self.tb.add_scalars(
            "Accuracy",
            {"trainAcc": train_accuracy, "validAcc": valid_accuracy},
            self.epoch_count,
        )  # Add scalar is use when at the end of epoch

        for name, param in self.model.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            if param.grad != None:
                self.tb.add_histogram(f"{name}.grad", param.grad, self.epoch_count)

        results = {}
        results["run"] = self.run_count
        results['model name'] = self.run_params.model.__class__.__name__
        results["epoch"] = self.epoch_count
        results["train loss"] = train_loss
        results["valid loss"] = valid_loss
        results["train accuracy"] = train_accuracy
        results["valid accuracy"] = valid_accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        for k, v in self.run_params._asdict().items():
            if (k != 'model'):
                results[k] = v  # Add the hyperparameter to words to easy report
        self.run_data.append(results)
        if self.useEarlyStop:
            self.checkEarlyStop(
                valid_loss, self.model, checkptFolderPath, self.run_count
            )
            if self.earlyStop.early_stop:
                self.stop = True

        # print(f'Current run data: {self.run_data}')

    def track_train_loss(self, loss):
        self.epoch_train_loss += loss.item() * self.train_loader.batch_size

    def track_valid_loss(self, loss):
        self.epoch_valid_loss += loss.item() * self.valid_loader.batch_size

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def track_numTrain_correct(self, preds, labels):
        self.epoch_numTrain_correct += self._get_num_correct(preds, labels)

    def track_numValid_correct(self, preds, labels):
        self.epoch_numValid_correct += self._get_num_correct(preds, labels)

    def checkEarlyStop(self, valLoss, model, folderPath, run):
        fPath = Path(folderPath)
        Path.mkdir(fPath, exist_ok=True, parents=True)
        filePath = Path(fPath / f"earlyStop_run_{run}.pt")

        self.earlyStop(valLoss, model, filePath)

    def saveModel(self, pathName, result):
        moduleName = ""
        for k, v in result.items():
            if (k == 'epoch' or k == 'run' or k == 'epoch' or k == 'model name' or k == 'train loss' or k == 'valid loss' or k == 'train accuracy' or k == 'valid accuracy'):
                moduleName += f"_{k}:{v}"
        folderPath = Path(pathName)
        Path.mkdir(folderPath, exist_ok=True, parents=True)
        filePath = Path(folderPath / f"{self.sanitize_param_name(moduleName)}.pt")
        torch.save(self.model.state_dict(), filePath)

    def writeError(self, msg=""):
        with open(self.errorPath, "a") as f:
            f.write(
                f"Error at runs: {self.run_count}\nParameters: {self.run_params}\nAdditional msg: {msg}"
            )

    def writeToCSV(self):
        oldStatsDF = None
        # print(f"The stats file is {self.statsFileCSV}")
        try:
            with open(self.statsFileCSV, "r") as f:
                oldStatsDF = pd.read_csv(f)
                oldStatsDF = pd.concat(
                    [
                        oldStatsDF,
                        pd.DataFrame.from_records(self.run_data[-1], index=[0]),
                    ]
                )
        except FileNotFoundError:
            oldStatsDF = pd.DataFrame.from_dict(self.run_data)

        try:
            # Allow open file in create mode and write the new data
            with open(self.statsFileCSV, "w", newline="") as f:
                oldStatsDF.to_csv(f, index=False)
        except Exception as e:
            print(f"Error in writeToCSV: {e}")
            self.writeError(f"Error in writeToCSV: {e}")

    def sanitize_param_name(self, param_name):
        return (
            param_name.replace("(", "")
            .replace(")", "")
            .replace(",", "_")
            .replace(" ", "_")
            .replace("<", "")
            .replace(">", "")
            .replace("'", "")
            .replace(":", "")
        )
