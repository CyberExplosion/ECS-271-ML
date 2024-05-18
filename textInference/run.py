from collections import OrderedDict
import time
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from transformers import RobertaForSequenceClassification
from runManager import RunBuilder, RunManager
from loader import PreconditionStatementDataset, custom_collate_fn

torch.manual_seed(0)
SAVE_MODEL_PATH = "./textInference/savedModels"
STATISTIC_PATH = "./textInference/savedStatistics"
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

params = OrderedDict(
    epoch=[50, 100],
    lr=[0.001],
    model=[RobertaForSequenceClassification.from_pretrained('FacebookAI/roberta-large-mnli')],
    optim=[torch.optim.Adam],
    criterion=[torch.nn.CrossEntropyLoss],
    batch_size=[1024],
    num_worker=[0],
)

class Train:
    def freeze_roberta_layers(self, model):
        for param in model.roberta.parameters():
            param.requires_grad = False
    
    def __init__(self, params, trainData: Dataset, devData: Dataset) -> None:
        self.manager = RunManager(STATISTIC_PATH, TIMESTAMP)
        self.runBuilder = RunBuilder(params)
        self.trainData = trainData
        self.devData = devData
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        for k, run in enumerate(self.runBuilder.runs):
            model = run.model.to(self.device)
            self.freeze_roberta_layers(model)
            train_loader = DataLoader(self.trainData, run.batch_size, run.num_worker, collate_fn=custom_collate_fn)
            dev_loader = DataLoader(self.devData, run.batch_size, run.num_worker, collate_fn=custom_collate_fn)
            criterion = run.criterion()
            optimizer = run.optim(model.parameters(), lr=run.lr)

            self.manager.begin_run(run, run.model, train_loader, dev_loader)
            # for epoch in range(run.epoch):
            #     print(f"Run {k+1}/{len(self.runBuilder)} - Epoch {epoch+1}/{run.epoch}")
            for _ in trange(run.epoch, desc=f"Run {k+1}/{len(self.runBuilder)}'s epoch progress"):
                self.manager.begin_epoch()
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs).logits
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    self.manager.track_train_loss(loss)
                    self.manager.track_numTrain_correct(outputs, labels)
                model.eval()
                with torch.no_grad():
                    for dev_inputs, dev_labels in dev_loader:
                    # for _, (dev_inputs, dev_labels) in enumerate(tqdm(dev_loader, desc="Evaluation progress")):
                        dev_inputs, dev_labels = dev_inputs.to(self.device), dev_labels.to(self.device)
                        dev_outputs = model(dev_inputs).logits
                        dev_loss = criterion(dev_outputs, dev_labels)
                        self.manager.track_valid_loss(dev_loss)
                        self.manager.track_numValid_correct(dev_outputs, dev_labels)
                self.manager.end_epoch(SAVE_MODEL_PATH)
                if self.manager.stop:
                    break
            self.manager.end_run(SAVE_MODEL_PATH, save=True)

# Test the train
if __name__ == "__main__":
    trainData = PreconditionStatementDataset('./data/pnli_train.csv')
    devData = PreconditionStatementDataset('./data/pnli_dev.csv', 'dev')
    train = Train(params, trainData, devData).run()