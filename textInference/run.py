from collections import OrderedDict
import time
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup
from runManager import RunBuilder, RunManager
from loader import PreconditionStatementDataset, custom_collate_fn

# TODO: Lower batch size now to 16. Prev was 32. Training got to pretty good val now
# TRY MSE LOSS with scaling??

torch.manual_seed(0)
SAVE_MODEL_PATH = "./textInference/savedModels"
STATISTIC_PATH = "./textInference/savedStatistics"
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

params = OrderedDict(
    epoch=[100],
    lr=[0.000001],
    model=[RobertaForSequenceClassification.from_pretrained('FacebookAI/roberta-large-mnli', num_labels=2, ignore_mismatched_sizes=True)],
    optim=[torch.optim.Adam],
    criterion=[torch.nn.CrossEntropyLoss],
    batch_size=[128],
    num_worker=[0],
    num_layers_to_unfreeze=[8]
)

class Train:
    def freeze_roberta_layers_modified_layers(self, model, num_layers_to_unfreeze=0):
        # Freeze the embedding layer
        for layer in model.roberta.embeddings.parameters():
            layer.requires_grad = False
        model.roberta.embeddings.eval()   # Set to eval mode to avoid BatchNorm and Dropout layers to update their running stats
        # Freeze the encoder layers
        total_layers = len(model.roberta.encoder.layer)
        for i, layers in enumerate(model.roberta.encoder.layer):
            if i < total_layers - num_layers_to_unfreeze:
                for params in layers.parameters():
                    params.requires_grad = False
                layers.eval()   # Set to eval mode to avoid BatchNorm and Dropout layers to update their running stats
            else:
                layers.requires_grad = True

        # unfreeze classifier layer
        for params in model.classifier.parameters():
            params.requires_grad = True
    
    def __init__(self, params, trainData: Dataset, devData: Dataset) -> None:
        self.manager = RunManager(STATISTIC_PATH, TIMESTAMP)
        self.runBuilder = RunBuilder(params)
        self.trainData = trainData
        self.devData = devData
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler()

    def run(self):
        for k, run in enumerate(self.runBuilder.runs):
            model = run.model.to(self.device)
            train_loader = DataLoader(self.trainData, run.batch_size, run.num_worker, collate_fn=custom_collate_fn)
            dev_loader = DataLoader(self.devData, run.batch_size, run.num_worker, collate_fn=custom_collate_fn)
            criterion = run.criterion()
            optimizer = run.optim(model.parameters(), lr=run.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)

            total_steps = len(train_loader) * run.epoch
            warmup_steps = int(0.1 * total_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

            self.manager.begin_run(run, run.model, train_loader, dev_loader)
            # for epoch in range(run.epoch):
            #     print(f"Run {k+1}/{len(self.runBuilder)} - Epoch {epoch+1}/{run.epoch}")
            for _ in trange(run.epoch, desc=f"Run {k+1}/{len(self.runBuilder)}'s epoch progress"):
                self.manager.begin_epoch()
                model.train()   # Need to set before modify the layers inside the model
                self.freeze_roberta_layers_modified_layers(model, run.num_layers_to_unfreeze) # freeze some layers
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs).logits
                        loss = criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    scheduler.step()
                    self.manager.track_train_loss(loss)
                    self.manager.track_numTrain_correct(outputs, labels)
                model.eval()
                with torch.no_grad():
                    for dev_inputs, dev_labels in dev_loader:
                    # for _, (dev_inputs, dev_labels) in enumerate(tqdm(dev_loader, desc="Evaluation progress")):
                        dev_inputs, dev_labels = dev_inputs.to(self.device), dev_labels.to(self.device)
                        with torch.cuda.amp.autocast():
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