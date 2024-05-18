from typing import List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import pandas as pd
import os

# Maybe incorporate loading into embedding

TOKEN_FOLDER = "./textInference/tokenized"

class PreconditionStatementDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, type='train') -> None:
        self.data = pd.read_csv(csv_file, header=None)
        self.dataset_type: str = type
        self.tokenized_data: List[Tuple[torch.Tensor, int]] = []

        tokenized_file = f'{TOKEN_FOLDER}/pnli_{self.dataset_type}_tokenized.pt'
        if os.path.exists(tokenized_file):
            self.tokenized_data = torch.load(tokenized_file)
        else:
            self.tokenized_data = self.tokenize_and_save(tokenized_file)

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, label = self.tokenized_data[idx]
        return torch.tensor(tokens), torch.tensor(label)

    def tokenize_and_save(self, tokenize_file) -> List[Tuple[torch.Tensor, int]]:
        tokenized_data = []
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-large-mnli')
        for _, row in tqdm(self.data.iterrows(), 'Tokenizing data', total=self.data.shape[0]):
            precondition, statement, label = row
            tokenized_data.append((tokenizer.encode(str(precondition), str(statement)), int(label)))
        torch.save(tokenized_data, tokenize_file)
        return tokenized_data

def custom_collate_fn(batch):
    tokens_batch, labels_batch = zip(*batch)
    max_length = max([len(tokens) for tokens in tokens_batch])
    tokens_batch = [torch.nn.functional.pad(tokens, (0, max_length - len(tokens))) for tokens in tokens_batch]
    return torch.stack(tokens_batch), torch.tensor(labels_batch)

# Use the dataset
# dataset = PreconditionStatementDataset("./data/pnli_train.csv")
# print(dataset[0])
# print(len(dataset[0][0]))
# print(dataset[1])
# print(len(dataset[1][0]))
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=0, collate_fn=custom_collate_fn)
# for i, (inputs, labels) in enumerate(dataloader):
#     print(f'Inputs: {inputs} and labels: {labels}')
#     if i == 0:
#         break
# TODO HAVE TO USE PYTHON lower than 3.11

# things = PreconditionStatementDataset("./data/pnli_dev.csv")
# print(things.data.head())
# print(len(things))
# print(things[0])
# dataloader = DataLoader(things, 1, 0)
# for i, (precondition, statement, label) in enumerate(dataloader):
#     print(f'Precondition: {precondition} and statement: {statement} and label: {label}')
#     print(f'Precondition type: {type(precondition)} and statement type: {type(statement)} and label type: {type(label)}')
#     print(f'Precondition: {precondition[0]} and statement: {statement[0]} and label: {label.item()}')
#     if i == 0:
#         break

# roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
# roberta.register_classification_head('mnli', num_classes=2)
# roberta.eval()
# #
# tokens = roberta.encode('Sometimes do exercise.', 'A person typically desire healthy life.')
# res1 = roberta.predict('mnli', tokens).argmax()
# print(f'result 1: {res1}')
#
# tokens = roberta.encode('Who eats junk foods.', 'A person typically desire healthy life.')
# res2 = roberta.predict('mnli', tokens).argmax()
# print(f'result 2: {res2}')

# Try run
# params = OrderedDict(
#     epoch=[1],
#     batch_size=[1],
#     num_worker=[0],
#     # optim=[torch.optim.Adam],
#     criterion=[torch.nn.CrossEntropyLoss],
#     # autocast=[False],
# )
# 
# SAVE_MODEL_PATH = "./savedModels"
# STATISTIC_PATH = "./savedStatistics"
# TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
# manager = RunManager(STATISTIC_PATH, TIMESTAMP)
# runBuilder = RunBuilder(params)
# 
# for k, run in enumerate(runBuilder.runs):
#     print(f"Run: {k}")
#     print(run)
# 
#     loader = DataLoader(things, run.batch_size, run.num_worker)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 
#     model = torch.hub.load("pytorch/fairseq", "roberta.large.mnli")
#     model.register_classification_head("mnli", num_classes=2)
#     model.eval()
#     criterion = run.criterion()
#     model.to(device)
# 
# 
#     total_loss = 0
#     total_correct = 0
#     with torch.no_grad():
#         for precondition_tup, statement_tup, label_tensor in loader:
#             precondition, statement = precondition_tup[0], statement_tup[0]
#             tokens = model.encode(precondition, statement)
#             res = model.predict("mnli", tokens)
# 
#             # print(f'Shape of res {res.shape}, type of res {type(res)}')
#             # break
#             loss = criterion(res, label_tensor)
#             total_loss += loss.item()
#             total_correct += res.argmax(dim=1).eq(label_tensor).sum().item()
#     
#     accuracy = total_correct / len(loader.dataset)
#     print(f"Accuracy: {accuracy}")